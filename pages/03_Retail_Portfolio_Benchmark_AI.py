# pages/03_Retail_Portfolio_Heatmap.py
import os, sys
from datetime import datetime
import pytz
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------- Imports / mapping ----------
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))
from shop_mapping import SHOP_NAME_MAP
from helpers_normalize import normalize_vemcount_response

st.set_page_config(page_title="Retail Portfolio Heatmap", page_icon="🌍", layout="wide")
st.title("🌍 Retail Portfolio Heatmap")

API_URL = st.secrets["API_URL"]

# ---------- Styling ----------
PFM_RED    = "#F04438"
PFM_GREEN  = "#22C55E"
PFM_PURPLE = "#6C4EE3"
PFM_GRAY   = "#6B7280"

st.markdown(f"""
<style>
.kpi {{ border:1px solid #eee; border-radius:14px; padding:16px; }}
.kpi .t {{ font-weight:600; color:#0C111D; }}
.kpi .v {{ font-size:38px; font-weight:800; }}
.badge {{ font-size:13px; font-weight:700; padding:4px 10px; border-radius:999px; display:inline-block; }}
.badge.up {{ color:{PFM_GREEN}; background: rgba(34,197,94,.10); }}
.badge.down {{ color:{PFM_RED}; background: rgba(240,68,56,.10); }}
.badge.flat {{ color:{PFM_GRAY}; background: rgba(107,114,128,.10); }}
.box {{ border:1px dashed #ddd; border-radius:12px; padding:14px; background:#FAFAFC; }}
.box h4 {{ margin:0 0 8px 0; }}
</style>
""", unsafe_allow_html=True)

# ---------- Inputs ----------
PERIODS = ["this_year","last_year","this_quarter","last_quarter"]
period = st.selectbox("Periode", PERIODS, index=0)

# ---------- Helpers ----------
TZ = pytz.timezone("Europe/Amsterdam")
TODAY = datetime.now(TZ).date()
ALL_IDS = list(SHOP_NAME_MAP.keys())
METRICS = ["count_in","conversion_rate","turnover","sales_per_visitor","sq_meter"]

def post_report(params):
    r = requests.post(API_URL, params=params, timeout=45)
    r.raise_for_status()
    return r

def add_effective_date(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    ts = pd.to_datetime(d.get("timestamp"), errors="coerce")
    d["date_eff"] = pd.to_datetime(d.get("date", ts), errors="coerce").dt.date
    return d

def fetch_df(shop_ids, period, step, metrics):
    params = [("data", sid) for sid in shop_ids]
    params += [("data_output", m) for m in metrics]
    params += [("source","shops"), ("period", period), ("step", step)]
    r = post_report(params)
    js = r.json()
    df = normalize_vemcount_response(js, SHOP_NAME_MAP, kpi_keys=metrics)
    return add_effective_date(df)

def weighted_avg(series, weights):
    w = weights.fillna(0.0); s = series.fillna(0.0)
    d = w.sum()
    return (s*w).sum()/d if d else np.nan

def fmt_eur0(x): return f"€{x:,.0f}".replace(",", ".")
def fmt_eur2(x): return f"€{x:,.2f}".replace(",", ".")
def fmt_pct2(x): return f"{x:.2f}%"

# ---------- Data ophalen ----------
df = fetch_df(ALL_IDS, period, "month", METRICS)

if df.empty:
    st.warning("Geen data gevonden voor deze periode.")
    st.stop()

# ---------- Aggregatie per winkel ----------
def agg_store(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(["shop_id","date_eff"], as_index=False).agg({
        "count_in":"sum","turnover":"sum"
    })
    w = d.groupby(["shop_id","date_eff"]).apply(lambda x: pd.Series({
        "conversion_rate": weighted_avg(x["conversion_rate"], x["count_in"]),
        "sales_per_visitor": weighted_avg(x["sales_per_visitor"], x["count_in"]),
    })).reset_index()
    g = g.merge(w, on=["shop_id","date_eff"], how="left")
    sqm = (d.groupby("shop_id")["sq_meter"]
           .apply(lambda s: float(s.dropna().iloc[-1]) if s.dropna().size else np.nan))
    g = g.merge(sqm, on="shop_id", how="left")
    g["sales_per_sqm"] = g.apply(
        lambda r: r["turnover"]/r["sq_meter"] if (pd.notna(r["sq_meter"]) and r["sq_meter"]>0) else np.nan, axis=1)
    g["shop_name"] = g["shop_id"].map(SHOP_NAME_MAP)
    return g

agg = agg_store(df)

# ---------- Regionale KPI's ----------
total_turn = agg.groupby("shop_id")["turnover"].sum().sum()
total_vis  = agg.groupby("shop_id")["count_in"].sum().sum()
total_sqm  = agg["sq_meter"].fillna(0).sum()

avg_conv   = weighted_avg(agg["conversion_rate"], agg["count_in"])
avg_spv    = weighted_avg(agg["sales_per_visitor"], agg["count_in"])
avg_spsqm  = (agg["turnover"].sum()/total_sqm) if total_sqm>0 else np.nan

# ---------- KPI Cards ----------
c1,c2,c3,c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="kpi"><div class="t">💶 Totale omzet</div>
    <div class="v">{fmt_eur0(total_turn)}</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="kpi"><div class="t">🛒 Gem. conversie</div>
    <div class="v">{fmt_pct2(avg_conv)}</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="kpi"><div class="t">💸 Gem. SPV</div>
    <div class="v">{fmt_eur2(avg_spv)}</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="kpi"><div class="t">🏁 Gem. sales/m²</div>
    <div class="v">{fmt_eur2(avg_spsqm)}</div></div>""", unsafe_allow_html=True)

st.markdown("---")

# ---------- Heatmap ----------
st.subheader("🔥 Sales/m² Heatmap per winkel")

heat = agg.pivot_table(index="shop_name", columns="date_eff", values="sales_per_sqm", aggfunc="mean")
fig = px.imshow(heat, text_auto=".1f", aspect="auto", color_continuous_scale="Viridis",
                labels=dict(x="Periode", y="Winkel", color="Sales/m²"))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ---------- AI Portfolio Coach ----------
st.subheader("🤖 AI Portfolio Coach")

median_spsqm = agg.groupby("shop_id")["sales_per_sqm"].median().median()

underperf = agg.groupby("shop_name")["sales_per_sqm"].median().sort_values().head(3)
overperf  = agg.groupby("shop_name")["sales_per_sqm"].median().sort_values(ascending=False).head(3)

cA, cB = st.columns(2)
with cA:
    st.markdown("**📉 Onderpresteerders (kandidaten herziening):**")
    st.table(underperf.map(fmt_eur2))
with cB:
    st.markdown("**🚀 Sterke presteerders (opschalen of benchmarken):**")
    st.table(overperf.map(fmt_eur2))

st.markdown("""
<div class="box">
  <h4>AI Aanbevelingen</h4>
  <ul>
    <li>Winkels met <b>sales/m² < regio-median</b> → check huurcontracten en assortimentsmix.</li>
    <li>Winkels met <b>stijgende street traffic maar dalende capture</b> → extra personeel of marketing inzetten.</li>
    <li>Gebruik top-3 winkels als <b>best practices</b> voor training en cross-store benchmarks.</li>
  </ul>
</div>
""", unsafe_allow_html=True)
