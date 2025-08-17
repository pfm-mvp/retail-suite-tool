# pages/03_Portfolio_Benchmark_AI.py
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

st.set_page_config(page_title="Retail Portfolio Benchmark (AI)", page_icon="🌍", layout="wide")
st.title("🌍 Retail Portfolio Benchmark (AI)")

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
.box {{ border:1px dashed #ddd; border-radius:12px; padding:14px; background:#FAFAFC; }}
</style>
""", unsafe_allow_html=True)

# ---------- Inputs ----------
PERIODS = ["this_year","last_year","this_quarter","last_quarter"]
period = st.selectbox("Periode", PERIODS, index=0)

# ---------- Helpers ----------
TZ = pytz.timezone("Europe/Amsterdam")
ALL_IDS = list(SHOP_NAME_MAP.keys())
METRICS = ["count_in","conversion_rate","turnover","sales_per_visitor","sq_meter"]

def post_report(params):
    r = requests.post(API_URL, params=params, timeout=45)
    r.raise_for_status()
    return r

def add_effective_date(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # prefer 'date' if present, else fallback to timestamp
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

def fmt_eur0(x): return f"€{x:,.0f}".replace(",", ".")
def fmt_eur2(x): return f"€{x:,.2f}".replace(",", ".")
def fmt_pct2(x): return f"{x:.2f}%"

# ---------- Data ophalen (maandstap om trends te zien) ----------
raw = fetch_df(ALL_IDS, period, "month", METRICS)
if raw.empty:
    st.warning("Geen data gevonden voor deze periode.")
    st.stop()

# ---------- Aggregatie per winkel en maand (NO groupby.apply/reset_index) ----------
# 1) sommen
grp_sum = (
    raw.groupby(["shop_id","date_eff"], as_index=False)
       .agg(count_in=("count_in","sum"), turnover=("turnover","sum"))
)

# 2) gewogen sommen voorbereiden en aggregeren
tmp = raw.copy()
tmp["w"]      = tmp["count_in"].fillna(0.0)
tmp["conv_w"] = tmp["conversion_rate"].fillna(0.0)   * tmp["w"]
tmp["spv_w"]  = tmp["sales_per_visitor"].fillna(0.0) * tmp["w"]

grp_w = (
    tmp.groupby(["shop_id","date_eff"], as_index=False)
       .agg(w=("w","sum"), conv_w=("conv_w","sum"), spv_w=("spv_w","sum"))
)

agg = grp_sum.merge(grp_w, on=["shop_id","date_eff"], how="left")

# 3) vaste m² per winkel (laatst bekende niet-NaN)
sq = (
    raw.sort_values("date_eff")
       .groupby("shop_id")["sq_meter"]
       .apply(lambda s: float(s.dropna().iloc[-1]) if s.dropna().size else np.nan)
       .rename("sq_meter")
       .reset_index()
)
agg = agg.merge(sq, on="shop_id", how="left")

# 4) weighted metrics + sales_per_sqm
agg["conversion_rate"]   = np.where(agg["w"]>0,   agg["conv_w"]/agg["w"], np.nan)
agg["sales_per_visitor"] = np.where(agg["w"]>0,   agg["spv_w"]/agg["w"],  np.nan)
agg["sales_per_sqm"]     = np.where(agg["sq_meter"]>0, agg["turnover"]/agg["sq_meter"], np.nan)
agg["shop_name"]         = agg["shop_id"].map(SHOP_NAME_MAP)

# ---------- Regionale KPI's ----------
total_turn = agg["turnover"].sum()
total_vis  = agg["count_in"].sum()
total_sqm  = agg["sq_meter"].fillna(0).sum()

# gewogen op bezoekers
avg_conv  = (agg["conversion_rate"].fillna(0.0)   * agg["count_in"].fillna(0.0)).sum()
avg_conv  = (avg_conv / total_vis) if total_vis > 0 else np.nan

avg_spv   = (agg["sales_per_visitor"].fillna(0.0) * agg["count_in"].fillna(0.0)).sum()
avg_spv   = (avg_spv / total_vis) if total_vis > 0 else np.nan

avg_spsqm = (total_turn / total_sqm) if total_sqm > 0 else np.nan

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

# ---------- Heatmap (sales/m² per winkel over tijd) ----------
st.subheader("🔥 Sales/m² Heatmap per winkel (maanden)")
heat = agg.pivot_table(index="shop_name", columns="date_eff",
                       values="sales_per_sqm", aggfunc="mean").sort_index()
fig = px.imshow(
    heat,
    text_auto=".1f",
    aspect="auto",
    color_continuous_scale="Viridis",
    labels=dict(x="Periode", y="Winkel", color="Sales/m²")
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ---------- Leaderboard: Δ vs regio-gemiddelde sales/m² ----------
st.subheader("🏁 Leaderboard — sales/m² t.o.v. regio-gemiddelde")

last_period = agg["date_eff"].max()
snap = agg[agg["date_eff"] == last_period].copy()

snap["region_avg_spsqm"] = avg_spsqm
snap["delta_eur_sqm"] = snap["sales_per_sqm"] - snap["region_avg_spsqm"]
snap["delta_pct"] = np.where(
    snap["region_avg_spsqm"] > 0,
    (snap["delta_eur_sqm"] / snap["region_avg_spsqm"]) * 100.0,
    np.nan
)

sort_best_first = st.toggle("Beste afwijking eerst", value=True)
snap = snap.sort_values("delta_eur_sqm", ascending=not sort_best_first)

show = snap[["shop_name","sales_per_sqm","region_avg_spsqm","delta_eur_sqm","delta_pct"]].rename(columns={
    "shop_name": "winkel",
    "sales_per_sqm": "sales/m²",
    "region_avg_spsqm": "gem. sales/m² (regio)",
    "delta_eur_sqm": "Δ vs gem. (€/m²)",
    "delta_pct": "Δ vs gem. (%)",
})

def color_delta(series):
    styles = []
    for v in series:
        if pd.isna(v) or v == 0:
            styles.append("color: #6B7280;")
        elif v > 0:
            styles.append("color: #22C55E;")
        else:
            styles.append("color: #F04438;")
    return styles

styler = (
    show.style
        .format({
            "sales/m²": "€{:.2f}",
            "gem. sales/m² (regio)": "€{:.2f}",
            "Δ vs gem. (€/m²)": "€{:+.2f}",
            "Δ vs gem. (%)": "{:+.1f}%"
        })
        .apply(color_delta, subset=["Δ vs gem. (€/m²)"])
        .apply(color_delta, subset=["Δ vs gem. (%)"])
)
st.dataframe(styler, use_container_width=True)

st.markdown("---")

# ---------- 🤖 AI Portfolio Coach ----------
st.subheader("🤖 AI Portfolio Coach")

median_spsqm = agg.groupby("shop_name")["sales_per_sqm"].median().median()
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
    <li>Winkels met <b>sales/m² &lt; regio-median</b> → check huurcontracten, routing en assortiment.</li>
    <li>Gebruik top-3 winkels als <b>best practices</b> voor coaching en schappenplan.</li>
    <li>Monitor heatmap op <b>doorlopende dalers</b> (3+ maanden): actieplan voor conversie en SPV.</li>
  </ul>
</div>
""", unsafe_allow_html=True)
