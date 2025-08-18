# pages/03_Retail_Portfolio_Benchmark_AI.py
import os, sys
from datetime import datetime
import pytz
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import calendar  # voor maandnamen en sortering

# ---------- Imports / mapping ----------
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))
from helpers_shop import ID_TO_NAME, NAME_TO_ID, REGIONS, get_ids_by_region
from helpers_normalize import normalize_vemcount_response

st.set_page_config(page_title="Portfolio Benchmark (AI)", page_icon="📊", layout="wide")
st.title("📊 Portfolio Benchmark (AI)")

API_URL = st.secrets["API_URL"]

# ---------- PFM-styling ----------
PFM_RED    = "#F04438"
PFM_GREEN  = "#22C55E"
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
</style>
""", unsafe_allow_html=True)

# ---------- Inputs ----------
PERIODS = ["this_week","last_week","this_month","last_month","this_quarter","last_quarter","this_year","last_year"]
period = st.selectbox("Periode", PERIODS, index=4, key="p03_period")
regio  = st.selectbox("Regio", REGIONS, index=0, key="p03_region")
shop_ids = get_ids_by_region(regio)
if not shop_ids:
    st.warning("Geen winkels in deze regio.")
    st.stop()

# ---------- Helpers ----------
TZ     = pytz.timezone("Europe/Amsterdam")
TODAY  = datetime.now(TZ).date()
METRICS = ["count_in","conversion_rate","turnover","sales_per_visitor","sq_meter"]

def step_for(period: str) -> str:
    """Gebruik day voor week/maand, month voor kwartaal/jaar."""
    if period.endswith("week") or period.endswith("month"):
        return "day"
    return "month"

def post(params):
    r = requests.post(API_URL, params=params, timeout=45)
    r.raise_for_status()
    return r

def add_effective_date(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "date" not in d.columns:
        d["date"] = pd.NaT
    ts = pd.to_datetime(d.get("timestamp"), errors="coerce")
    d["date_eff"] = pd.to_datetime(d["date"], errors="coerce").fillna(ts)
    d["date_eff"] = d["date_eff"].dt.date
    d["year"]  = pd.to_datetime(d["date_eff"]).dt.year
    d["month"] = pd.to_datetime(d["date_eff"]).dt.month
    return d

def fetch(period: str, ids: list[int]) -> pd.DataFrame:
    params = [("data", sid) for sid in ids]
    params += [("data_output", m) for m in METRICS]
    params += [("source","shops"), ("period", period), ("step", step_for(period))]
    r  = post(params)
    js = r.json()
    # naam-mapping alleen voor gekozen regio
    id2name = {sid: ID_TO_NAME.get(sid, str(sid)) for sid in ids}
    df = normalize_vemcount_response(js, id2name, kpi_keys=METRICS)
    df = add_effective_date(df)

    # Exclude vandaag voor “this_*”
    if period.startswith("this_"):
        df = df[df["date_eff"] < TODAY]

    # Forward/backward-fill sq_meter per shop
    df = df.sort_values(["shop_id","date_eff"])
    df["sq_meter"] = df.groupby("shop_id")["sq_meter"].ffill().bfill()
    return df

def fmt_eur0(x): return f"€{x:,.0f}".replace(",", ".")
def fmt_eur2(x): return f"€{x:,.2f}".replace(",", ".")
def fmt_pct2(x): return f"{x:.2f}%"

def weighted_avg(series, weights):
    w = weights.fillna(0.0); s = series.fillna(0.0)
    d = w.sum()
    return (s*w).sum()/d if d else np.nan

# ---------- Get data ----------
df = fetch(period, shop_ids)
if df.empty:
    st.warning("Geen data voor deze periode/regio.")
    st.stop()

# ---------- Portfolio KPIs ----------
# Aggregate per shop (Σ traffic/turnover; conversie/SPV gewogen op traffic; laatste m²)
g = df.groupby("shop_id", as_index=False).agg({"count_in":"sum","turnover":"sum"})
w = df.groupby("shop_id").apply(lambda x: pd.Series({
    "conversion_rate": weighted_avg(x["conversion_rate"], x["count_in"]),
    "sales_per_visitor": weighted_avg(x["sales_per_visitor"], x["count_in"]),
})).reset_index()
g = g.merge(w, on="shop_id", how="left")

sqm = (df.groupby("shop_id")["sq_meter"]
         .apply(lambda s: float(s.dropna().iloc[-1]) if s.dropna().size else np.nan)
         .reset_index())
g = g.merge(sqm, on="shop_id", how="left")

g["sales_per_sqm"] = np.where(g["sq_meter"]>0, g["turnover"]/g["sq_meter"], np.nan)
g["shop_name"] = g["shop_id"].map(ID_TO_NAME)

# Region totals / averages
total_turn = g["turnover"].sum()
avg_conv   = weighted_avg(g["conversion_rate"],   g["count_in"])
avg_spv    = weighted_avg(g["sales_per_visitor"], g["count_in"])
region_sqm = g["sq_meter"].fillna(0).sum()
avg_spsqm  = (total_turn/region_sqm) if region_sqm>0 else np.nan

# KPI cards
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

# ---------- 🔥 Sales/m² Heatmap per winkel (maanden) ----------
st.subheader("🔥 Sales/m² Heatmap per winkel (maanden)")

# 1) Bouw maanddata: Σ omzet per shop/maand / laatste m² van die shop
monthly = (
    df.groupby(["shop_id","month"], as_index=False)
      .agg(turnover=("turnover","sum"), sq_meter=("sq_meter","last"))
)
monthly["sales_per_sqm"] = np.where(
    monthly["sq_meter"] > 0, monthly["turnover"] / monthly["sq_meter"], np.nan
)
monthly["shop_name"] = monthly["shop_id"].map(ID_TO_NAME)

# 2) Maandnaam-kolom + chronologische volgorde (Jan…Dec)
monthly["month_name"] = monthly["month"].apply(lambda m: calendar.month_abbr[int(m)])
month_order = [calendar.month_abbr[i] for i in range(1, 13)]

# 3) Pivot naar heatmap (shops × maanden met namen)
hm = (monthly
      .pivot(index="shop_name", columns="month_name", values="sales_per_sqm")
      .sort_index()
)
# Kolommen reordenen naar Jan…Dec, maar alleen de maanden die aanwezig zijn
hm = hm.reindex([m for m in month_order if m in hm.columns], axis=1)

if hm.isna().all(None):
    st.info("Geen maanddata beschikbaar voor deze periode.")
else:
    # 4) PFM-brand colorscale (3-kleuren gradient)
    pfm_colorscale = ["#762181", "#D8456C", "#FEAC76"]

    fig = px.imshow(
        hm,
        color_continuous_scale=pfm_colorscale,
        labels=dict(color="€ / m²"),
        aspect="auto",
    )
    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=10, b=10),
        coloraxis_colorbar=dict(title="€ / m²"),
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ---------- 🥇 vs regio-gemiddelde (sales/m²) ----------
st.subheader("🏁 Leaderboard — sales/m² t.o.v. regio-gemiddelde")

comp = g[["shop_name","sales_per_sqm"]].copy()
comp["reg_avg_spsqm"] = avg_spsqm
comp["Δ €/m²"] = comp["sales_per_sqm"] - comp["reg_avg_spsqm"]
comp["Δ %"]   = np.where(comp["reg_avg_spsqm"]>0, (comp["Δ €/m²"]/comp["reg_avg_spsqm"])*100, np.nan)
comp = comp.sort_values("Δ €/m²", ascending=False)

styled = (comp.rename(columns={"shop_name":"winkel","sales_per_sqm":"sales/m²","reg_avg_spsqm":"gem. sales/m² (regio)"})
          .style.format({"sales/m²":"€{:.2f}","gem. sales/m² (regio)":"€{:.2f}","Δ €/m²":"€{:+.2f}","Δ %":"{:+.1f}%"})
          .apply(lambda s: ["color:#22C55E" if v>0 else ("color:#F04438" if v<0 else "color:#6B7280")
                            for v in s], subset=["Δ €/m²","Δ %"])
         )
st.dataframe(styled, use_container_width=True)

# ---------- 🤖 Mini AI summary ----------
st.markdown("## 🤖 Portfolio Coach (samenvatting)")
top_up   = comp.nlargest(3, "Δ €/m²")[["shop_name","Δ €/m²"]]
top_down = comp.nsmallest(3, "Δ €/m²")[["shop_name","Δ €/m²"]]

def list_lines(df):
    return "<br>".join([f"• {r['shop_name']}: {fmt_eur2(r['Δ €/m²'])}/m²" for _,r in df.iterrows()])

st.markdown(f"""
**Sterk t.o.v. regio-gemiddelde (€/m²):**<br>{list_lines(top_up)}<br><br>
**Zwak t.o.v. regio-gemiddelde (€/m²):**<br>{list_lines(top_down)}<br><br>
**Tip:** focus op winkels <span style="color:{PFM_RED};font-weight:600">onder</span> het gemiddelde; check conversie-uurtoppen en SPV-bundels.
""", unsafe_allow_html=True)

# ---------- Debug ----------
with st.expander("🔧 Debug"):
    st.write("period:", period, "step:", step_for(period))
    st.write("regio:", regio, "ids:", shop_ids[:10], "… (totaal", len(shop_ids), ")")
    st.write("df head:", df.head())
    st.write("monthly head:", monthly.head())
    st.write("agg per shop head:", g.head())
