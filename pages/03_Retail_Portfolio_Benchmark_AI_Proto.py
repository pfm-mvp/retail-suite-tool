# pages/03_Retail_Portfolio_Benchmark_AI_Proto.py
import os, sys
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st

# ── Imports uit projectroot ─────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))
from helpers_shop import ID_TO_NAME, REGIONS, get_ids_by_region
from helpers_normalize import normalize_vemcount_response
from utils_pfmx import api_get_report, friendly_error, inject_css

# ── Page config + styling ──────────────────────────────────────────────────────
st.set_page_config(page_title="Retail Portfolio Benchmark — Proto", page_icon="🧪", layout="wide")
inject_css()
st.title("🧪 Retail Portfolio Benchmark — Proto (uurgemiddelden)")

# ── UI: periode + regio + KPI (ELKE widget heeft unieke key!) ──────────────────
PERIODS = ["this_week","last_week","this_month","last_month","this_quarter","last_quarter"]
period = st.selectbox("Periode", PERIODS, index=1, key="p3_period")

regio = st.selectbox("Regio", ["All"] + REGIONS, index=0, key="p3_region")

KPI_KEYS = {
    "Conversie (%)": "conversion_rate",
    "SPV (€)": "sales_per_visitor",
    "Omzet (€)": "turnover",
    "Bezoekers": "count_in",
}
kpi_label = st.selectbox("KPI", list(KPI_KEYS.keys()), index=0, key="p3_kpi")
kpi = KPI_KEYS[kpi_label]

# ── Regio → shop selectie ──────────────────────────────────────────────────────
if regio == "All":
    sel_ids = list(ID_TO_NAME.keys())
else:
    sel_ids = get_ids_by_region(regio) or list(ID_TO_NAME.keys())

sel_names_default = [ID_TO_NAME[i] for i in sel_ids][:6]
sel_names = st.multiselect(
    "Winkels in vergelijking (max 8)",
    [ID_TO_NAME[i] for i in sel_ids],
    default=sel_names_default,
    max_selections=8,
    key="p3_shops"
)
shop_ids = [sid for sid, nm in ID_TO_NAME.items() if nm in sel_names]
if not shop_ids:
    st.warning("Selecteer ten minste één winkel.")
    st.stop()

# ── Data ophalen (step='hour') ─────────────────────────────────────────────────
def fetch_hourly(shop_ids, period, metrics):
    params = []
    for sid in shop_ids:
        params.append(("data", sid))
    for m in metrics:
        params.append(("data_output", m))
    params += [("source","shops"), ("period", period), ("step","hour")]
    js = api_get_report(params)
    return js, params

js, params = fetch_hourly(shop_ids, period, metrics=["count_in","conversion_rate","turnover","sales_per_visitor"])
if friendly_error(js, period):
    st.stop()

df = normalize_vemcount_response(js, ID_TO_NAME, kpi_keys=["count_in","conversion_rate","turnover","sales_per_visitor"])
if df is None or df.empty:
    st.info("Geen data beschikbaar voor deze selectie.")
    st.stop()

# Zorg dat timestamp aanwezig is en hour te bepalen is
ts = pd.to_datetime(df.get("timestamp"), errors="coerce")
df = df.assign(hour=ts.dt.hour)

# ── Uurgemiddelden per winkel ──────────────────────────────────────────────────
# Voor conversie/SPV nemen we een gewogen gemiddelde op basis van count_in
def weighted_mean(x, value_col, weight_col):
    v = x[value_col].astype(float)
    w = x[weight_col].fillna(0).astype(float)
    denom = w.sum()
    return float((v * w).sum() / denom) if denom > 0 else float(v.mean())

if kpi in ("conversion_rate", "sales_per_visitor"):
    agg = (
        df.groupby(["shop_id","shop_name","hour"])
          .apply(lambda x: weighted_mean(x, kpi, "count_in"))
          .reset_index(name=kpi)
    )
else:
    # count_in en turnover: gemiddeld per uur (over de dagen) – kan ook som zijn,
    # maar gemiddeld is handiger voor ‘typisch uur’
    agg = (
        df.groupby(["shop_id","shop_name","hour"], as_index=False)
          .agg({kpi: "mean"})
    )

# ── Pivot: uren (0..23) als kolommen ───────────────────────────────────────────
pivot = (
    agg.pivot_table(index="shop_name", columns="hour", values=kpi, aggfunc="mean")
       .reindex(columns=sorted(agg["hour"].dropna().unique()))
)

# Pretty format voor tabel
def fmt_value(x):
    if pd.isna(x): return ""
    if kpi == "conversion_rate":
        return f"{x:.2f}%"
    if kpi == "sales_per_visitor":
        return f"€{x:,.2f}".replace(",", ".")
    if kpi == "turnover":
        return f"€{x:,.0f}".replace(",", ".")
    return f"{int(round(x)):,}".replace(",", ".")

fmt_table = pivot.copy()
fmt_table = fmt_table.applymap(fmt_value)

st.subheader(f"📊 Uurgemiddelden — {kpi_label} (periode: {period.replace('_',' ')})")
st.dataframe(fmt_table, use_container_width=True)

# Optioneel: een heatmap (Plotly) voor snel visueel overzicht
try:
    import plotly.express as px
    fig = px.imshow(
        pivot.fillna(np.nan),
        labels=dict(x="Uur van de dag", y="Winkel", color=kpi_label),
        x=pivot.columns, y=pivot.index,
        aspect="auto", color_continuous_scale="Purples"
    )
    fig.update_layout(height=520, margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig, use_container_width=True, key="p3_heatmap")
except Exception:
    pass

# ── Debug ──────────────────────────────────────────────────────────────────────
with st.expander("🔧 Debug — API / sample", expanded=False):
    st.write("Params (first 14):", params[:14], "…")
    st.write("Rows:", len(df), "Shops:", df["shop_id"].nunique())
    st.dataframe(df.head(10))
