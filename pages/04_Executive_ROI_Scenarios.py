# pages/04_Executive_ROI_Scenarios.py
import os, sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

# ── Import pad naar projectroot ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Helpers / utils ────────────────────────────────────────────────────────────
from helpers_shop import ID_TO_NAME, NAME_TO_ID
from helpers_normalize import normalize_vemcount_response, to_wide
from utils_pfmx import api_get_report, friendly_error, inject_css

# ── Pagina setup ───────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Executive ROI Scenarios", page_icon="💼")
inject_css()
TZ = ZoneInfo("Europe/Amsterdam")

# ── Kleine helpers ─────────────────────────────────────────────────────────────
def fmt_int(n):
    try:
        return f"{int(round(float(n))):,}".replace(",", ".")
    except Exception:
        return "0"

def fmt_eur(n):
    try:
        return f"€{float(n):,.0f}".replace(",", ".")
    except Exception:
        return "€0"

def fmt_pct(x, digits=1):
    try:
        return f"{float(x):.{digits}f}%"
    except Exception:
        return f"{0:.{digits}f}%"

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("💼 Executive ROI Scenarios")

ids = list(ID_TO_NAME.keys())
period = st.selectbox(
    "Periode",
    ["last_month", "this_quarter", "last_quarter", "this_year", "last_year"],
    index=0,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    # +pp in fracties: 0.05 = +5pp
    conv_add = st.slider("Conversie uplift (+pp)", 0.0, 0.20, 0.05, 0.01)
with c2:
    # We behandelen dit als ATV-uplift om dubbeltelling te vermijden
    atv_uplift = st.slider("SPV-uplift (%)", 0, 50, 10, 1) / 100.0
with c3:
    gross_margin = st.slider("Brutomarge (%)", 20, 80, 55, 1) / 100.0
with c4:
    capex = st.number_input("CAPEX per store (€)", min_value=0, value=1500, step=100)
payback_target = st.slider("Payback-target (mnd)", 6, 24, 12, 1)

# ── API call ───────────────────────────────────────────────────────────────────
params = []
for sid in ids:
    params.append(("data", sid))
for k in ["count_in", "conversion_rate", "turnover", "sales_per_visitor"]:
    params.append(("data_output", k))
params += [("source", "shops"), ("period", period)]

js = api_get_report(params)
if friendly_error(js, period):
    st.stop()

# mapping moet id->name zijn
df = normalize_vemcount_response(js, ID_TO_NAME, kpi_keys=["count_in","conversion_rate","turnover","sales_per_visitor"])
wide = to_wide(df)  # 1 rij per (date, shop)

# ── Baselines per winkel ───────────────────────────────────────────────────────
base = (
    wide.groupby(["shop_id", "shop_name"], as_index=False)
    .agg(
        count_in=("count_in", "sum"),
        turnover=("turnover", "sum"),
        # conversie in % (zoals je data elders liet zien)
        conversion_rate=("conversion_rate", "mean"),
        # SPV in €
        sales_per_visitor=("sales_per_visitor", "mean"),
    )
)

# SPV = conversie × ATV  =>  ATV = SPV / conversie
conv_f = base["conversion_rate"].astype(float) / 100.0
base["ATV"] = np.where(conv_f > 0, base["sales_per_visitor"] / conv_f, 0.0)

# ── Scenario zonder dubbeltelling ──────────────────────────────────────────────
# Conversie +pp (clamp 0..1)
new_conv_f = np.clip(conv_f + conv_add, 0.0, 1.0)
# ATV uplift (slider "SPV-uplift" gebruiken we bewust voor ATV)
new_atv = base["ATV"] * (1.0 + atv_uplift)
# Nieuwe SPV = conversie × ATV
new_spv = new_conv_f * new_atv

# Extra omzet / brutowinst
extra_rev = base["count_in"] * (new_spv - base["sales_per_visitor"])
extra_gp  = extra_rev * gross_margin

# ── CAPEX / payback / ROI ──────────────────────────────────────────────────────
stores = len(base)
total_capex = capex * stores

# Bepaal lengte van periode uit datumbereik
if "date" in df.columns and not df.empty:
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.notna().any():
        days = int((dates.max() - dates.min()).days) + 1
    else:
        days = 30
else:
    days = 30
months = max(1.0, days / 30.44)

monthly_gp = extra_gp.sum() / months if months > 0 else extra_gp.sum()
payback_months = (total_capex / monthly_gp) if monthly_gp > 0 else np.inf
roi_pct = ((extra_gp.sum() - total_capex) / total_capex * 100.0) if total_capex > 0 else 0.0

# ── Metrics ────────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("📈 Extra omzet (scenario)", fmt_eur(extra_rev.sum()))
m2.metric("💵 Extra brutowinst", fmt_eur(extra_gp.sum()))
m3.metric(
    "⏳ Payback",
    "∞ mnd" if np.isinf(payback_months) else f"{payback_months:.1f} mnd",
    delta=f"Target {payback_target} mnd",
)
m4.metric("📊 ROI", fmt_pct(roi_pct, 1))

# ── Tabel ──────────────────────────────────────────────────────────────────────
st.subheader("Stores (top 10 extra brutowinst)")
base = base.copy()
base["extra_revenue"] = extra_rev
base["extra_gross_profit"] = extra_gp
base["conversion_rate_fmt"] = base["conversion_rate"].map(lambda x: fmt_pct(x, 2))
base["sales_per_visitor_fmt"] = base["sales_per_visitor"].map(lambda x: f"€{x:,.2f}".replace(",", "."))

top = base.sort_values("extra_gross_profit", ascending=False).head(10).copy()
top["extra_gross_profit_fmt"] = top["extra_gross_profit"].map(fmt_eur)
top["ATV_fmt"] = top["ATV"].map(lambda x: f"€{x:,.2f}".replace(",", "."))

st.dataframe(
    top[["shop_name", "count_in", "sales_per_visitor_fmt", "conversion_rate_fmt", "ATV_fmt", "extra_gross_profit_fmt"]]
    .rename(columns={
        "shop_name": "Winkel",
        "count_in": "Bezoekers",
        "sales_per_visitor_fmt": "SPV",
        "conversion_rate_fmt": "Conversie",
        "ATV_fmt": "ATV",
        "extra_gross_profit_fmt": "Extra brutowinst",
    }),
    use_container_width=True,
)

# ── Board tips ─────────────────────────────────────────────────────────────────
st.subheader("🤖 Board Tips")
st.markdown(
    f"- **+{atv_uplift*100:.0f}% ATV** en **+{conv_add*100:.0f}pp conversie** geven samen **{fmt_eur(extra_gp.sum())}** extra brutowinst "
    f"in ~{months:.1f} mnd; payback ≈ **{('∞' if np.isinf(payback_months) else f'{payback_months:.1f}')} mnd**."
)
st.markdown("- Richt CAPEX op winkels met **hoog verkeer** maar **lage SPV/conversie** (zie Region Performance).")

# ── Debug ──────────────────────────────────────────────────────────────────────
with st.expander("🔧 Debug"):
    st.code(params)
    st.dataframe(base.head(10))
