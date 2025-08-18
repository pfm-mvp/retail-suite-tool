# pages/04_Executive_ROI_Scenarios.py
import os, sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

# Ensure root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helpers_shop import ID_TO_NAME
from utils_pfmx import api_get_report, friendly_error, inject_css
from helpers_normalize import normalize_vemcount_response, to_wide

st.set_page_config(layout="wide")
inject_css()
TZ = ZoneInfo("Europe/Amsterdam")

def fmt_int(n):
    try: return f"{int(round(float(n))):,}".replace(",", ".")
    except: return "0"

def fmt_eur(n):
    try: return f"€{float(n):,.0f}".replace(",", ".")
    except: return "€0"

def fmt_pct(x, digits=1):
    try: return f"{float(x):.{digits}f}%"
    except: return "0.0%"

st.title("💼 Executive ROI Scenarios")

# ── Inputs ────────────────────────────────────────────────────────────────────
ids = list(ID_TO_NAME.keys())
period = st.selectbox(
    "Periode",
    ["last_month","this_quarter","last_quarter","this_year","last_year"],
    index=0
)

c1,c2,c3,c4 = st.columns(4)
with c1:
    # Conversie-uplift in procentpunten (pp)
    conv_add = st.slider("Conversie uplift (+pp)", 0.0, 20.0, 5.0, 0.5)
with c2:
    spv_uplift = st.slider("SPV-uplift (%)", 0, 50, 10, 1) / 100.0
with c3:
    gross_margin = st.slider("Brutomarge (%)", 20, 80, 55, 1) / 100.0
with c4:
    capex = st.number_input("CAPEX per store (€)", min_value=0, value=1500, step=100)
payback_target = st.slider("Payback-target (mnd)", 6, 24, 12, 1)

# ── Data ophalen ───────────────────────────────────────────────────────────────
KPI_KEYS = ["count_in", "conversion_rate", "turnover", "sales_per_visitor"]

params = []
for sid in ids:
    params.append(("data", sid))
for k in KPI_KEYS:
    params.append(("data_output", k))
# IMPORTANT: day step so we can compute months robustly
params += [("source","shops"), ("period", period), ("step","day")]

js = api_get_report(params)
if friendly_error(js, period):
    st.stop()

# normalize expects id->name
df = normalize_vemcount_response(js, ID_TO_NAME, kpi_keys=KPI_KEYS)

if df is None or df.empty:
    st.warning("Geen data ontvangen voor deze periode/parameters.")
    with st.expander("🔧 Debug"):
        st.write("Params:", params)
        st.write("Normalize → empty DataFrame")
    st.stop()

# Ensure shop_id present
if "shop_id" not in df.columns:
    st.error("Response mist 'shop_id' kolom na normalisatie.")
    with st.expander("🔧 Debug"):
        st.write(df.head())
    st.stop()

# Build a reliable date column
# - keep a datetime64[ns] 'date_eff' for grouping / length of period
# - keep a nice string 'date' for display (if needed)
ts = pd.to_datetime(df.get("timestamp"), errors="coerce")
df["date_eff"] = ts.dt.floor("D")
df["date"] = df["date_eff"].dt.date.astype("string")

# Drop rows that have no shop or date
df = df[df["shop_id"].notna() & df["date_eff"].notna()]
if df.empty:
    st.warning("Er zijn wel winkels, maar geen rijen met datums in deze periode.")
    with st.expander("🔧 Debug"):
        st.write("Params:", params)
    st.stop()

with st.expander("🔧 Debug — fetch result"):
    st.write("Rows:", len(df), "Shops:", df["shop_id"].nunique())
    st.dataframe(df.head(10))

# ── Wide view & baselines per winkel ──────────────────────────────────────────
# to_wide returns one row per date+shop; it expects a normalised schema.
wide = to_wide(df)

if wide is None or wide.empty:
    st.warning("to_wide() gaf geen rijen terug — kan gebeuren als alle KPI's nul/NaN zijn.")
    with st.expander("🔧 Debug"):
        st.write("Params:", params)
        st.dataframe(df.head(10))
    st.stop()

# Aggregate over period per shop
def _safe_mean(s):
    s = pd.to_numeric(s, errors="coerce")
    return float(s.mean()) if s.notna().any() else 0.0

base = (
    wide.groupby(["shop_id","shop_name"], as_index=False)
        .agg({
            "count_in":"sum",           # verkeer
            "turnover":"sum",           # omzet
            "conversion_rate": _safe_mean,     # gemiddelde conv (gewichtloos OK)
            "sales_per_visitor": _safe_mean,   # gemiddelde SPV
        })
)

if base.empty:
    st.warning("Geen geaggregeerde rijen per winkel (base is leeg).")
    with st.expander("🔧 Debug"):
        st.dataframe(wide.head(10))
    st.stop()

# ATV afleiden: SPV = conv * ATV  =>  ATV = SPV / conv
# Let op: conversie is in %, dus eerst naar fractie.
conv_frac = pd.to_numeric(base["conversion_rate"], errors="coerce") / 100.0
spv_val   = pd.to_numeric(base["sales_per_visitor"], errors="coerce")
base["ATV"] = np.where(conv_frac > 0, spv_val / conv_frac, 0.0)

# ── Scenario berekenen ────────────────────────────────────────────────────────
# Conversie-uplift in pp → we gebruiken het alleen om de tekst te tonen,
# de omzet-berekening zelf gebruikt SPV-uplift (zoals eerder).
new_spv = spv_val * (1 + spv_uplift)

# Extra omzet en brutowinst
count_in_val = pd.to_numeric(base["count_in"], errors="coerce").fillna(0.0)
extra_rev = count_in_val * (new_spv - spv_val)
extra_gp  = extra_rev * gross_margin

# Payback/ROI
stores = len(base)
total_capex = capex * stores

# Aantal dagen uit date_eff
days = df["date_eff"].nunique()
months = max(1.0, days / 30.44)  # safeguard tegen 0
monthly_gp = extra_gp.sum()/months if months > 0 else extra_gp.sum()
payback_months = (total_capex / monthly_gp) if monthly_gp > 0 else np.inf
roi_pct = ((extra_gp.sum() - total_capex) / total_capex * 100) if total_capex > 0 else 0.0

# ── KPIs tonen ────────────────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
c1.metric("📈 Extra omzet (scenario)", fmt_eur(extra_rev.sum()))
c2.metric("💵 Extra brutowinst", fmt_eur(extra_gp.sum()))
c3.metric(
    "⏳ Payback",
    "∞ mnd" if np.isinf(payback_months) else f"{payback_months:.1f} mnd",
    delta=f"Target {payback_target} mnd"
)
c4.metric("📊 ROI", fmt_pct(roi_pct,1))

# ── Stores tabel ──────────────────────────────────────────────────────────────
st.subheader("Stores (top 10 extra brutowinst)")
base = base.copy()
base["extra_revenue"] = extra_rev
base["extra_gross_profit"] = extra_gp
top = base.sort_values("extra_gross_profit", ascending=False).head(10)
top["extra_gross_profit_fmt"] = top["extra_gross_profit"].map(fmt_eur)

st.dataframe(
    top[["shop_name","count_in","sales_per_visitor","conversion_rate","ATV","extra_gross_profit_fmt"]],
    use_container_width=True
)

# ── Tips ──────────────────────────────────────────────────────────────────────
st.subheader("🤖 Board Tips")
st.markdown(
    f"- **+{conv_add:.1f} pp conversie** en **+{spv_uplift*100:.0f}% SPV** "
    f"leveren **{fmt_eur(extra_gp.sum())}** extra brutowinst op in {months:.1f} mnd; "
    f"payback ≈ **{('∞' if np.isinf(payback_months) else f'{payback_months:.1f}')} mnd**."
)
st.markdown(
    "- Richt CAPEX op winkels met **hoog verkeer maar lage SPV/conversie** "
    "(zie Region Performance)."
)

# ── Debug ─────────────────────────────────────────────────────────────────────
with st.expander("🔧 Debug"):
    st.write("Params:", params)
    st.dataframe(base.head(10))
