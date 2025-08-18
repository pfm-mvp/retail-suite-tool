import os, sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

# ───────────────── Base setup ─────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helpers_shop import ID_TO_NAME
from utils_pfmx import api_get_report, friendly_error, inject_css
from helpers_normalize import normalize_vemcount_response, to_wide

st.set_page_config(layout="wide")
inject_css()
TZ = ZoneInfo("Europe/Amsterdam")

# ───────────────── Kleine helpers ─────────────────
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
        return "0.0%"

# ───────────────── UI ─────────────────
st.title("💼 Executive ROI Scenarios")

ids = list(ID_TO_NAME.keys())
period = st.selectbox(
    "Periode",
    ["last_month", "this_quarter", "last_quarter", "this_year", "last_year"],
    index=0,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    # Conversie-uplift in procentpunten (pp)
    conv_add_pp = st.slider("Conversie uplift (+pp)", 0.0, 20.0, 5.0, 0.5)
with c2:
    spv_uplift = st.slider("SPV-uplift (%)", 0, 50, 10, 1) / 100.0
with c3:
    gross_margin = st.slider("Brutomarge (%)", 20, 80, 55, 1) / 100.0
with c4:
    capex = st.number_input("CAPEX per store (€)", min_value=0, value=1500, step=100)

payback_target = st.slider("Payback-target (mnd)", 6, 24, 12, 1)

# ───────────────── Data ophalen ─────────────────
KPI_KEYS = ["count_in", "conversion_rate", "turnover", "sales_per_visitor"]

params = []
for sid in ids:
    params.append(("data", sid))
for k in KPI_KEYS:
    params.append(("data_output", k))
# step=day zorgt dat to_wide/time-aggregaties goed werken
params += [("source", "shops"), ("period", period), ("step", "day")]

js = api_get_report(params)
if friendly_error(js, period):
    st.stop()

# normalize expects an id->name map
df = normalize_vemcount_response(js, ID_TO_NAME, kpi_keys=KPI_KEYS)

if df is None or df.empty:
    st.warning("Geen data ontvangen voor deze periode/parameters.")
    with st.expander("🔧 Debug"):
        st.write("Params:", params)
        st.write("Normalize → empty DataFrame")
    st.stop()

# Zorg dat 'date' bestaat
if "date" not in df.columns:
    df["date"] = pd.to_datetime(df.get("timestamp"), errors="coerce").dt.date

# Safety: verwijder rijen zonder shop_id
df = df[pd.notna(df.get("shop_id"))]

# Eén rij per dag+shop
wide = to_wide(df)

with st.expander("🔧 Debug — fetch result"):
    st.write("Rows:", len(df), "Shops:", df.get("shop_id", pd.Series(dtype=float)).nunique())
    st.dataframe(df.head(10))

# ───────────────── Baselines per winkel ─────────────────
base = wide.groupby(["shop_id", "shop_name"], as_index=False).agg(
    {
        "count_in": "sum",
        "turnover": "sum",
        "conversion_rate": "mean",
        "sales_per_visitor": "mean",
    }
)

# Detecteer eenheid van conversie en normaliseer naar FRAC (0–1)
# Als waarden > 1 zijn, interpreteren we ze als percentages (bv. 36.5) en delen door 100.
conv_raw = base["conversion_rate"].astype(float)
conv_frac = np.where(conv_raw > 1.0, conv_raw / 100.0, conv_raw)

# ATV uit SPV en conversie: SPV (€) = conv_frac × ATV  =>  ATV = SPV / conv_frac
base["ATV"] = np.where(conv_frac > 0, base["sales_per_visitor"] / conv_frac, 0.0)

# ───────────────── Scenario-berekening ─────────────────
# Nieuwe conversie-fractie: +pp op huidige conversie
conv_add_frac = conv_add_pp / 100.0
new_conv_frac = conv_frac + conv_add_frac

# Nieuwe SPV = nieuwe conversie × ATV × (1 + SPV-uplift)
new_spv = new_conv_frac * base["ATV"] * (1.0 + spv_uplift)

# Extra omzet en brutowinst
extra_rev = base["count_in"] * (new_spv - base["sales_per_visitor"])
extra_gp = extra_rev * gross_margin

# ───────────────── Financiële KPI's ─────────────────
stores = len(base)
total_capex = capex * stores

# Schatting maandlengte op basis van aantal unieke datums
days = pd.to_datetime(df["date"]).nunique() if "date" in df.columns else 30
months = max(1.0, days / 30.44)

monthly_gp = extra_gp.sum() / months if months > 0 else extra_gp.sum()
payback_months = (total_capex / monthly_gp) if monthly_gp > 0 else np.inf
roi_pct = ((extra_gp.sum() - total_capex) / total_capex * 100.0) if total_capex > 0 else 0.0

# ───────────────── Headline cards ─────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("📈 Extra omzet (scenario)", fmt_eur(extra_rev.sum()))
c2.metric("💵 Extra brutowinst", fmt_eur(extra_gp.sum()))
c3.metric(
    "⏳ Payback",
    "∞ mnd" if np.isinf(payback_months) else f"{payback_months:.1f} mnd",
    delta=f"Target {payback_target} mnd",
)
c4.metric("📊 ROI", fmt_pct(roi_pct, 1))

# ───────────────── Tabel per store ─────────────────
st.subheader("Stores (top 10 extra brutowinst)")

base = base.copy()
base["extra_revenue"] = extra_rev
base["extra_gross_profit"] = extra_gp

top = base.sort_values("extra_gross_profit", ascending=False).head(10).copy()
top["count_in"] = top["count_in"].map(fmt_int)
top["sales_per_visitor"] = top["sales_per_visitor"].map(lambda x: f"€{x:,.2f}".replace(",", "."))
top["conversion_rate"] = np.where(
    conv_raw.head(len(top)) > 1.0,  # gebruik de ruwe eenheid voor nette weergave
    conv_raw.head(len(top)).map(lambda x: f"{x:.2f}%"),
    (conv_raw.head(len(top)) * 100.0).map(lambda x: f"{x:.2f}%"),
)
top["ATV"] = top["ATV"].map(lambda x: f"€{x:,.2f}".replace(",", "."))
top["extra_gross_profit_fmt"] = top["extra_gross_profit"].map(fmt_eur)

st.dataframe(
    top[["shop_name", "count_in", "sales_per_visitor", "conversion_rate", "ATV", "extra_gross_profit_fmt"]],
    use_container_width=True,
)

# ───────────────── Board tips ─────────────────
st.subheader("🤖 Board Tips")
st.markdown(
    f"- **+{conv_add_pp:.1f} pp conversie** en **+{spv_uplift*100:.0f}% SPV** leveren "
    f"**{fmt_eur(extra_gp.sum())}** extra brutowinst op in ~{months:.1f} mnd; payback ≈ "
    f"**{('∞' if np.isinf(payback_months) else f'{payback_months:.1f}')} mnd**."
)
st.markdown(
    "- Richt CAPEX op winkels met **hoog verkeer maar lage SPV/conversie** "
    "(zie Region Performance)."
)

# ───────────────── Debug ─────────────────
with st.expander("🔧 Debug"):
    st.code(params)
    st.dataframe(base.head(10))
