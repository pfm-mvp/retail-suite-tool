# pages/04_Executive_ROI_Scenarios.py
import os, sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

# ── Imports / setup ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helpers_shop import ID_TO_NAME, NAME_TO_ID, get_ids_by_region  # maps gebruiken, niet overschrijven
from helpers_normalize import normalize_vemcount_response
from utils_pfmx import api_get_report, friendly_error, inject_css

st.set_page_config(page_title="Executive ROI Scenarios", layout="wide")
inject_css()
TZ = ZoneInfo("Europe/Amsterdam")

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_int(n):
    try: return f"{int(round(float(n))):,}".replace(",", ".")
    except: return "0"

def fmt_eur(n):
    try: return f"€{float(n):,.0f}".replace(",", ".")
    except: return "€0"

def fmt_pct(x, digits=1):
    try: return f"{float(x):.{digits}f}%"
    except: return "0.0%"

def weighted_avg(series, weights):
    try:
        w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
        s = pd.to_numeric(series,  errors="coerce").fillna(0.0)
        d = w.sum()
        return (s*w).sum()/d if d else np.nan
    except Exception:
        return np.nan

# ── Inputs ────────────────────────────────────────────────────────────────────
st.title("💼 Executive ROI Scenarios")
period = st.selectbox("Periode",
                      ["last_month","this_quarter","last_quarter","this_year","last_year"],
                      index=0)

c1,c2,c3,c4 = st.columns(4)
with c1:
    # Slider direct in procentpunten (bijv. 5 = +5 pp)
    conv_add = st.slider("Conversie uplift (+pp)", 0.0, 20.0, 5.0, 0.5)

# Nieuwe conversie = huidige conversie + uplift in pp
new_conv = base["conversion_rate"] + conv_add
with c2: atv_uplift = st.slider("ATV-uplift (%)", 0, 50, 10, 1) / 100.0
with c3: gross_margin = st.slider("Brutomarge (%)", 20, 80, 55, 1) / 100.0
with c4: capex = st.number_input("CAPEX per store (€)", min_value=0, value=1500, step=100)
payback_target = st.slider("Payback-target (mnd)", 6, 24, 12, 1)

# ── Data ophalen ───────────────────────────────────────────────────────────────
METRICS = ["count_in","conversion_rate","turnover","sales_per_visitor"]
ids = list(ID_TO_NAME.keys())

params = [("data", sid) for sid in ids]
params += [("data_output", m) for m in METRICS]
params += [("source","shops"), ("period", period), ("step","day")]  # ← belangrijk: step=day

js = api_get_report(params)
if friendly_error(js, period):
    st.stop()

df = normalize_vemcount_response(js, ID_TO_NAME, kpi_keys=METRICS)

if df is None or df.empty:
    st.info("Geen data ontvangen voor deze periode.")
    with st.expander("🔧 Debug"):
        st.write("Payload params:", params)
        st.write("JSON (truncated):", str(js)[:1200])
    st.stop()

# ── Aggregatie per winkel (WTD/period) ────────────────────────────────────────
g = df.groupby("shop_id", as_index=False).agg({
    "count_in": "sum",
    "turnover": "sum",
})
# Gewogen conversie en SPV (op basis van verkeer)
wa = df.groupby("shop_id").apply(
    lambda x: pd.Series({
        "conversion_rate": weighted_avg(x["conversion_rate"], x["count_in"]),
        "sales_per_visitor": weighted_avg(x["sales_per_visitor"], x["count_in"]),
    })
).reset_index()

base = g.merge(wa, on="shop_id", how="left")
base["shop_name"] = base["shop_id"].map(ID_TO_NAME)

# Als echt leeg na merge
if base.empty or base["count_in"].fillna(0).sum() == 0:
    st.info("Er is geen verkeer in de gekozen periode; scenario kan niet worden berekend.")
    with st.expander("🔧 Debug"):
        st.write("Params:", params)
        st.dataframe(df.head())
    st.stop()

# ── Baselines & scenario ──────────────────────────────────────────────────────
# Conversie uit API is in %-punten (0..100). Om naar fractie te gaan: /100
conv_f = pd.to_numeric(base["conversion_rate"], errors="coerce").fillna(0.0) / 100.0
spv    = pd.to_numeric(base["sales_per_visitor"], errors="coerce").fillna(0.0)
vis    = pd.to_numeric(base["count_in"], errors="coerce").fillna(0.0)

# ATV = SPV / conversie (als conversie > 0)
ATV = np.where(conv_f > 0, spv / conv_f, 0.0)

# Scenario: ATV ↑ (multiplicatief), conversie ↑ in +pp
new_conv_f = np.clip(conv_f + conv_add_pp, 0.0, 1.0)
new_ATV    = ATV * (1.0 + atv_uplift)
new_spv    = new_conv_f * new_ATV

delta_spv  = new_spv - spv
extra_rev  = vis * delta_spv                    # extra omzet
extra_gp   = extra_rev * gross_margin           # extra brutowinst

# Periode-naar-maanden voor payback schatting
days = pd.to_datetime(df.get("date", df.get("timestamp"))).dt.date.nunique()
months = max(1.0, (days or 30) / 30.44)

stores = len(base)
total_capex = capex * stores
monthly_gp = (extra_gp.sum()/months) if months>0 else extra_gp.sum()
payback_months = (total_capex / monthly_gp) if monthly_gp>0 else np.inf
roi_pct = ((extra_gp.sum() - total_capex) / total_capex * 100.0) if total_capex>0 else 0.0

# ── KPI-kaarten ───────────────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
c1.metric("📈 Extra omzet (scenario)", fmt_eur(extra_rev.sum()))
c2.metric("💵 Extra brutowinst", fmt_eur(extra_gp.sum()))
c3.metric("⏳ Payback",
          "∞ mnd" if np.isinf(payback_months) else f"{payback_months:.1f} mnd",
          delta=f"Target {payback_target} mnd")
c4.metric("📊 ROI", fmt_pct(roi_pct,1))

# ── Tabel top 10 op extra brutowinst ──────────────────────────────────────────
base = base.copy()
base["ATV"] = ATV
base["extra_revenue"] = extra_rev
base["extra_gross_profit"] = extra_gp

top = base.sort_values("extra_gross_profit", ascending=False).head(10)
top_view = top[["shop_name","count_in","sales_per_visitor","conversion_rate","ATV","extra_gross_profit"]].copy()
top_view.columns = ["Winkel","Bezoekers","SPV","Conversie","ATV","Extra brutowinst"]
top_view["Bezoekers"] = top_view["Bezoekers"].map(fmt_int)
top_view["SPV"] = top_view["SPV"].map(lambda x: f"€{x:,.2f}".replace(",", "."))
top_view["Conversie"] = top_view["Conversie"].map(lambda x: f"{x:.2f}%")
top_view["ATV"] = top_view["ATV"].map(lambda x: f"€{x:,.2f}".replace(",", "."))
top_view["Extra brutowinst"] = top_view["Extra brutowinst"].map(fmt_eur)

st.subheader("Stores (top 10 extra brutowinst)")
st.dataframe(top_view, use_container_width=True)

# ── Board tips ────────────────────────────────────────────────────────────────
st.subheader("🤖 Board Tips")
st.markdown(
    f"- **+{int(atv_uplift*100)}% ATV** en **+{int(conv_add_pp*100)}pp conversie** geven samen **{fmt_eur(extra_gp.sum())}** extra "
    f"brutowinst in ~{months:.1f} mnd; payback ≈ **{'∞' if np.isinf(payback_months) else f'{payback_months:.1f}'} mnd**."
)
st.markdown("- Richt CAPEX op winkels met **hoog verkeer maar lage SPV/conversie** (zie Region Performance).")

# ── Debug ─────────────────────────────────────────────────────────────────────
with st.expander("🔧 Debug"):
    st.write("Params:", params)
    st.dataframe(base[["shop_id","shop_name","count_in","turnover","conversion_rate","sales_per_visitor","ATV","extra_revenue","extra_gross_profit"]].head(20))
