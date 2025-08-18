# pages/04_Executive_ROI_Scenarios.py
import os, sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

# Ensure root importable (helpers in project root)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helpers_shop import ID_TO_NAME
from utils_pfmx import api_get_report, friendly_error, inject_css
from helpers_normalize import normalize_vemcount_response, to_wide

# ───────────────────────────────────────────────────────────────────────────────
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

# ── Inputs ─────────────────────────────────────────────────────────────────────
ids = list(ID_TO_NAME.keys())
period = st.selectbox(
    "Periode",
    ["last_month","this_quarter","last_quarter","this_year","last_year"],
    index=0
)

c1,c2,c3,c4 = st.columns(4)
with c1:
    # Conversie uplifts in percentagepunten (bijv. 6.0 = +6 pp)
    conv_add_pp = st.slider("Conversie uplift (+pp)", 0.0, 20.0, 6.0, 0.5)
with c2:
    # ATV-uplift in %, multiplicatief op ATV
    atv_uplift_pct = st.slider("ATV-uplift (%)", 0, 50, 10, 1) / 100.0
with c3:
    gross_margin = st.slider("Brutomarge (%)", 20, 80, 55, 1) / 100.0
with c4:
    capex = st.number_input("CAPEX per store (€)", min_value=0, value=1500, step=100)

payback_target = st.slider("Payback-target (mnd)", 6, 24, 12, 1)

# ── Data ophalen (robust) ──────────────────────────────────────────────────────
KPI_KEYS = ["count_in", "conversion_rate", "turnover", "sales_per_visitor"]

params = []
for sid in ids:
    params.append(("data", sid))
for k in KPI_KEYS:
    params.append(("data_output", k))
# dagresolutie helpt bij vaste aggregaties
params += [("source", "shops"), ("period", period), ("step", "day")]

js = api_get_report(params)
if friendly_error(js, period):
    st.stop()

# normalize expects id->name map
df = normalize_vemcount_response(js, ID_TO_NAME, kpi_keys=KPI_KEYS)

if df is None or df.empty:
    st.warning("Geen data ontvangen voor deze periode/parameters.")
    with st.expander("🔧 Debug"):
        st.write("Params:", params)
        st.write("Normalize → empty DataFrame")
    st.stop()

# Zorg dat 'date' aanwezig is i.v.m. to_wide
if "date" not in df.columns:
    df["date"] = pd.to_datetime(df.get("timestamp"), errors="coerce").dt.date

# Safety: drop rows zonder shop_id
if "shop_id" in df.columns:
    df = df[pd.notna(df["shop_id"])]

# Eén rij per datum × winkel
wide = to_wide(df)

# ── Debug (optioneel) ──────────────────────────────────────────────────────────
with st.expander("🔧 Debug — fetch result"):
    st.write("Rows:", len(df), "Shops:", df.get("shop_id", pd.Series(dtype=float)).nunique())
    st.dataframe(df.head(10))

# ── Baselines per winkel ───────────────────────────────────────────────────────
base = wide.groupby(["shop_id","shop_name"], as_index=False).agg({
    "count_in": "sum",
    "turnover": "sum",
    "conversion_rate": "mean",       # let op: in % (bijv. 36.5)
    "sales_per_visitor": "mean"      # SPV in €
})

# Conversie in fractie (0–1) en ATV afleiden
# (SPV = conversie_f * ATV  →  ATV = SPV / conversie_f)
conv_f_base = (base["conversion_rate"] / 100.0).clip(lower=0.0, upper=1.0)
# voorkom delen door 0
base["ATV"] = np.where(conv_f_base > 0, base["sales_per_visitor"] / conv_f_base, 0.0)

# ── Scenario ───────────────────────────────────────────────────────────────────
# Compute ATV from SPV and conversion: SPV = conv * ATV  =>  ATV = SPV/conv
base["ATV"] = np.where(base["conversion_rate"] > 0,
                       base["sales_per_visitor"] / base["conversion_rate"],
                       0.0)

# ✅ Nieuwe conversie (in procentpunten, pp)
new_conv = base["conversion_rate"] + conv_add

# Nieuwe SPV = nieuwe conversie × ATV × (1 + SPV uplift)
new_spv = new_conv * base["ATV"] * (1 + spv_uplift)

# Extra omzet = aantal bezoekers × (nieuw SPV – huidig SPV)
extra_rev = base["count_in"] * (new_spv - base["sales_per_visitor"])

# Extra brutowinst
extra_gp = extra_rev * gross_margin

# ── Payback/ROI ────────────────────────────────────────────────────────────────
stores = len(base)
total_capex = capex * stores
days = df["date"].nunique() if "date" in df.columns else 30
months = max(1.0, days/30.44)
monthly_gp = extra_gp.sum()/months if months > 0 else extra_gp.sum()
payback_months = (total_capex / monthly_gp) if monthly_gp > 0 else np.inf
roi_pct = (extra_gp.sum() - total_capex) / total_capex * 100 if total_capex > 0 else 0

# ── KPI-tegel weergave ─────────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
c1.metric("📈 Extra omzet (scenario)", fmt_eur(extra_rev.sum()))
c2.metric("💵 Extra brutowinst", fmt_eur(extra_gp.sum()))
c3.metric("⏳ Payback", "∞ mnd" if np.isinf(payback_months) else f"{payback_months:.1f} mnd",
          delta=f"Target {payback_target} mnd")
c4.metric("📊 ROI", fmt_pct(roi_pct,1))

# ── Tabel top 10 ───────────────────────────────────────────────────────────────
st.subheader("Stores (top 10 extra brutowinst)")
base = base.copy()
base["extra_revenue"] = extra_rev
base["extra_gross_profit"] = extra_gp
top = base.sort_values("extra_gross_profit", ascending=False).head(10)

# weergave-kolommen
show = top[[
    "shop_name", "count_in", "sales_per_visitor", "conversion_rate", "ATV", "extra_gross_profit"
]].rename(columns={
    "shop_name": "Winkel",
    "count_in": "Bezoekers",
    "sales_per_visitor": "SPV",
    "conversion_rate": "Conversie",
    "ATV": "ATV",
    "extra_gross_profit": "Extra brutowinst"
})

show["Bezoekers"]       = show["Bezoekers"].map(fmt_int)
show["SPV"]             = show["SPV"].map(lambda x: f"€{x:,.2f}".replace(",", "."))
show["Conversie"]       = show["Conversie"].map(lambda x: f"{x:.2f}%")
show["ATV"]             = show["ATV"].map(lambda x: f"€{x:,.2f}".replace(",", "."))
show["Extra brutowinst"]= show["Extra brutowinst"].map(fmt_eur)

st.dataframe(show, use_container_width=True)

# ── Tips ───────────────────────────────────────────────────────────────────────
st.subheader("🤖 Board Tips")
st.markdown(
    f"- **+{conv_add_pp:.1f} pp conversie** en **+{atv_uplift_pct*100:.0f}% ATV** "
    f"leveren **{fmt_eur(extra_gp.sum())}** extra brutowinst op in {months:.1f} mnd; "
    f"payback ≈ **{('∞' if np.isinf(payback_months) else f'{payback_months:.1f}')} mnd**."
)
st.markdown("- Richt CAPEX op winkels met **hoog verkeer maar lage SPV/conversie** (zie Region Performance).")

# ── Debug ──────────────────────────────────────────────────────────────────────
with st.expander("🔧 Debug"):
    st.code(params)
    dbg = base[["shop_name","count_in","sales_per_visitor","conversion_rate","ATV"]].copy()
    dbg["conv_f_base"] = conv_f_base
    dbg["conv_f_new"]  = conv_f_new
    dbg["spv_base"]    = spv_base
    dbg["spv_new"]     = spv_new
    dbg["extra_rev"]   = extra_rev
    dbg["extra_gp"]    = extra_gp
    st.dataframe(dbg.head(10))
