
import os, sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

# Ensure root importable (shop_mapping.py in root)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shop_mapping import SHOP_NAME_MAP
from utils_pfmx import api_get_report, friendly_error, inject_css
from helpers_normalize import normalize_vemcount_response, to_wide

st.set_page_config(layout="wide")
inject_css()
TZ = ZoneInfo("Europe/Amsterdam")
NAME_TO_ID = {v:k for k,v in SHOP_NAME_MAP.items()}
ID_TO_NAME = {k:v for k,v in SHOP_NAME_MAP.items()}

def fmt_int(n): 
    try: return f"{int(round(float(n))):,}".replace(",", ".")
    except: return "0"

def fmt_eur(n):
    try: return f"€{float(n):,.0f}".replace(",", ".")
    except: return "€0"

def fmt_pct(x, digits=1):
    try: return f"{float(x):.{digits}f}%"
    except: return "0.0%"

import numpy as np
st.title("💼 Executive ROI Scenarios")

ids = list(SHOP_NAME_MAP.keys())
period = st.selectbox("Periode", ["last_month","this_quarter","last_quarter","this_year","last_year"], index=0)

c1,c2,c3,c4 = st.columns(4)
with c1: conv_add = st.slider("Conversie uplift (+pp)", 0.0, 0.20, 0.05, 0.01)
with c2: spv_uplift = st.slider("SPV‑uplift (%)", 0, 50, 10, 1) / 100.0
with c3: gross_margin = st.slider("Brutomarge (%)", 20, 80, 55, 1) / 100.0
with c4: capex = st.number_input("CAPEX per store (€)", min_value=0, value=1500, step=100)
payback_target = st.slider("Payback‑target (mnd)", 6, 24, 12, 1)

params = []
for sid in ids: params.append(("data", sid))
for k in ["count_in","conversion_rate","turnover","sales_per_visitor"]:
    params.append(("data_output", k))
params += [("source","shops"), ("period", period)]

js = api_get_report(params)
if friendly_error(js, period): st.stop()

df = normalize_vemcount_response(js, SHOP_NAME_MAP)
wide = to_wide(df)  # one row per date+shop

# Baselines per store (sum over period)
base = wide.groupby(["shop_id","shop_name"], as_index=False).agg({
    "count_in":"sum",
    "turnover":"sum",
    "conversion_rate":"mean",
    "sales_per_visitor":"mean"
})

# Compute ATV from SPV and conversion: SPV = conv * ATV  =>  ATV = SPV/conv
base["ATV"] = np.where(base["conversion_rate"]>0, base["sales_per_visitor"]/base["conversion_rate"], 0.0)

# Scenario
new_conv = base["conversion_rate"] + (conv_add*100)  # conv in % units from API (looks like 0-100), adjust if 0-1
# If API returns 0-100, keep as is; SPV is absolute €, uplift multiplicative:
new_spv = base["sales_per_visitor"] * (1 + spv_uplift)

# New revenue per visitor = new_spv; extra revenue = count_in * (new_spv - current_spv)
extra_rev = base["count_in"] * (new_spv - base["sales_per_visitor"])
extra_gp  = extra_rev * gross_margin

stores = len(base)
total_capex = capex * stores
# Rough months in period: infer from number of unique dates; fallback 1 month
days = df["date"].nunique() if "date" in df.columns else 30
months = max(1.0, days/30.44)
monthly_gp = extra_gp.sum()/months if months>0 else extra_gp.sum()
payback_months = (total_capex / monthly_gp) if monthly_gp>0 else np.inf
roi_pct = (extra_gp.sum() - total_capex) / total_capex * 100 if total_capex>0 else 0

c1,c2,c3,c4 = st.columns(4)
c1.metric("📈 Extra omzet (scenario)", fmt_eur(extra_rev.sum()))
c2.metric("💵 Extra brutowinst", fmt_eur(extra_gp.sum()))
c3.metric("⏳ Payback", "∞ mnd" if np.isinf(payback_months) else f"{payback_months:.1f} mnd",
          delta=f"Target {payback_target} mnd")
c4.metric("📊 ROI", fmt_pct(roi_pct,1))

st.subheader("Stores (top 10 extra brutowinst)")
base["extra_revenue"] = extra_rev
base["extra_gross_profit"] = extra_gp
top = base.sort_values("extra_gross_profit", ascending=False).head(10)
top["extra_gross_profit_fmt"] = top["extra_gross_profit"].map(fmt_eur)
st.dataframe(top[["shop_name","count_in","sales_per_visitor","conversion_rate","ATV","extra_gross_profit_fmt"]], use_container_width=True)

st.subheader("🤖 Board Tips")
st.markdown(f"- **+{spv_uplift*100:.0f}% SPV** levert **{fmt_eur(extra_gp.sum())}** extra brutowinst op in {months:.1f} mnd; payback ≈ **{('∞' if np.isinf(payback_months) else f'{payback_months:.1f}')} mnd**.")
st.markdown("- Overweeg CAPEX te richten op winkels met **hoog verkeer maar lage SPV/conversie** (zie Region Performance).")

with st.expander("🔧 Debug"):
    st.code(params)
    st.dataframe(base.head(10))
