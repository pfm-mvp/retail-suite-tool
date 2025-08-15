
# pages/04_Executive_ROI_Scenarios.py
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

# --- ensure root on path to import root modules ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shop_mapping import SHOP_NAME_MAP
from utils_pfmx import api_get_report
from helpers_normalize import normalize_vemcount_response, to_wide

st.set_page_config(page_title="Executive ROI Scenarios", page_icon="💼", layout="wide")

st.title("💼 Executive ROI Scenarios")
st.caption("Simuleer omzet- en brutowinstuplift per winkel op basis van conversie- en SPV-scenario's.")

TZ = ZoneInfo("Europe/Amsterdam")

# --- Controls ---
ids = list(SHOP_NAME_MAP.keys())
period = st.selectbox("Periode", ["last_month","this_quarter","last_quarter","this_year","last_year"], index=0)

c1,c2,c3,c4 = st.columns(4)
with c1:
    conv_add = st.slider("Conversie uplift (+pp)", 0.00, 0.20, 0.05, 0.01)
with c2:
    spv_uplift = st.slider("SPV‑uplift (%)", 0, 50, 10, 1) / 100.0
with c3:
    gross_margin = st.slider("Brutomarge (%)", 10, 90, 55, 1) / 100.0
with c4:
    capex = st.number_input("CAPEX per store (€)", min_value=0, value=1500, step=100)

payback_target = st.slider("Payback‑target (mnd)", 6, 24, 12, 1)

# --- Fetch data (multi-shop, multi-KPI) ---
params = [
    ("source","shops"),
    ("period", period),
    ("data", ids),
    ("data_output", ["count_in","conversion_rate","turnover","sales_per_visitor"]),
]
with st.spinner("Data ophalen..."):
    js = api_get_report(params, prefer_brackets=True)

df = normalize_vemcount_response(js, SHOP_NAME_MAP)
if df.empty:
    st.error("Geen data ontvangen voor de gekozen periode.")
    st.stop()

# Gebruik dag-totalen per winkel voor de gekozen periode
wide = to_wide(df, index=["shop_id","shop_name"])

# Zorg dat de KPI-kolommen bestaan
for col in ["count_in","conversion_rate","turnover","sales_per_visitor"]:
    if col not in wide.columns:
        wide[col] = 0.0

# --- Scenario berekening ---
res = wide.copy()
# Prevent divide-by-zero: als conversie 0 is, neem een kleine epsilon om SPV/conv te kunnen delen
eps = 1e-9
res["conv_safe"] = res["conversion_rate"].clip(lower=eps)
# ATV afleiden uit SPV (= conv * ATV)
res["ATV_est"] = res["sales_per_visitor"] / res["conv_safe"]
# Nieuwe SPV = (conv + conv_add) * ATV * (1 + spv_uplift)
res["new_spv"] = (res["conv_safe"] + conv_add) * res["ATV_est"] * (1.0 + spv_uplift)
res["new_spv"] = res["new_spv"].fillna(0.0)

# Baseline omzet via turnover indien aanwezig, anders herleiden via count_in * SPV
res["baseline_turnover_calc"] = res["count_in"] * res["sales_per_visitor"]
res["baseline_turnover"] = res["turnover"].where(res["turnover"].notna() & (res["turnover"]>0), res["baseline_turnover_calc"])

# Nieuwe omzet op basis van count_in * nieuwe SPV
res["new_turnover"] = res["count_in"] * res["new_spv"]
res["extra_turnover"] = (res["new_turnover"] - res["baseline_turnover"]).clip(lower=0.0)

# Brutowinstuplift
res["extra_gross_profit"] = res["extra_turnover"] * gross_margin

# Periode in maanden schatten op basis van datumbereik
# Haal min/max datum uit het oorspronkelijke df
try:
    df_dates = pd.to_datetime(df["date"].dropna(), errors="coerce")
    if df_dates.notna().any():
        days = (df_dates.max() - df_dates.min()).days + 1
        months_in_period = max(days / 30.44, 0.01)
    else:
        months_in_period = 1.0
except Exception:
    months_in_period = 1.0

# Maandelijkse brutowinst uplift
res["monthly_gp_uplift"] = res["extra_gross_profit"] / months_in_period

# Payback (maanden)
res["payback_months"] = res.apply(lambda r: (capex / r["monthly_gp_uplift"]) if r["monthly_gp_uplift"] > 0 else None, axis=1)

# KPI weergave & formatting
view_cols = [
    "shop_name","count_in","conversion_rate","sales_per_visitor",
    "baseline_turnover","new_turnover","extra_turnover",
    "extra_gross_profit","monthly_gp_uplift","payback_months"
]
res_view = res[view_cols].copy()

# Format helper
def eur(x): 
    try:
        return f"€{x:,.0f}".replace(",", ".")
    except Exception:
        return x
def pct(x):
    try:
        return f"{x*100:,.1f}%".replace(",", ".")
    except Exception:
        return x
def months(x):
    if x is None or pd.isna(x) or x==float("inf"):
        return "—"
    try:
        return f"{x:,.1f}".replace(",", ".")
    except Exception:
        return x

# Totals
tot = pd.DataFrame({
    "shop_name": ["Totaal"],
    "count_in": [res["count_in"].sum()],
    "conversion_rate": [res["conversion_rate"].mean()],
    "sales_per_visitor": [res["sales_per_visitor"].mean()],
    "baseline_turnover": [res["baseline_turnover"].sum()],
    "new_turnover": [res["new_turnover"].sum()],
    "extra_turnover": [res["extra_turnover"].sum()],
    "extra_gross_profit": [res["extra_gross_profit"].sum()],
    "monthly_gp_uplift": [res["monthly_gp_uplift"].sum()],
    "payback_months": [capex / (res["monthly_gp_uplift"].sum() or 1e-9)]
})

disp = pd.concat([res_view, tot], ignore_index=True)

# Pretty print
disp_fmt = disp.copy()
disp_fmt["conversion_rate"] = disp_fmt["conversion_rate"].apply(pct)
disp_fmt["sales_per_visitor"] = disp_fmt["sales_per_visitor"].apply(eur)
for c in ["baseline_turnover","new_turnover","extra_turnover","extra_gross_profit","monthly_gp_uplift"]:
    disp_fmt[c] = disp_fmt[c].apply(eur)
disp_fmt["payback_months"] = disp["payback_months"].apply(months)

st.subheader("Scenario-resultaat per winkel")
st.dataframe(disp_fmt, use_container_width=True)

# KPI-cards
total_extra_turnover = res["extra_turnover"].sum()
total_extra_gp = res["extra_gross_profit"].sum()
portfolio_monthly_gp = res["monthly_gp_uplift"].sum()
portfolio_payback = (capex * len(ids)) / portfolio_monthly_gp if portfolio_monthly_gp > 0 else None

c1,c2,c3,c4 = st.columns(4)
c1.metric("Extra omzet (periode)", eur(total_extra_turnover))
c2.metric("Extra brutowinst (periode)", eur(total_extra_gp))
c3.metric("Maandelijkse brutowinst uplift", eur(portfolio_monthly_gp))
c4.metric("Payback (maanden, totaal)", months(portfolio_payback))

# Alert vs target
if portfolio_payback and portfolio_payback <= payback_target:
    st.success(f"✅ Portfolio payback ({months(portfolio_payback)} mnd) ligt binnen de target van {payback_target} mnd.")
else:
    st.warning(f"⚠️ Portfolio payback ({months(portfolio_payback)} mnd) is boven de target van {payback_target} mnd.")
