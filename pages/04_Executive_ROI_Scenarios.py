import streamlit as st
import pandas as pd
import numpy as np
from shop_mapping import SHOP_NAME_MAP
from utils_pfmx import inject_css, api_get_report, normalize_vemcount_daylevel, friendly_error

st.set_page_config(page_title="Executive ROI Scenarios", page_icon="💼", layout="wide")
inject_css()

ids = list(SHOP_NAME_MAP.keys())
period = st.selectbox("Periode", ["last_month","this_quarter","last_quarter","this_year","last_year"], index=0)

# Demo inputs
c1,c2,c3,c4 = st.columns(4)
with c1: conv_add = st.slider("Conversie uplift (+pp)", 0.0, 0.20, 0.05, 0.01)
with c2: spv_uplift = st.slider("SPV‑uplift (%)", 0, 50, 10, 1) / 100.0
with c3: gross_margin = st.slider("Brutomarge (%)", 20, 80, 55, 1) / 100.0
with c4: capex = st.number_input("CAPEX per store (€)", min_value=0, value=1500, step=100)

payback_target = st.slider("Payback‑target (mnd)", 6, 24, 12, 1)

params = [("source","shops"), ("period", period)]
for sid in ids: params.append(("data", sid))
for k in ["count_in","conversion_rate","turnover","sales_per_visitor"]:
    params.append(("data_output", k))

js = api_get_report(params)
if friendly_error(js, period):
    st.stop()

df = normalize_vemcount_daylevel(js)
st.dataframe(df.head(50))
