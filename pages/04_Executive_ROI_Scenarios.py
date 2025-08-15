
import os, sys
from pathlib import Path
import streamlit as st
import pandas as pd
from datetime import datetime, time
from zoneinfo import ZoneInfo

# Ensure root importable (shop_mapping.py in root)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shop_mapping import SHOP_NAME_MAP
from utils_pfmx import api_get_report, friendly_error, inject_css
from helpers_normalize import normalize_vemcount_response

st.set_page_config(layout="wide")
inject_css()
TZ = ZoneInfo("Europe/Amsterdam")
NAME_TO_ID = {v:k for k,v in SHOP_NAME_MAP.items()}
ID_TO_NAME = {k:v for k,v in SHOP_NAME_MAP.items()}

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

# Build params (plain keys)
params = []
for sid in ids: params.append(("data", sid))
for k in ["count_in","conversion_rate","turnover","sales_per_visitor"]:
    params.append(("data_output", k))
params += [("source","shops"), ("period", period)]

js = api_get_report(params)
if friendly_error(js, period):
    st.stop()

df = normalize_vemcount_response(js, SHOP_NAME_MAP)
st.dataframe(df.head(50))

# (voorbeeld vervolg: scenario berekening kan hierop voortbouwen; df bevat KPI's per dag/shop)
