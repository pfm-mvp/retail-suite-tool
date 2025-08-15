
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

st.title("📡 Region Performance Radar")

sel_names = st.multiselect("Winkels", list(NAME_TO_ID.keys()), default=list(NAME_TO_ID.keys())[:3])
shop_ids = [NAME_TO_ID[n] for n in sel_names] or [next(iter(ID_TO_NAME.keys()))]

kpis = st.multiselect("KPI's", ["count_in","conversion_rate","turnover","sales_per_visitor"],
                      default=["count_in","conversion_rate","turnover","sales_per_visitor"])

period = st.selectbox("Periode", ["last_week","this_month","last_month","this_quarter","last_quarter","this_year","last_year"], index=0)

# Build plain repeated params
params = []
for sid in shop_ids: params.append(("data", sid))
for k in kpis: params.append(("data_output", k))
params += [("source","shops"), ("period", period)]

with st.spinner("Data ophalen..."):
    js = api_get_report(params)

if friendly_error(js, period):
    st.stop()

df = normalize_vemcount_response(js, SHOP_NAME_MAP)
st.dataframe(df)
