
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

st.title("📈 Store Daily Trend")

sname = st.selectbox("Winkel", list(NAME_TO_ID.keys()), index=0)
shop_id = NAME_TO_ID[sname]
kpis = ["count_in","conversion_rate","turnover","sales_per_visitor"]
period = st.selectbox("Periode", ["this_month","last_month","last_week","this_quarter","this_year"], index=0)

params = [("data", shop_id)]
for k in kpis: params.append(("data_output", k))
params += [("source","shops"), ("period", period)]

with st.spinner("Data ophalen..."):
    js = api_get_report(params)

if friendly_error(js, period):
    st.stop()

df = normalize_vemcount_response(js, SHOP_NAME_MAP)
st.dataframe(df)

for k in kpis:
    if k in df.columns and not df.empty:
        st.line_chart(df.set_index("date")[k], height=220)
