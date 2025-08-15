
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
from datetime import datetime, time
from zoneinfo import ZoneInfo

# Ensure root on path (shop_mapping in root)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shop_mapping import SHOP_NAME_MAP
from utils_pfmx import api_get_report
from helpers_normalize import normalize_vemcount_response, to_wide

st.set_page_config(layout="wide")
TZ = ZoneInfo("Europe/Amsterdam")

st.title("📈 Store Daily Trend")

names = [SHOP_NAME_MAP[sid] for sid in SHOP_NAME_MAP]
name2id = {v:k for k,v in SHOP_NAME_MAP.items()}
sname = st.selectbox("Winkel", names, index=0)
shop_id = name2id[sname]

kpis = ["count_in","conversion_rate","turnover","sales_per_visitor"]
period = st.selectbox("Periode", ["this_month","last_month","last_week","this_quarter","this_year"], index=0)

params = [("source","shops"),("period",period),("data",[shop_id]),("data_output",kpis)]
with st.spinner("Data ophalen..."):
    resp = api_get_report(params, prefer_brackets=True)
df = normalize_vemcount_response(resp, SHOP_NAME_MAP)

st.dataframe(df)

# Simple charts (only if columns present)
for k in kpis:
    if k in df.columns and not df.empty:
        st.line_chart(df.set_index("date")[k], height=220)
