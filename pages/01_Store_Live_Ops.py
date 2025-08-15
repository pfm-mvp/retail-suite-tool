
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

st.title("👣 Store Live Ops — vandaag vs. gisteren")

names = list(NAME_TO_ID.keys())
sname = st.selectbox("Winkel", names, index=0)
shop_id = NAME_TO_ID[sname]

open_t, close_t = st.slider("Venster vandaag & gisteren", value=(time(9,0), time(18,0)), format="HH:mm")
now = datetime.now(TZ).time()
now_cut = max(open_t, min(now, close_t))
h_from = open_t.strftime("%H:%M")
h_to   = now_cut.strftime("%H:%M")

# Plain keys, repeated
base_params = [
    ("data", shop_id),
    ("data_output", "count_in"),
    ("source","shops"),
    ("show_hours_from", h_from),
    ("show_hours_to",   h_to),
]

params_today = [("period","day")] + base_params
params_yest  = [("period","yesterday")] + base_params

with st.spinner("Data ophalen..."):
    js_today = api_get_report(params_today)
    js_yest  = api_get_report(params_yest)

if friendly_error(js_today, "day") or friendly_error(js_yest, "yesterday"):
    st.stop()

df_today = normalize_vemcount_response(js_today, SHOP_NAME_MAP)
df_yest  = normalize_vemcount_response(js_yest,  SHOP_NAME_MAP)

v_today = int(float(df_today["count_in"].sum())) if not df_today.empty and "count_in" in df_today else 0
v_yest  = int(float(df_yest["count_in"].sum()))  if not df_yest.empty  and "count_in" in df_yest  else 0
delta = v_today - v_yest
pct = (delta / v_yest * 100.0) if v_yest > 0 else 0.0

c1,c2,c3 = st.columns(3)
c1.metric("Vandaag (tot nu)", f"{v_today:,}".replace(",","."))
c2.metric("Gisteren (zelfde venster)", f"{v_yest:,}".replace(",","."))
c3.metric("Verschil", f"{delta:+,}".replace(",","."), f"{pct:.1f}%")

with st.expander("🔧 Debug"):
    st.code(params_today)
    st.code(params_yest)
