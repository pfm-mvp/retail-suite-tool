
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

st.title("👣 Store Live Ops")

# Store selector
names = [SHOP_NAME_MAP[sid] for sid in SHOP_NAME_MAP]
name2id = {v:k for k,v in SHOP_NAME_MAP.items()}
sname = st.selectbox("Winkel", names, index=0)
shop_id = name2id[sname]

# Opening hours slider
open_t, close_t = st.slider("Venster vandaag & gisteren", value=(time(9,0), time(18,0)), format="HH:mm")

now = datetime.now(TZ).time()
now_cut = max(open_t, min(now, close_t))
h_from = open_t.strftime("%H:%M")
h_to   = now_cut.strftime("%H:%M")

base = [("source","shops"),("data",[shop_id]),("data_output",["count_in"]),("show_hours_from",h_from),("show_hours_to",h_to)]

with st.spinner("Data ophalen..."):
    r_today = api_get_report([("period","day")] + base,  prefer_brackets=True)
    r_yest  = api_get_report([("period","yesterday")] + base, prefer_brackets=True)

df_today = normalize_vemcount_response(r_today, SHOP_NAME_MAP)
df_yest  = normalize_vemcount_response(r_yest,  SHOP_NAME_MAP)

val_today = int(float(df_today["count_in"].sum())) if not df_today.empty and "count_in" in df_today else 0
val_yest  = int(float(df_yest["count_in"].sum()))  if not df_yest.empty  and "count_in" in df_yest  else 0
delta = val_today - val_yest
pct = (delta/val_yest*100.0) if val_yest>0 else 0.0

c1,c2,c3 = st.columns(3)
c1.metric("Vandaag (tot nu)", f"{val_today:,}".replace(",","."))
c2.metric("Gisteren (zelfde venster)", f"{val_yest:,}".replace(",","."))
c3.metric("Verschil", f"{delta:+,}".replace(",","."), f"{pct:.1f}%")

with st.expander("Debug (API)"):
    st.json({"today_params":[("period","day")]+base, "today_resp_ok": "data" in r_today})
    st.json({"yesterday_params":[("period","yesterday")]+base, "yesterday_resp_ok": "data" in r_yest})
