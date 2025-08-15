
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

st.title("👻 Ghost / API Smoke Test")

sid = next(iter(SHOP_NAME_MAP.keys()))
tests = {
    "A_single_shop_single_kpi": [("source","shops"),("period","day"),("data",[sid]),("data_output",["count_in"])],
    "B_single_shop_multi_kpi":  [("source","shops"),("period","last_week"),("data",[sid]),("data_output",["count_in","conversion_rate","turnover","sales_per_visitor"])],
    "C_multi_shop_multi_kpi":   [("source","shops"),("period","yesterday"),("data",list(SHOP_NAME_MAP.keys())[:3]),("data_output",["count_in","turnover"])],
}

for name, p in tests.items():
    st.subheader(name)
    with st.spinner(f"Call {name}..."):
        r = api_get_report(p, prefer_brackets=True)
        ok = "data" in r
        df = normalize_vemcount_response(r, SHOP_NAME_MAP)
    st.json({"ok": ok, "rows": 0 if df is None else len(df)})
    if not df.empty:
        st.dataframe(df.head(10))
