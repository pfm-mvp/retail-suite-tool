import sys
from pathlib import Path
import streamlit as st
from datetime import datetime
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shop_mapping import SHOP_NAME_MAP
from utils_pfmx import api_get_report, friendly_error
from helpers_normalize import normalize_vemcount_response

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
        if friendly_error(r):
            st.code(p)
            continue
        df = normalize_vemcount_response(r, SHOP_NAME_MAP)
    st.success(f"OK • rows: {0 if df is None else len(df)}")
    st.dataframe(df.head(10))
