# pages/99_Ghost_API_Smoke_Test.py (uses no-encode helper + friendly_error)
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shop_mapping import SHOP_NAME_MAP
from utils_pfmx import api_get_report, friendly_error
from helpers_normalize import normalize_vemcount_response

st.title("👻 Ghost / API Smoke Test (no-encode brackets)")
sid = next(iter(SHOP_NAME_MAP.keys()))

tests = {
    "A_single_shop_single_kpi": [
        ("source","shops"),("period","day"),
        ("data",[sid]),("data_output",["count_in"])
    ],
    "B_single_shop_multi_kpi": [
        ("source","shops"),("period","last_week"),
        ("data",[sid]),("data_output",["count_in","conversion_rate","turnover","sales_per_visitor"])
    ],
    "C_multi_shop_multi_kpi": [
        ("source","shops"),("period","yesterday"),
        ("data",list(SHOP_NAME_MAP.keys())[:3]),("data_output",["count_in","turnover"])
    ]
}

for name, params in tests.items():
    st.subheader(name)
    with st.spinner(f"Calling {name} ..."):
        js = api_get_report(params, prefer_brackets=True)
    # Show raw params
    st.code(repr(params))
    # Error?
    if friendly_error(js):
        continue
    # OK → normalize
    df = normalize_vemcount_response(js, SHOP_NAME_MAP)
    st.json({"rows": 0 if df is None else len(df), "has_data_key": "data" in js})
    if df is not None and not df.empty:
        st.dataframe(df.head(10))
