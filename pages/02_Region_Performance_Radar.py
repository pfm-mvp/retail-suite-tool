import streamlit as st
import pandas as pd
import plotly.express as px
from shop_mapping import SHOP_NAME_MAP
from utils_pfmx import inject_css, api_get_report, normalize_vemcount_daylevel, fmt_eur, fmt_pct, friendly_error

st.set_page_config(page_title="Region Performance Radar", page_icon="🧭", layout="wide")
inject_css()

ids = list(SHOP_NAME_MAP.keys())
st.markdown("### 🎯 Targets (demo)")
t1, t2 = st.columns(2)
with t1: conv_target = st.slider("Conversie‑target (%)", 0, 50, 25, 1) / 100.0
with t2: spv_target = st.number_input("SPV‑target (€)", min_value=0, value=45, step=1)

params = [("source","shops"), ("period","last_month")]
for sid in ids: params.append(("data", saimport streamlit as st
import pandas as pd
import plotly.express as px
from shop_mapping import SHOP_NAME_MAP
from utils_pfmx import inject_css, api_get_report, normalize_vemcount_daylevel, friendly_error

st.set_page_config(page_title="Region Performance Radar", page_icon="🧭", layout="wide")
inject_css()

ids = list(SHOP_NAME_MAP.keys())

# POST /get-report
params = [("source","shops"), ("period","last_month")]
for sid in ids: params.append(("data", sid))
for k in ["count_in","conversion_rate","turnover","sales_per_visitor"]:
    params.append(("data_output", k))

js = api_get_report(params)
if friendly_error(js, "last_month"):
    st.stop()

df = normalize_vemcount_daylevel(js)
if df.empty:
    st.info("Geen data.")
    st.stop()

agg = df.groupby("shop_id", as_index=False).agg(
    count_in=("count_in","sum"),
    conversion_rate=("conversion_rate","mean"),
    sales_per_visitor=("sales_per_visitor","mean"),
    turnover=("turnover","sum"),
)
agg["name"] = agg["shop_id"].map(SHOP_NAME_MAP)

fig = px.scatter(
    agg, x="conversion_rate", y="sales_per_visitor", size="count_in", hover_name="name",
    labels={"conversion_rate":"Conversie","sales_per_visitor":"SPV","count_in":"Visitors"},
)
st.plotly_chart(fig, use_container_width=True)
id))
for k in ["count_in","conversion_rate","turnover","sales_per_visitor"]:
    params.append(("data_output", k))

js = api_get_report(params, st.secrets["API_URL"])
if friendly_error(js, "last_month"):
    st.stop()

df = normalize_vemcount_daylevel(js)
if df.empty:
    st.info("Geen data.")
    st.stop()

agg = df.groupby("shop_id", as_index=False).agg(
    count_in=("count_in","sum"),
    conversion_rate=("conversion_rate","mean"),
    sales_per_visitor=("sales_per_visitor","mean"),
    turnover=("turnover","sum"),
)

agg["name"] = agg["shop_id"].map(SHOP_NAME_MAP)
fig = px.scatter(
    agg, x="conversion_rate", y="sales_per_visitor", size="count_in", hover_name="name",
    labels={"conversion_rate":"Conversie","sales_per_visitor":"SPV","count_in":"Visitors"},
)
st.plotly_chart(fig, use_container_width=True)
