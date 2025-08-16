# pages/01_Store_Live_Ops_with_leaderboard.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

from shop_mapping import SHOP_NAME_MAP

st.set_page_config(page_title="Store Live Ops Leaderboard", page_icon="📊", layout="wide")

API_URL = st.secrets["API_URL"]

# ========================
# Helpers
# ========================

def fetch_report(shop_ids, period="last_week", step="day", metrics=None):
    if metrics is None:
        metrics = ["count_in", "conversion_rate", "turnover", "sales_per_visitor"]

    params = [("data", sid) for sid in shop_ids]
    params += [("data_output", m) for m in metrics]
    params += [
        ("source", "shops"),
        ("period", period),
        ("step", step),
    ]

    r = requests.post(API_URL, params=params, timeout=40)
    r.raise_for_status()
    return r.json()


def normalize_vemcount_daylevel(js, shop_ids, metrics):
    """Parse nested vemcount response into flat DataFrame with 'date' col."""
    rows = []
    for date_key, shops in js.get("data", {}).items():
        if not date_key.startswith("date_"):
            continue
        date = date_key.replace("date_", "")
        for sid in shop_ids:
            values = shops.get(str(sid), {})
            row = {"date": date, "shop_id": sid}
            for m in metrics:
                row[m] = values.get(m, None)
            rows.append(row)
    return pd.DataFrame(rows)


def fmt_delta(val, is_pct=False, is_eur=False):
    if pd.isna(val):
        return "–"
    arrow = "▲" if val > 0 else "▼"
    color = "green" if val > 0 else "red"
    if is_pct:
        return f"<span style='color:{color};font-weight:600'>{arrow} {val:.1f}%</span> tov dag ervoor"
    if is_eur:
        return f"<span style='color:{color};font-weight:600'>{arrow} €{val:,.0f}</span> tov dag ervoor".replace(",", ".")
    return f"<span style='color:{color};font-weight:600'>{arrow} {val:.0f}</span> tov dag ervoor"


# ========================
# UI
# ========================

NAME_TO_ID = {v: k for k, v in SHOP_NAME_MAP.items()}
ID_TO_NAME = {k: v for k, v in SHOP_NAME_MAP.items()}

selected_name = st.selectbox("Selecteer je vestiging", list(NAME_TO_ID.keys()), index=0)
shop_id = NAME_TO_ID[selected_name]

metrics = ["count_in", "conversion_rate", "turnover", "sales_per_visitor"]

# Fetch laatste week (dag-niveau)
js = fetch_report([shop_id] + list(SHOP_NAME_MAP.keys()), period="last_week", step="day", metrics=metrics)
df = normalize_vemcount_daylevel(js, [shop_id] + list(SHOP_NAME_MAP.keys()), metrics)

if df.empty:
    st.error("Geen data gevonden")
    st.stop()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# ========================
# Yesterday vs Day before
# ========================

yesterday = (datetime.today() - timedelta(days=1)).date()
day_before = (datetime.today() - timedelta(days=2)).date()

df_y = df[df["date"].dt.date == yesterday]
df_d = df[df["date"].dt.date == day_before]

row_y = df_y[df_y["shop_id"] == shop_id]
row_d = df_d[df_d["shop_id"] == shop_id]

if row_y.empty or row_d.empty:
    st.warning("Niet genoeg data om gisteren vs dag ervoor te vergelijken.")
    st.stop()

row_y = row_y.iloc[0]
row_d = row_d.iloc[0]

# Deltas
delta_visitors = row_y["count_in"] - row_d["count_in"]
delta_conv = (row_y["conversion_rate"] - row_d["conversion_rate"]) * 100
delta_turnover = row_y["turnover"] - row_d["turnover"]
delta_spv = row_y["sales_per_visitor"] - row_d["sales_per_visitor"]

# ========================
# KPI Cards
# ========================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("👥 Bezoekers gisteren", f"{int(row_y['count_in']):,}".replace(",", "."),
              delta=None)
    st.markdown(fmt_delta(delta_visitors), unsafe_allow_html=True)

with c2:
    st.metric("🎯 Conversie gisteren", f"{row_y['conversion_rate']*100:.1f}%",
              delta=None)
    st.markdown(fmt_delta(delta_conv, is_pct=True), unsafe_allow_html=True)

with c3:
    st.metric("💶 Omzet gisteren", f"€{row_y['turnover']:,.0f}".replace(",", "."),
              delta=None)
    st.markdown(fmt_delta(delta_turnover, is_eur=True), unsafe_allow_html=True)

with c4:
    st.metric("🛍️ Sales/Visitor gisteren", f"€{row_y['sales_per_visitor']:,.2f}".replace(",", "."),
              delta=None)
    st.markdown(fmt_delta(delta_spv, is_eur=True), unsafe_allow_html=True)

# ========================
# Leaderboard
# ========================

st.subheader("🏆 Leaderboard")

toggle_metric = st.radio("Rangschik op:", ["conversion_rate", "sales_per_visitor"], horizontal=True)

# Pak gisteren per store
df_rank = df[df["date"].dt.date == yesterday].copy()
df_rank["store_name"] = df_rank["shop_id"].map(ID_TO_NAME)

df_rank = df_rank[["store_name", "conversion_rate", "sales_per_visitor", "turnover", "count_in"]]

df_rank["Rank"] = df_rank[toggle_metric].rank(ascending=False, method="min").astype(int)
df_rank = df_rank.sort_values("Rank")

# Markeer geselecteerde store
def highlight_row(row):
    if row["store_name"] == selected_name:
        return [f"background-color:#762181; color:white; font-weight:600"] * len(row)
    return [""] * len(row)

styler = df_rank.style.apply(highlight_row, axis=1)

st.dataframe(styler, use_container_width=True)
