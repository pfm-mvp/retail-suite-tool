# pages/01_Store_Live_Ops_with_leaderboard.py
import os, sys
import pandas as pd
import requests
import streamlit as st
import numpy as np
import plotly.express as px

# === Imports
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))
from shop_mapping import SHOP_NAME_MAP

st.set_page_config(page_title="Store Live Ops + Leaderboard", page_icon="📈", layout="wide")

# === Styling
PFM_PURPLE = "#762181"
PFM_RED = "#F04438"
PFM_GREEN = "#12B76A"
PFM_GRAY = "#6B7280"

def metric_card(title, value, diff, fmt="int", is_currency=False):
    """Render een card met waarde en vergelijking t.o.v. dag ervoor"""
    if diff > 0:
        arrow, color = "↑", PFM_GREEN
    elif diff < 0:
        arrow, color = "↓", PFM_RED
    else:
        arrow, color = "→", PFM_GRAY

    if fmt == "pct":
        val_str = f"{value:.1f}%"
        diff_str = f"{arrow} {abs(diff):.1f}% t.o.v. dag ervoor"
    elif is_currency:
        val_str = f"€{value:,.0f}".replace(",", ".")
        diff_str = f"{arrow} €{abs(diff):,.0f} t.o.v. dag ervoor".replace(",", ".")
    else:
        val_str = f"{int(value):,}".replace(",", ".")
        diff_str = f"{arrow} {abs(int(diff))} t.o.v. dag ervoor"

    st.markdown(
        f"""
        <div style="background-color:#F9FAFB;padding:1rem;border-radius:0.8rem;
                    border:1px solid #E5E7EB;min-width:180px">
          <div style="font-size:0.9rem;color:{PFM_GRAY};margin-bottom:0.3rem">{title}</div>
          <div style="font-size:1.6rem;font-weight:700">{val_str}</div>
          <div style="font-size:0.9rem;color:{color};font-weight:600">{diff_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# === API Call
API_URL = st.secrets["API_URL"]

def fetch_report(shop_ids, period, step, metrics):
    params = [("data", sid) for sid in shop_ids]
    params += [("data_output", m) for m in metrics]
    params += [("source","shops"), ("period",period), ("step",step)]
    r = requests.post(API_URL, params=params, timeout=40)
    r.raise_for_status()
    return r.json()

def normalize_day(js, shop_ids, metrics):
    rows=[]
    for sid in shop_ids:
        if "data" not in js: continue
        recs = js["data"].get(str(sid),{}).get("dates",{})
        for date,v in recs.items():
            row={"shop_id":sid,"date":date}
            for m in metrics:
                row[m]=v["data"].get(m,np.nan)
            rows.append(row)
    return pd.DataFrame(rows)

# === UI
shop_options = list(SHOP_NAME_MAP.values())
default_shop = shop_options[0]
selected_shop = st.selectbox("Selecteer vestiging", shop_options, index=0)
selected_id = [k for k,v in SHOP_NAME_MAP.items() if v==selected_shop][0]

# === Data ophalen: gisteren & eergisteren
metrics = ["count_in","conversion_rate","turnover","sales_per_visitor"]
js = fetch_report(list(SHOP_NAME_MAP.keys()), "last_week", "day", metrics)
df = normalize_day(js, list(SHOP_NAME_MAP.keys()), metrics)

df["date"]=pd.to_datetime(df["date"])
df=df.sort_values("date")

# Pak gisteren & dag ervoor
last_date = df["date"].max()
yesterday = last_date
day_before = last_date - pd.Timedelta(days=1)
df_y = df[df["date"]==yesterday]
df_d = df[df["date"]==day_before]

# Metrics voor geselecteerde winkel
my_y = df_y[df_y["shop_id"]==selected_id].set_index("shop_id")
my_d = df_d[df_d["shop_id"]==selected_id].set_index("shop_id")

# === Cards
st.subheader(f"📊 Store metrics — {selected_shop}")
c1,c2,c3,c4=st.columns(4)
with c1:
    metric_card("Bezoekers gisteren",
                my_y["count_in"].values[0],
                my_y["count_in"].values[0]-my_d["count_in"].values[0])
with c2:
    metric_card("Conversie",
                my_y["conversion_rate"].values[0]*100,
                (my_y["conversion_rate"].values[0]-my_d["conversion_rate"].values[0])*100,
                fmt="pct")
with c3:
    metric_card("Omzet",
                my_y["turnover"].values[0],
                my_y["turnover"].values[0]-my_d["turnover"].values[0],
                is_currency=True)
with c4:
    metric_card("Sales/Visitor",
                my_y["sales_per_visitor"].values[0],
                my_y["sales_per_visitor"].values[0]-my_d["sales_per_visitor"].values[0],
                is_currency=True)

# === Leaderboard
st.subheader("🏆 Leaderboard (week t/m gisteren)")
mode = st.radio("Ranking op basis van:", ["Conversie","Sales per Visitor"], horizontal=True)

df_lb = df_y.copy()
df_lb["store_name"]=df_lb["shop_id"].map(SHOP_NAME_MAP)

if mode=="Conversie":
    df_lb["rank_metric"]=df_lb["conversion_rate"]
else:
    df_lb["rank_metric"]=df_lb["sales_per_visitor"]

df_lb=df_lb.sort_values("rank_metric",ascending=False)
df_lb["Rank"]=range(1,len(df_lb)+1)

# Vergelijk met dag ervoor
df_cmp=df_d[["shop_id",mode.lower().replace(" ","_")]].rename(columns={mode.lower().replace(" ","_"):"prev"})
df_lb=df_lb.merge(df_cmp,on="shop_id",how="left")
df_lb["PosΔ"]=df_lb["Rank"].diff().fillna(0).astype(int)

def highlight(val, sid, my_sid):
    if sid==my_sid:
        return f"background-color:{PFM_PURPLE};color:white;font-weight:700"
    return ""

styled = df_lb[["Rank","store_name","rank_metric"]].style.apply(
    lambda s:[highlight(v, df_lb.loc[s.name,"shop_id"], selected_id) for v in s],
    axis=1
)

st.dataframe(styled,use_container_width=True)
