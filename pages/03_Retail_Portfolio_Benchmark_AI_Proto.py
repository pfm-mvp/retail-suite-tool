# pages/03A_Region_Hourly_Conversion_Prototype.py
import os, sys
from datetime import datetime
import numpy as np
import pandas as pd
import pytz
import requests
import streamlit as st
import plotly.express as px

# --- project imports ---
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))
from helpers_shop import REGIONS, get_ids_by_region, ID_TO_NAME
from helpers_normalize import normalize_vemcount_response

st.set_page_config(page_title="Regio • Uurprofielen (prototype)", page_icon="⏱️", layout="wide")
st.title("⏱️ Regio • Conversie-uurprofielen (prototype)")

API_URL = st.secrets["API_URL"]
TZ = pytz.timezone("Europe/Amsterdam")

# ---------- UI ----------
PERIODS = ["this_week","last_week","this_month","last_month"]
c1, c2 = st.columns([1,1])
with c1:
    period = st.selectbox("Periode", PERIODS, index=0)
with c2:
    regio = st.selectbox("Regio", REGIONS, index=0)

shop_ids = get_ids_by_region(regio)
if not shop_ids:
    st.warning("Geen winkels in deze regio.")
    st.stop()
    
regio = st.selectbox("Regio", REGIONS, index=0)
shop_ids = get_ids_by_region(regio)
if not shop_ids:
    st.warning("Geen winkels in deze regio.")
    st.stop()

# Minimaal aantal bezoekers om een uur “betekenisvol” te noemen
traffic_threshold = st.slider("Traffic-drempel (min. bezoekers/uur)", 10, 100, 30, 5)

# ---------- API helpers ----------
METRICS = ["count_in","conversion_rate","sales_per_visitor"]
def post_report(params):
    r = requests.post(API_URL, params=params, timeout=40)
    r.raise_for_status()
    return r

def fetch_df_hourly(ids, period, metrics):
    params = [("data", sid) for sid in ids]
    params += [("data_output", m) for m in metrics]
    params += [("source","shops"), ("period", period), ("step","hour")]
    r = post_report(params)
    js = r.json()
    df = normalize_vemcount_response(js, {sid: ID_TO_NAME[sid] for sid in ids}, kpi_keys=metrics)
    # timestamp → hour
    if "timestamp" in df.columns:
        df["hour"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.hour
    else:
        df["hour"] = np.nan
    return df, params, r.status_code

# ---------- Data ophalen ----------
df_h, req_params, status = fetch_df_hourly(shop_ids, period, METRICS)
if df_h.empty or df_h["hour"].isna().all():
    st.warning("Geen hourly data beschikbaar voor deze reeks.")
    st.stop()

# Uitsluiten van toekomstige uren (alleen tot 'nu' Amsterdam-tijd)
now_local = datetime.now(TZ).hour
if period.startswith("this_"):
    df_h = df_h[df_h["hour"] <= now_local]

# ---------- Analyse per winkel ----------
# Conversie kan als %, zorg dat het float is
df_h["conversion_rate"] = pd.to_numeric(df_h["conversion_rate"], errors="coerce")

# Gemiddelde per winkel (dag/periodeniveau) en per uur
avg_by_shop = df_h.groupby("shop_id", as_index=False).agg(
    conv_mean=("conversion_rate","mean")
)
hourly_by_shop = df_h.groupby(["shop_id","shop_name","hour"], as_index=False).agg(
    conv=("conversion_rate","mean"),
    traffic=("count_in","sum")
)
# Merge winkelgemiddelde erbij
hourly_by_shop = hourly_by_shop.merge(avg_by_shop, on="shop_id", how="left")

# “Zwakke” uren = genoeg traffic + conv < eigen daggemiddelde
weak = hourly_by_shop[(hourly_by_shop["traffic"] >= traffic_threshold) &
                      (hourly_by_shop["conv"] < hourly_by_shop["conv_mean"])].copy()
weak["gap_pp"] = (hourly_by_shop["conv_mean"] - hourly_by_shop["conv"]).round(2)

# ---------- Heatmap (uren × winkels) ----------
st.subheader("🔥 Heatmap — conversie per uur (PFM-kleuren)")
hm = hourly_by_shop.pivot(index="shop_name", columns="hour", values="conv").sort_index()
pfm_colorscale = ["#762181", "#D8456C", "#FEAC76"]  # PFM heatmap-schaal
fig = px.imshow(
    hm,
    color_continuous_scale=pfm_colorscale,
    labels=dict(color="Conversie (%)"),
    aspect="auto",
)
fig.update_layout(
    height=520,
    margin=dict(l=0, r=0, t=10, b=10),
    coloraxis_colorbar=dict(title="Conversie (%)"),
)
st.plotly_chart(fig, use_container_width=True)

# ---------- AI-aanbevelingen ----------
st.subheader("🤖 AI — Zwakke uren per winkel")
if weak.empty:
    st.success("Geen duidelijke zwakke uren gevonden in deze regio. Nice!")
else:
    # Top 3 zwakste blokken per winkel (grootste daling in pp)
    out = (weak.sort_values(["gap_pp"], ascending=False)
               .groupby("shop_name").head(3)
               .sort_values(["shop_name","hour"]))
    # Rapporteer per winkel
    for shop, grp in out.groupby("shop_name"):
        bullets = []
        for _, r in grp.iterrows():
            bullets.append(f"{int(r['hour']):02d}:00 — {r['conv']:.2f}% (−{r['gap_pp']:.2f}pp t.o.v. winkelgem.)")
        st.markdown(f"""
        <div style="border:1px dashed #ddd; border-radius:10px; padding:10px; margin-bottom:6px;">
          <b>{shop}</b><br/>
          {'<br/>'.join(bullets)}
          <div style="color:#6B7280; font-size:13px; margin-top:6px;">
            Advies: verhoog vloerbezetting of activeer kassascripts/promoties in deze uren; 
            test bundels voor SPV in de rustige blokken.
          </div>
        </div>
        """, unsafe_allow_html=True)

# ---------- Debug ----------
with st.expander("🔧 Debug"):
    st.write("Params:", req_params, "status:", status)
    st.dataframe(df_h.head())