import streamlit as st
import pandas as pd
import numpy as np
import requests
import holidays
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys
import pytz
sys.path.append(str(Path(__file__).parent.parent))
from shop_mapping import SHOP_NAME_MAP
from helpers_shop import ID_TO_NAME, get_ids_by_region
from helpers_normalize import normalize_vemcount_response

st.set_page_config(page_title="AI Retail Advisor", layout="wide", page_icon="Bag")
st.title("AI Retail Advisor: Regio- & Winkelvoorspellingen")

# ─── SECRETS ───
API_URL = st.secrets["API_URL"]
OW_KEY  = st.secrets["openweather_api_key"]
CBS_ID  = st.secrets["cbs_dataset"]
CBS_URL = f"https://opendata.cbs.nl/ODataFeed/odata/{CBS_ID}/Consumentenvertrouwen"

# ─── SIDEBAR ───
PERIODS = ["this_week","last_week","this_month","last_month","this_quarter","last_quarter","this_year","last_year"]
regio = st.sidebar.selectbox("Regio", ["All"] + list(set(i["region"] for i in SHOP_NAME_MAP.values())), index=0)
period = st.sidebar.selectbox("Periode", PERIODS, index=3)  # last_month

# Shop IDs
if regio == "All":
    shop_ids = list(SHOP_NAME_MAP.keys())
else:
    shop_ids = get_ids_by_region(regio)
if not shop_ids:
    st.stop()

# ─── HELPER FUNCTIES ───
TZ = pytz.timezone("Europe/Amsterdam")
TODAY = datetime.now(TZ).date()
METRICS = ["count_in","conversion_rate","turnover","sales_per_visitor","sq_meter"]

def step_for(p: str) -> str:
    return "day" if p.endswith("week") or p.endswith("month") else "month"

def add_effective_date(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    ts = pd.to_datetime(d.get("timestamp"), errors="coerce")
    d["date_eff"] = pd.to_datetime(d["date"], errors="coerce").fillna(ts)
    d["date_eff"] = d["date_eff"].dt.date
    d["year"] = pd.to_datetime(d["date_eff"]).dt.year
    d["month"] = pd.to_datetime(d["date_eff"]).dt.month
    d["week"] = pd.to_datetime(d["date_eff"]).dt.isocalendar().week
    d["shop_id"] = d["shop_id"].astype(int)  # FIX KeyError
    return d

def fetch(shop_ids, period: str) -> pd.DataFrame:
    params = {
        "source": "shops",
        "period": period,
        "step": step_for(period)
    }
    # Query params zoals in werkend script
    for sid in shop_ids:
        params[f"data[]"] = str(sid)
    for m in METRICS:
        params[f"data_output[]"] = m

    try:
        r = requests.post(API_URL, params=params, timeout=45)
        r.raise_for_status()
        js = r.json()
        st.success(f"Planet PFM online – {len(js)} records")
        df = normalize_vemcount_response(js, ID_TO_NAME, METRICS)
        df = add_effective_date(df)
        if period.startswith("this_"):
            df = df[df["date_eff"] < TODAY]
        df = df.sort_values(["shop_id", "date_eff"])
        df["sq_meter"] = df.groupby("shop_id")["sq_meter"].ffill().bfill()
        df["conversion_rate"] = df["conversion_rate"] * 100  # To %
        return df
    except Exception as e:
        st.error(f"Planet PFM: {str(e)[:100]}")
        return pd.DataFrame()

df = fetch(shop_ids, period)

# ─── WEER ───
@st.cache_data(ttl=1800)
def weer(pc):
    try:
        r = requests.get("https://api.openweathermap.org/data/2.5/forecast", params={"q": f"{pc},NL", "appid": OW_KEY, "units":"metric"})
        r.raise_for_status()
        js = r.json()["list"]
        daily = pd.DataFrame([{"date": pd.to_datetime(d["dt_txt"]).date(), "temp": d["main"]["temp"], "rain": d.get("rain", {}).get("3h", 0)} for d in js])
        return daily.groupby("date").mean(numeric_only=True).reset_index().head(28)
    except:
        return None

# ─── CBS ───
@st.cache_data(ttl=86400)
def cbs():
    try:
        raw = requests.get(CBS_URL).json()["value"]
        c = pd.DataFrame(raw)[["Perioden","Consumentenvertrouwen_1","Koopbereidheid_5"]]
        c["maand"] = pd.to_datetime(c["Perioden"].str[:7] + "-01")
        return c.rename(columns={"Consumentenvertrouwen_1":"CBS_vertrouwen", "Koopbereidheid_5":"CBS_koop"})[["maand","CBS_vertrouwen","CBS_koop"]]
    except:
        return pd.DataFrame({"maand": pd.date_range("2025-01", "2025-11", freq="MS"), "CBS_vertrouwen": [-8,-7,-9,-6,-10,-11,-9,-12,-10,-13,-14], "CBS_koop": [-15,-14,-16,-13,-17,-18,-16,-19,-17,-20,-21]})

cbs_df = cbs()

# ─── VORIG JAAR (fallback als niet beschikbaar) ───
df_last_year = fetch(shop_ids, "last_year")
if df_last_year.empty:
    df_last_year = df.copy()  # Gebruik huidige als fallback

# ─── KPI’s ───
if not df.empty:
    total_foot = df["count_in"].sum()
    total_omzet = df["turnover"].sum()
    avg_conv = df["conversion_rate"].mean()
    avg_spv = df["sales_per_visitor"].mean()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Footfall", f"{int(total_foot):,}".replace(",","."))
    c2.metric("Omzet", f"€{int(total_omzet):,}".replace(",","."))
    c3.metric("Conversie", f"{avg_conv:.1f}%")
    c4.metric("SPV", f"€{avg_spv:.0f}")

# ─── GRAFIEK YTD ───
tab1,tab2,tab3 = st.tabs(["YTD
