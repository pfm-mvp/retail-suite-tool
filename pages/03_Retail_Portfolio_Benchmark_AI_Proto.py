# pages/03_Retail_Portfolio_Benchmark_AI_Proto.py
import os, sys
from datetime import datetime
import pytz
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------- Imports / mapping ----------
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))
from helpers_shop import ID_TO_NAME, NAME_TO_ID, REGIONS, get_ids_by_region
from helpers_normalize import normalize_vemcount_response

# ---------- Page ----------
st.set_page_config(page_title="Portfolio Benchmark (AI) — Proto", page_icon="🧪", layout="wide")
st.title("🧪 Portfolio Benchmark (AI) — Proto")

API_URL = st.secrets["API_URL"]

# ---------- PFM-styling / colors ----------
PFM_COLORSCALE = ["#21114E", "#5B167E", "#922B80", "#CC3F71", "#F56B5C", "#FEAC76"]  # brand
PFM_GRAY = "#6B7280"

st.markdown("""
<style>
.kpi { border:1px solid #eee; border-radius:14px; padding:16px; }
.kpi .t { font-weight:600; color:#0C111D; }
.kpi .v { font-size:38px; font-weight:800; }
.tip { color:#6B7280; font-size:13px; margin-top:6px; }
</style>
""", unsafe_allow_html=True)

# ---------- Inputs ----------
TZ     = pytz.timezone("Europe/Amsterdam")
TODAY  = datetime.now(TZ).date()

PERIODS = ["this_week","last_week","this_month","last_month","this_quarter","last_quarter","this_year","last_year"]

col1, col2, col3 = st.columns([1,1,1.2])
with col1:
    period = st.selectbox("Periode", PERIODS, index=1, key="proto_period")
with col2:
    regio = st.selectbox("Regio", ["All"] + REGIONS, index=0, key="proto_region")
with col3:
    # Openingstijden -> mask voor heatmap (alleen deze uren zichtbaar)
    open_from, open_to = st.slider("Openingstijden (in uren, 24h)", 0, 23, (10, 18), 1, key="proto_opening_hours")
    max_shops = st.slider("Max winkels in vergelijking", 5, 30, 30, 1, key="proto_max_shops")

# Bepaal shop_ids o.b.v. regio
if regio == "All":
    SHOP_IDS = list(ID_TO_NAME.keys())
else:
    SHOP_IDS = get_ids_by_region(regio) or list(ID_TO_NAME.keys())

if not SHOP_IDS:
    st.warning("Geen winkels gevonden (mapping/regio leeg).")
    st.stop()

# ---------- Helpers ----------
def step_for(p: str) -> str:
    # uurdata voor week/maand; dag/maand voor langere perioden
    if p.endswith("week") or p.endswith("month"):
        return "hour"
    return "day"

def post(params):
    r = requests.post(API_URL, params=params, timeout=45)
    r.raise_for_status()
    return r

def add_effective_date(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "date" not in d.columns:
        d["date"] = pd.NaT
    ts = pd.to_datetime(d.get("timestamp"), errors="coerce")
    d["date_eff"] = pd.to_datetime(d["date"], errors="coerce").fillna(ts)
    # bewaar datetime voor uur-extractie; voor filters op “t/m gisteren” gebruiken we .dt.date
    return d

def fetch(shop_ids, period, metrics):
    params = [("data", sid) for sid in shop_ids]
    params += [("data_output", m) for m in metrics]
    params += [("source","shops"), ("period", period), ("step", step_for(period))]
    r = post(params)
    js = r.json()
    df = normalize_vemcount_response(js, ID_TO_NAME, kpi_keys=metrics)
    df = add_effective_date(df)

    # Alleen t/m gisteren als het een “this_*” periode is
    ddate = ddate = pd.to_datetime(df["date_eff"], errors="coerce")
    df = df.assign(date_eff=ddate, date_only=ddate.dt.date)
    if period.startswith("this_"):
        df = df[df["date_only"] < TODAY]

    # hour kolom (0–23) voor heatmap
    df["hour"] = pd.to_datetime(df["date_eff"], errors="coerce").dt.hour

    # sq_meter netjes vullen per winkel
    df = df.sort_values(["shop_id","date_eff"])
    df["sq_meter"] = df.groupby("shop_id")["sq_meter"].ffill().bfill()
    return df

def fmt_eur2(x): return f"€{x:,.2f}".replace(",", ".")

# ---------- Data ophalen ----------
METRICS = ["count_in","conversion_rate","turnover","sales_per_visitor","sq_meter"]
df = fetch(SHOP_IDS, period, METRICS)

if df.empty:
    st.info("Geen data beschikbaar voor deze periode/regio.")
    st.stop()

# ---------- Gemiddelde sales/m² per uur (per winkel) ----------
# Som omzet per uur en deel door laatste/laatst bekende m² per winkel in periode
# (alternatief zou omzet/m² per record zijn en dan middelen op traffic, maar voor overzicht is dit prima)
agg = (
    df.groupby(["shop_id","hour"], as_index=False)
      .agg(turnover=("turnover","sum"), sq_meter=("sq_meter","last"))
)
agg["sales_per_sqm"] = np.where(agg["sq_meter"]>0, agg["turnover"]/agg["sq_meter"], np.nan)
agg["shop_name"] = agg["shop_id"].map(ID_TO_NAME)

# ---------- Openingstijden-masker toepassen ----------
# Behoud alleen uren binnen [open_from, open_to). Buiten dit venster -> NaN
def mask_closed_hours(d: pd.DataFrame, h_from: int, h_to: int) -> pd.DataFrame:
    x = d.copy()
    ok = (x["hour"]>=h_from) & (x["hour"]<h_to)
    x.loc[~ok, "sales_per_sqm"] = np.nan
    return x

agg = mask_closed_hours(agg, open_from, open_to)

# ---------- Selectie winkels (max_shops) ----------
shops_sorted = (
    agg.groupby(["shop_id","shop_name"], as_index=False)
       .agg(total_turn=("turnover","sum"))
       .sort_values("shop_name")
)
sel_shop_ids = shops_sorted["shop_id"].head(max_shops).tolist()
agg = agg[agg["shop_id"].isin(sel_shop_ids)]

# ---------- Heatmap bouwen ----------
# Zorg dat alle gekozen uren als kolommen aanwezig zijn
hours_range = list(range(open_from, open_to))  # inclusieve start, exclusieve end
hm = (
    agg.pivot(index="shop_name", columns="hour", values="sales_per_sqm")
       .reindex(columns=hours_range)           # toon alleen openingstijden
       .sort_index()
)

if hm.isna().all(None) or hm.empty:
    st.info("Geen uurdata binnen de ingestelde openingstijden.")
else:
    fig = px.imshow(
        hm,
        color_continuous_scale=PFM_COLORSCALE,
        labels=dict(color="€ / m²"),
        aspect="auto",
        origin="upper",
    )
    fig.update_layout(
        height=560,
        margin=dict(l=0, r=0, t=10, b=10),
        coloraxis_colorbar=dict(title="€ / m²"),
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------- Kleine toelichting onder de heatmap ----------
st.markdown(
    f"""
<div class="tip">
ℹ️ Heatmap toont <b>gemiddelde sales per m² per uur</b> in “{period.replace('_',' ')}”
voor regio <b>{regio}</b> ({len(sel_shop_ids)} winkels, max {max_shops}).
Uren buiten de ingestelde openingstijden ({open_from}:00–{open_to}:00) worden verborgen.
</div>
""",
    unsafe_allow_html=True,
)
