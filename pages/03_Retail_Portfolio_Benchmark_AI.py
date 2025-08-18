# pages/03_Retail_Portfolio_Benchmark_AI.py
import os, sys
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
import streamlit as st

# ── Project imports ─────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))
from helpers_shop import ID_TO_NAME, REGIONS, get_ids_by_region
from helpers_normalize import normalize_vemcount_response

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Retail Portfolio Benchmark", page_icon="📊", layout="wide")
st.title("📊 Retail Portfolio Benchmark")

API_URL = st.secrets["API_URL"]
TZ = pytz.timezone("Europe/Amsterdam")

# ── UI: periode & regio (met ALL) ──────────────────────────────────────────────
PERIODS = ["this_week","last_week","this_month","last_month","this_quarter","last_quarter","this_year","last_year"]
colP, colR = st.columns([1,1])
with colP:
    period = st.selectbox("Periode", PERIODS, index=3, key="rb_period")
with colR:
    regio = st.selectbox("Regio", ["ALL"] + REGIONS, index=0, key="rb_region")

# ── Shop selection obv regio ───────────────────────────────────────────────────
if regio == "ALL":
    shop_ids = list(ID_TO_NAME.keys())
else:
    shop_ids = get_ids_by_region(regio) or list(ID_TO_NAME.keys())

if not shop_ids:
    st.warning("Geen winkels gevonden voor deze selectie.")
    st.stop()

# ── Helpers ────────────────────────────────────────────────────────────────────
KPI_KEYS = ["count_in","conversion_rate","turnover","sales_per_visitor","sq_meter"]
TODAY = datetime.now(TZ).date()

def post_report(params):
    import requests
    r = requests.post(API_URL, params=params, timeout=45)
    r.raise_for_status()
    return r

def add_effective_date(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "date" not in d.columns:
        d["date"] = pd.NaT
    ts = pd.to_datetime(d.get("timestamp"), errors="coerce")
    d["date_eff"] = pd.to_datetime(d["date"], errors="coerce").fillna(ts)
    return d

def fetch_df(ids, period, step="day"):
    params = [("data", sid) for sid in ids]
    params += [("data_output", m) for m in KPI_KEYS]
    params += [("source","shops"), ("period", period), ("step", step)]
    resp = post_report(params)
    js = resp.json()
    df = normalize_vemcount_response(js, ID_TO_NAME, kpi_keys=KPI_KEYS)
    df = add_effective_date(df)
    # tot en met gisteren voor lopende periodes
    if str(period).startswith("this_"):
        df = df[df["date_eff"].dt.date < TODAY]
    return df, params, resp.status_code

def weighted_avg(series, weights):
    try:
        w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
        s = pd.to_numeric(series, errors="coerce").fillna(0.0)
        d = w.sum()
        return (s*w).sum()/d if d else np.nan
    except Exception:
        return np.nan

def agg_stores(d: pd.DataFrame) -> pd.DataFrame:
    if d is None or d.empty:
        return pd.DataFrame()
    # sommen
    g = d.groupby("shop_id", as_index=False).agg({"count_in":"sum","turnover":"sum"})
    # gewogen gemiddelden
    w = d.groupby("shop_id").apply(
        lambda x: pd.Series({
            "conversion_rate": weighted_avg(x["conversion_rate"], x["count_in"]),
            "sales_per_visitor": weighted_avg(x["sales_per_visitor"], x["count_in"])
        })
    ).reset_index()
    g = g.merge(w, on="shop_id", how="left")
    # laatste bekende m²
    sqm = (d.sort_values("date_eff").groupby("shop_id")["sq_meter"]
           .apply(lambda s: float(s.dropna().iloc[-1]) if s.dropna().size else np.nan)).reset_index()
    g = g.merge(sqm, on="shop_id", how="left")
    # sales per m²
    g["sales_per_sqm"] = g.apply(
        lambda r: (r["turnover"]/r["sq_meter"]) if (pd.notna(r["sq_meter"]) and r["sq_meter"]>0) else np.nan,
        axis=1
    )
    g["shop_name"] = g["shop_id"].map(ID_TO_NAME)
    return g

def eur0(x): 
    try: return f"€{float(x):,.0f}".replace(",", ".")
    except: return "€0"
def eur2(x):
    try: return f"€{float(x):,.2f}".replace(",", ".")
    except: return "€0,00"
def pct2(x):
    try: return f"{float(x):.2f}%"
    except: return "0,00%"

# ── Fetch & aggregate ──────────────────────────────────────────────────────────
df, params, status = fetch_df(shop_ids, period, "day")

with st.expander("🔧 Debug — fetch"):
    st.write("HTTP:", status)
    st.write("Params (head):", params[:10])
    st.write("Rows:", 0 if df is None else len(df))
    st.dataframe(df.head(10) if df is not None else pd.DataFrame())

if df is None or df.empty:
    st.info("Geen data voor deze selectie.")
    st.stop()

cur = agg_stores(df)
if cur.empty:
    st.info("Geen geaggregeerde data voor deze selectie.")
    st.stop()

# ── Regio-KPIs ─────────────────────────────────────────────────────────────────
total_turn = cur["turnover"].sum()
total_vis  = cur["count_in"].sum()
total_sqm  = cur["sq_meter"].fillna(0).sum()
avg_conv   = weighted_avg(cur["conversion_rate"], cur["count_in"])
avg_spv    = weighted_avg(cur["sales_per_visitor"], cur["count_in"])
avg_spsqm  = (total_turn/total_sqm) if total_sqm>0 else np.nan

c1,c2,c3,c4 = st.columns(4)
with c1:
    st.metric("💶 Totale omzet", eur0(total_turn))
with c2:
    st.metric("🛒 Gem. conversie", pct2(avg_conv))
with c3:
    st.metric("💸 Gem. SPV", eur2(avg_spv))
with c4:
    st.metric("🏁 Gem. sales/m²", eur2(avg_spsqm))

st.markdown("---")

# ── Tabel: benchmark per winkel ────────────────────────────────────────────────
show = cur[["shop_name","count_in","conversion_rate","sales_per_visitor","turnover","sales_per_sqm"]].copy()
show = show.rename(columns={
    "shop_name":"Winkel",
    "count_in":"Bezoekers",
    "conversion_rate":"Conversie",
    "sales_per_visitor":"SPV",
    "turnover":"Omzet",
    "sales_per_sqm":"Sales/m²",
})
show["Bezoekers"] = show["Bezoekers"].map(lambda x: f"{int(x):,}".replace(",", ".")) 
show["Conversie"] = show["Conversie"].map(pct2)
show["SPV"]       = show["SPV"].map(eur2)
show["Omzet"]     = show["Omzet"].map(eur0)
show["Sales/m²"]  = show["Sales/m²"].map(eur2)

st.subheader("🏁 Portfolio benchmark (per winkel)")
st.dataframe(show.sort_values("Sales/m²", ascending=False), use_container_width=True)
