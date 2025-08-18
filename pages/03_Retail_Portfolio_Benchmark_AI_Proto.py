# pages/03_Retail_Portfolio_Benchmark_AI_Proto.py
import os, sys, calendar
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ── Imports / mapping ───────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))
from helpers_shop import ID_TO_NAME, NAME_TO_ID, REGIONS, get_ids_by_region
from helpers_normalize import normalize_vemcount_response

# ── Page ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Portfolio Benchmark (Proto)", page_icon="🧪", layout="wide")
st.title("🧪 Portfolio Benchmark (Proto) — Uur-heatmap")

API_URL = st.secrets["API_URL"]

# PFM kleuren
PFM_RED   = "#F04438"
PFM_GREEN = "#22C55E"
PFM_GRAY  = "#6B7280"

# ── Inputs ─────────────────────────────────────────────────────────────────────
PERIODS = ["this_week","last_week","this_month","last_month","this_quarter","last_quarter","this_year","last_year"]

cA, cB, cC = st.columns([1.1,1.2,1])
with cA:
    period = st.selectbox("Periode", PERIODS, index=2, key="proto_period")
with cB:
    regio = st.selectbox("Regio", ["All"] + REGIONS, index=0, key="proto_region")
with cC:
    kpi_choice = st.selectbox("KPI voor heatmap", ["Sales/m²","Conversie","SPV"], index=0, key="proto_kpi")

# Openingstijden (mask buiten venster)
cH1, cH2 = st.columns(2)
with cH1:
    open_from = st.slider("Open vanaf (uur)", 0, 23, 10, key="proto_open_from")
with cH2:
    open_to   = st.slider("Open tot en met (uur)", 0, 23, 18, key="proto_open_to")

# Shop selectie obv regio
if regio == "All":
    region_ids = list(ID_TO_NAME.keys())
else:
    region_ids = get_ids_by_region(regio) or []

if not region_ids:
    st.warning("Geen winkels gevonden voor deze selectie.")
    st.stop()

# optionele multiselect (max 30)
default_names = [ID_TO_NAME[sid] for sid in region_ids][:12]
sel_names = st.multiselect("Vergelijk winkels (max 30)",
                           [ID_TO_NAME[sid] for sid in region_ids],
                           default=default_names,
                           max_selections=30,
                           key="proto_shop_multi")

shop_ids = [NAME_TO_ID[n] for n in sel_names] if sel_names else region_ids[:30]
if not shop_ids:
    st.info("Geen winkels geselecteerd.")
    st.stop()

# ── Helpers ────────────────────────────────────────────────────────────────────
TZ = pytz.timezone("Europe/Amsterdam")
TODAY = datetime.now(TZ).date()

METRICS = ["count_in","conversion_rate","turnover","sales_per_visitor","sq_meter"]

def step_for(period: str) -> str:
    # Voor proto willen we uurdata als het even kan; anders fallback op 'day'
    return "hour" if any(period.endswith(x) for x in ["week","month"]) else "day"

def post(params):
    r = __import__("requests").post(API_URL, params=params, timeout=45)
    r.raise_for_status()
    return r

def add_effective_date(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "date" not in d.columns:
        d["date"] = pd.NaT
    # timestamp -> datetime
    ts = pd.to_datetime(d.get("timestamp"), errors="coerce")
    # date_eff = date (indien beschikbaar) anders timestamp.date()
    d["date_eff"] = pd.to_datetime(d["date"], errors="coerce").fillna(ts).dt.date
    # Uurkolom (bij hour step verwachten we timestamps)
    d["hour"] = pd.to_datetime(ts, errors="coerce").dt.hour
    return d

def fetch(period: str, shop_ids: list[int]) -> pd.DataFrame:
    params = [("data", sid) for sid in shop_ids]
    params += [("data_output", m) for m in METRICS]
    params += [("source","shops"), ("period", period), ("step", step_for(period))]
    r  = post(params)
    js = r.json()
    df = normalize_vemcount_response(js, ID_TO_NAME, kpi_keys=METRICS)
    df = add_effective_date(df)

    # Exclude today for “this_*”
    if period.startswith("this_"):
        df = df[df["date_eff"] < TODAY]

    # Zorg dat sq_meter gevuld blijft per shop
    df = df.sort_values(["shop_id","date_eff","hour"])
    df["sq_meter"] = df.groupby("shop_id")["sq_meter"].ffill().bfill()
    return df

def weighted_avg(series, weights):
    w = weights.fillna(0.0); s = series.fillna(0.0)
    d = w.sum()
    return (s*w).sum()/d if d else np.nan

def fmt_eur2(x): return f"€{x:,.2f}".replace(",", ".")
def fmt_pct2(x): return f"{x:.2f}%"

# ── Data ophalen ────────────────────────────────────────────────────────────────
df = fetch(period, shop_ids)
if df.empty:
    st.warning("Geen data voor deze periode/regio/selectie.")
    st.stop()

# ── KPI Cards (op regioselectie) ───────────────────────────────────────────────
g = df.groupby("shop_id", as_index=False).agg({"count_in":"sum","turnover":"sum"})
w = df.groupby("shop_id").apply(lambda x: pd.Series({
    "conversion_rate": weighted_avg(x["conversion_rate"], x["count_in"]),
    "sales_per_visitor": weighted_avg(x["sales_per_visitor"], x["count_in"]),
})).reset_index()
g = g.merge(w, on="shop_id", how="left")
sqm = (df.groupby("shop_id")["sq_meter"]
         .apply(lambda s: float(s.dropna().iloc[-1]) if s.dropna().size else np.nan)
         .reset_index())
g = g.merge(sqm, on="shop_id", how="left")
g["sales_per_sqm"] = np.where(g["sq_meter"]>0, g["turnover"]/g["sq_meter"], np.nan)
g["shop_name"] = g["shop_id"].map(ID_TO_NAME)

total_turn = g["turnover"].sum()
avg_conv   = weighted_avg(g["conversion_rate"],   g["count_in"])
avg_spv    = weighted_avg(g["sales_per_visitor"], g["count_in"])
region_sqm = g["sq_meter"].fillna(0).sum()
avg_spsqm  = (total_turn/region_sqm) if region_sqm>0 else np.nan

c1,c2,c3,c4 = st.columns(4)
with c1:
    st.metric("💶 Totale omzet", f"{fmt_eur2(total_turn)}")
with c2:
    st.metric("🛒 Gem. conversie", fmt_pct2(avg_conv))
with c3:
    st.metric("💸 Gem. SPV", fmt_eur2(avg_spv))
with c4:
    st.metric("🏁 Gem. sales/m²", fmt_eur2(avg_spsqm))

st.markdown("---")

# ── Uur-Heatmap: gemiddelde per uur in de gekozen periode ──────────────────────
st.subheader("⏱️ Uur-heatmap per winkel (gemiddeld in periode)")

# KPI kolom bepalen
if kpi_choice == "Sales/m²":
    df["sales_per_sqm"] = np.where(df["sq_meter"]>0, df["turnover"]/df["sq_meter"], np.nan)
    value_col = "sales_per_sqm"
    colorbar_title = "€ / m²"
elif kpi_choice == "Conversie":
    value_col = "conversion_rate"
    colorbar_title = "%"
else:  # SPV
    value_col = "sales_per_visitor"
    colorbar_title = "€ / bezoeker"

# Alleen openingstijden tonen: mask buiten range naar NaN (blijft wit)
if "hour" in df.columns:
    df.loc[(df["hour"] < open_from) | (df["hour"] > open_to), value_col] = np.nan
else:
    df["hour"] = np.nan  # failsafe

# Gemiddelde per shop per uur
heat = (
    df.groupby(["shop_id","hour"], as_index=False)
      .agg(val=(value_col, "mean"))
)
heat["shop_name"] = heat["shop_id"].map(ID_TO_NAME)

# Zorg dat alle uren 0..23 aanwezig zijn (ook als NaN) zodat as compleet is
full_hours = pd.DataFrame({"hour": list(range(24))})
shops = heat["shop_id"].unique()
completed = []
for sid in shops:
    h_s = heat[heat["shop_id"]==sid][["hour","val"]].merge(full_hours, on="hour", how="right")
    h_s["shop_id"] = sid
    h_s["shop_name"] = ID_TO_NAME.get(sid, str(sid))
    completed.append(h_s)
heat = pd.concat(completed, ignore_index=True)

# Pivot: rijen = shop_name, kolommen = hour
hm = heat.pivot(index="shop_name", columns="hour", values="val").sort_index()

# Kleurschaal (PFM gradient)
pfm_colorscale = ["#21114E", "#5B167E", "#922B80", "#CC3F71", "#F56B5C", "#FEAC76"]

if hm.isna().all(None):
    st.info("Geen uurdata beschikbaar voor deze selectie.")
else:
    fig = px.imshow(
        hm,
        color_continuous_scale=pfm_colorscale,
        labels=dict(color=colorbar_title),
        aspect="auto",
    )
    fig.update_layout(
        height=560,
        margin=dict(l=0, r=0, t=10, b=10),
        coloraxis_colorbar=dict(title=colorbar_title),
    )
    st.plotly_chart(fig, use_container_width=True)

# Tooltip / uitleg
st.caption(
    "Tip: de heatmap toont het **gemiddelde per uur** in de gekozen periode. "
    "Witte vakken = buiten openingstijden of geen data. "
    f"KPI = **{kpi_choice}**. Pas de openingstijden-slider aan om ‘dode uren’ te maskeren."
)

# ── Debug ──────────────────────────────────────────────────────────────────────
with st.expander("🔧 Debug"):
    st.write("period:", period, "step:", step_for(period))
    st.write("shops selected:", shop_ids[:10], "… (n=", len(shop_ids), ")")
    st.write("df head:", df.head())
    st.write("heat head:", heat.head())