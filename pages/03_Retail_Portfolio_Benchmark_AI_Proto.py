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

# ───────────────────────── KPI-keuze + openingstijden + heatmap ─────────────────────────
st.subheader("📊 Uurprofiel — heatmap")

# 1) KPI-keuze
KPI_CHOICES = {
    "Sales per m²": "sales_per_sqm",           # wordt zo nodig berekend
    "Conversie": "conversion_rate",            # gewogen op traffic
    "SPV (sales/visitor)": "sales_per_visitor" # wordt zo nodig berekend
}
kpi_label = st.selectbox("Kies KPI", list(KPI_CHOICES.keys()), index=0, key="kpi_heat")
kpi_key   = KPI_CHOICES[kpi_label]

# 2) Openingstijden-slider (inclusief start, exclusief einduur)
open_start, open_end = st.slider(
    "Openingstijden (uur, lokaal)",
    min_value=0, max_value=24, value=(10, 18), step=1, key="open_hours_slider"
)

# 3) Zorg voor een robuuste hour-kolom in lokale tijd
df_hsrc = df.copy()
if "hour" not in df_hsrc.columns:
    # timestamp -> Europe/Amsterdam uur
    ts = pd.to_datetime(df_hsrc.get("timestamp"), errors="coerce", utc=True)
    # ts is nu UTC tz-aware; converteer naar Amsterdam en haal uur
    df_hsrc["hour"] = ts.dt.tz_convert("Europe/Amsterdam").dt.hour
else:
    # Forceer int en range-beperking 0..23
    df_hsrc["hour"] = pd.to_numeric(df_hsrc["hour"], errors="coerce").fillna(-1).astype(int)
    df_hsrc = df_hsrc[(df_hsrc["hour"] >= 0) & (df_hsrc["hour"] <= 23)]

# 4) Hulpfunctie: filter urenvenster (ook als het over middernacht loopt)
def in_window(h: pd.Series, start: int, end: int) -> pd.Series:
    # inclusieve start, exclusieve eind (zoals openingstijden meestal)
    if start == end:
        # 24u open -> geen filter
        return pd.Series([True] * len(h), index=h.index)
    if start < end:
        return (h >= start) & (h < end)
    else:
        # over middernacht, b.v. 20 → 3
        return (h >= start) | (h < end)

mask = in_window(df_hsrc["hour"], open_start, open_end)
df_hsrc = df_hsrc[mask]

# 5) KPI-afleidingen (indien nodig)
#    - SPV = turnover / count_in
#    - Sales per m² = turnover / sq_meter (laatste bekende/typisch constante waarde per shop)
if "sales_per_visitor" not in df_hsrc.columns or df_hsrc["sales_per_visitor"].isna().all():
    df_hsrc["sales_per_visitor"] = np.where(
        df_hsrc["count_in"] > 0, df_hsrc["turnover"] / df_hsrc["count_in"], np.nan
    )

def last_non_null(s):
    s = s.dropna()
    return s.iloc[-1] if len(s) else np.nan

if "sales_per_sqm" not in df_hsrc.columns or df_hsrc["sales_per_sqm"].isna().all():
    # neem per rij een benadering, maar we aggregeren per shop+uur straks toch opnieuw
    # en gebruiken dan de som(omzet) / (laatste niet-nul m² per shop)
    pass  # we rekenen dit pas in de aggregator hieronder

# 6) Aggregatie per shop+uur volgens KPI
def agg_for_kpi(block: pd.DataFrame, kpi: str) -> float:
    if kpi == "conversion_rate":
        w = block["count_in"].fillna(0)
        conv = block["conversion_rate"].fillna(0)
        denom = w.sum()
        return float((conv * w).sum() / denom) if denom > 0 else np.nan

    if kpi == "sales_per_visitor":
        # som omzet / som bezoekers
        turn = block["turnover"].sum(skipna=True)
        vis  = block["count_in"].sum(skipna=True)
        return float(turn / vis) if vis > 0 else np.nan

    if kpi == "sales_per_sqm":
        # som omzet / (laatste bekende m² voor die shop in de block)
        turn = block["turnover"].sum(skipna=True)
        sqm  = last_non_null(block.get("sq_meter", pd.Series(dtype=float)))
        # val terug op gemiddelde m² als laatste niet beschikbaar is
        if (pd.isna(sqm) or sqm == 0) and "sq_meter" in block.columns:
            sqm = block["sq_meter"].replace(0, np.nan).mean()
        return float(turn / sqm) if (pd.notna(sqm) and sqm > 0) else np.nan

    # fallback
    return float(block[kpi].mean(skipna=True))

agg = (
    df_hsrc
    .groupby(["shop_id", "shop_name", "hour"], as_index=False)
    .apply(lambda g: pd.Series({ "value": agg_for_kpi(g, kpi_key) }))
)

# 7) Kolomvolgorde alleen gekozen uren
if open_start == open_end:
    # 24u open (geen beperking)
    hours_order = list(range(0, 24))
elif open_start < open_end:
    hours_order = list(range(open_start, open_end))
else:
    hours_order = list(range(open_start, 24)) + list(range(0, open_end))

# 8) Pivot -> heatmapmatrix (shop x uur)
matrix = (
    agg.pivot(index="shop_name", columns="hour", values="value")
    .reindex(columns=hours_order)
    .sort_index()
)

# 9) Titel/eenheid
unit = {
    "conversion_rate": "%",
    "sales_per_visitor": "€",
    "sales_per_sqm": "€"
}.get(kpi_key, "")

title = f"{kpi_label} per uur (alleen {open_start}:00–{open_end}:00)"

# 10) Plotly express heatmap
import plotly.express as px
fig = px.imshow(
    matrix,
    labels=dict(x="hour", y="shop_name", color=unit),
    aspect="auto",
    color_continuous_scale="Magma"  # kies je eigen palette
)
fig.update_layout(title=title, height=520)
st.plotly_chart(fig, use_container_width=True)

# Kleine toelichting
st.caption(
    "De heatmap toont alleen de door jou gekozen openingstijden. "
    "Voor **Conversie** is gewogen gemiddeld per uur (gewogen op traffic)."
)

# ── Debug ──────────────────────────────────────────────────────────────────────
with st.expander("🔧 Debug"):
    st.write("period:", period, "step:", step_for(period))
    st.write("shops selected:", shop_ids[:10], "… (n=", len(shop_ids), ")")
    st.write("df head:", df.head())
    st.write("heat head:", heat.head())
