# pages/03_Retail_Portfolio_Benchmark_AI_Proto.py
import os, sys
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ───────────────────────── Imports / mapping ─────────────────────────
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))
from helpers_shop import ID_TO_NAME, NAME_TO_ID, REGIONS, get_ids_by_region
from helpers_normalize import normalize_vemcount_response

# ───────────────────────── Page & theming ────────────────────────────
st.set_page_config(page_title="Retail Portfolio Benchmark (Proto)", page_icon="🧪", layout="wide")

PFM_RED    = "#F04438"
PFM_GREEN  = "#22C55E"
PFM_PURPLE = "#6C4EE3"
PFM_GRAY   = "#6B7280"
PFM_GRAY_BG= "rgba(107,114,128,.08)"

st.markdown(f"""
<style>
.kpi-card {{
  border: 1px solid #EEE; border-radius: 14px; padding: 18px 18px 14px 18px;
  background: #FBFBFD;
}}
.kpi-title {{ color:#0C111D; font-weight:700; font-size:18px; margin-bottom:8px; }}
.kpi-value {{ font-size:44px; font-weight:900; line-height:1.05; }}
.kpi-delta {{
  font-size:13px; font-weight:700; padding:6px 10px; border-radius:999px; display:inline-block; 
}}
.kpi-delta.up {{ color:{PFM_GREEN}; background: rgba(34,197,94,.12); }}
.kpi-delta.down {{ color:{PFM_RED}; background: rgba(240,68,56,.12); }}
.kpi-delta.flat {{ color:{PFM_GRAY}; background: {PFM_GRAY_BG}; }}
</style>
""", unsafe_allow_html=True)

st.title("🧪 Retail Portfolio Benchmark — Proto")

API_URL = st.secrets["API_URL"]
TZ = pytz.timezone("Europe/Amsterdam")

# ───────────────────────── Controls ──────────────────────────────────
col1, col2, col3 = st.columns([1.2, 1, 1.2])
with col1:
    # Laat ook “All” kiezen
    regio = st.selectbox("Regio", ["All"] + REGIONS, index=0, key="proto_region")
with col2:
    period = st.selectbox(
        "Periode",
        ["this_week","last_week","this_month","last_month","this_quarter","last_quarter"],
        index=1, key="proto_period"
    )
with col3:
    # KPI-keuze voor heatmap
    kpi_label = st.selectbox("KPI voor heatmap",
                             ["Conversie (%)","SPV (€)","Sales/m² (€)"],
                             index=0, key="proto_kpi")

# Openingstijden-filter (heatmap)
colH1, colH2, colH3 = st.columns([1,1,1.6])
with colH1:
    start_hour = st.number_input("Open vanaf (uur)", min_value=0, max_value=23, value=10, step=1, key="proto_open_from")
with colH2:
    end_hour = st.number_input("Tot (uur, excl.)", min_value=1, max_value=24, value=18, step=1, key="proto_open_to")
with colH3:
    max_shops = st.slider("Max. winkels in heatmap", 5, 50, 30, 1, key="proto_max_shops")

# ───────────────────────── Helpers ───────────────────────────────────
def eur0(x): 
    try: return f"€{float(x):,.0f}".replace(",", ".")
    except: return "€0"

def pct2(x):
    try: return f"{float(x):.2f}%"
    except: return "0.00%"

def fetch_df(shop_ids, period, step, metrics):
    import requests
    params = [("data", sid) for sid in shop_ids]
    params += [("data_output", m) for m in metrics]
    params += [("source","shops"), ("period", period), ("step", step)]
    r = requests.post(API_URL, params=params, timeout=45)
    r.raise_for_status()
    js = r.json()
    df = normalize_vemcount_response(js, ID_TO_NAME, kpi_keys=metrics)
    # Zorg voor date, hour kolommen
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize("UTC").dt.tz_convert(TZ)
    else:
        ts = pd.to_datetime(df.get("date"), errors="coerce").dt.tz_localize(TZ, nonexistent='NaT', ambiguous='NaT')
    df["date_eff"] = ts.dt.date
    df["hour"] = ts.dt.hour
    return df, params, r.status_code

# KPI key mapping voor JSON → DataFrame
METRICS = ["count_in","conversion_rate","turnover","sales_per_visitor","sq_meter"]

if regio == "All":
    ALL_IDS = list(ID_TO_NAME.keys())
else:
    region_ids = get_ids_by_region(regio) or []
    ALL_IDS = region_ids

if not ALL_IDS:
    st.warning("Geen winkels gevonden voor deze regio/mapping.")
    st.stop()

# ───────────────────────── Data ophalen ──────────────────────────────
df, p_cur, s_cur = fetch_df(ALL_IDS, period, "hour", METRICS)

if df is None or df.empty:
    st.info("Geen data voor de gekozen periode/regio.")
    st.stop()

# Filter op openingstijden voor heatmap
df_h = df[(df["hour"] >= int(start_hour)) & (df["hour"] < int(end_hour))].copy()

# KPI kolommen consistent maken
# SPV mag 0 zijn; als het niet in data zit, fallback via turnover/count_in
if "sales_per_visitor" not in df_h.columns:
    df_h["sales_per_visitor"] = np.where(df_h["count_in"]>0, df_h["turnover"]/df_h["count_in"], 0.0)

# Sales per m² berekenen (als data aanwezig)
if "sq_meter" in df_h.columns:
    df_h["sales_per_sqm"] = np.where(df_h["sq_meter"]>0, df_h["turnover"]/df_h["sq_meter"], np.nan)
else:
    df_h["sales_per_sqm"] = np.nan

# ───────────────────────── KPI mapping & heatmap metric ──────────────
kpi_map = {
    "Conversie (%)": ("conversion_rate", "mean", "%"),
    "SPV (€)": ("sales_per_visitor", "mean", "€"),
    "Sales/m² (€)": ("sales_per_sqm", "mean", "€"),
}
kpi_key, agg_fun, unit = kpi_map[kpi_label]

# pivot: gemiddelde per uur per winkel, met urenfilter toegepast
pivot = (df_h.groupby(["shop_id","shop_name","hour"], as_index=False)
         .agg({kpi_key: agg_fun})
         .rename(columns={kpi_key: "value"}))

# kies top winkels qua totaal omzet (om interessante te tonen) en limiet
totals_by_shop = df_h.groupby(["shop_id","shop_name"], as_index=False)["turnover"].sum().sort_values("turnover", ascending=False)
keep_ids = totals_by_shop["shop_id"].head(int(max_shops)).tolist()
pivot = pivot[pivot["shop_id"].isin(keep_ids)].copy()

# Heatmap fig
if pivot.empty:
    fig = px.imshow(np.zeros((1,1)), color_continuous_scale="magma")
    fig.update_layout(height=400, margin=dict(l=10,r=10,t=10,b=10))
else:
    # Zorg dat uren-as alleen het gekozen uur-bereik toont
    hrs = list(range(int(start_hour), int(end_hour)))
    # Completed grid (optioneel, zodat missende uur/winkel cellen echt wit zijn)
    all_idx = pd.MultiIndex.from_product([pivot["shop_name"].unique(), hrs], names=["shop_name","hour"])
    heat = (pivot.set_index(["shop_name","hour"])["value"]
                 .reindex(all_idx)
                 .unstack("hour"))

    fig = px.imshow(
        heat,
        color_continuous_scale="magma",
        aspect="auto",
        labels=dict(x="hour", y="shop_name", color=unit),
        origin="lower"
    )
    fig.update_layout(height=520, margin=dict(l=10,r=10,t=10,b=10))

st.subheader("📊 Uren-heatmap")
st.caption(f"KPI: **{kpi_label}** • Uren: **{start_hour}–{end_hour}** • Max winkels: **{max_shops}**")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ───────────────────────── KPI Cards (totaal & delta) ───────────────
# Totaal berekeningen over gefilterde data
total_turn = float(df_h["turnover"].sum())
total_vis  = float(df_h["count_in"].sum())
# gewogen conversie en SPV over gefilterde set
w = df_h["count_in"].fillna(0.0)
avg_conv = (df_h["conversion_rate"].fillna(0.0)*w).sum()/w.sum() if w.sum()>0 else np.nan
avg_spv  = (df_h["sales_per_visitor"].fillna(0.0)*w).sum()/w.sum() if w.sum()>0 else np.nan
avg_spsqm= (df_h["turnover"].sum()/df_h["sq_meter"].fillna(0).sum()) if df_h["sq_meter"].fillna(0).sum()>0 else np.nan

# Delta vs vorige periode → alleen voor this_* (stabiele mapping exists)
this2last_map = {
    "this_week":"last_week",
    "this_month":"last_month",
    "this_quarter":"last_quarter",
}
if period in this2last_map:
    prev_df, _, _ = fetch_df(ALL_IDS, this2last_map[period], "hour", METRICS)
    prev_df = prev_df[(prev_df["hour"] >= int(start_hour)) & (prev_df["hour"] < int(end_hour))].copy()
    if "sales_per_visitor" not in prev_df.columns:
        prev_df["sales_per_visitor"] = np.where(prev_df["count_in"]>0, prev_df["turnover"]/prev_df["count_in"], 0.0)
    if "sq_meter" in prev_df.columns:
        prev_df["sales_per_sqm"] = np.where(prev_df["sq_meter"]>0, prev_df["turnover"]/prev_df["sq_meter"], np.nan)
    else:
        prev_df["sales_per_sqm"] = np.nan

    prev_turn = float(prev_df["turnover"].sum())
    pw        = prev_df["count_in"].fillna(0.0)
    prev_conv = (prev_df["conversion_rate"].fillna(0.0)*pw).sum()/pw.sum() if pw.sum()>0 else np.nan
    prev_spv  = (prev_df["sales_per_visitor"].fillna(0.0)*pw).sum()/pw.sum() if pw.sum()>0 else np.nan
    prev_spsqm= (prev_df["turnover"].sum()/prev_df["sq_meter"].fillna(0).sum()) if prev_df["sq_meter"].fillna(0).sum()>0 else np.nan
else:
    prev_turn = prev_conv = prev_spv = prev_spsqm = np.nan

def delta_badge(delta_val, kind="eur"):
    if pd.isna(delta_val):
        return f'<span class="kpi-delta flat">n.v.t.</span>'
    if delta_val > 0: cls="up";   arrow="↑"
    elif delta_val < 0: cls="down"; arrow="↓"
    else: cls="flat"; arrow="→"
    if kind=="eur": disp = f"€{abs(delta_val):,.0f}".replace(",", ".")
    elif kind=="pct2": disp = f"{abs(delta_val):.2f}pp"
    else: disp = f"{abs(delta_val):.2f}"
    return f'<span class="kpi-delta {cls}">{arrow} {disp} vs vorige</span>'

c1,c2,c3 = st.columns(3)

with c1:
    d = total_turn - prev_turn if not pd.isna(prev_turn) else np.nan
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-title">💶 Totale omzet</div>
      <div class="kpi-value">{eur0(total_turn)}</div>
      {delta_badge(d,"eur")}
    </div>
    """, unsafe_allow_html=True)

with c2:
    d = (avg_conv - prev_conv) if not pd.isna(prev_conv) else np.nan
    show = pct2(avg_conv) if not pd.isna(avg_conv) else "n.v.t."
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-title">🛒 Gem. conversie</div>
      <div class="kpi-value">{show}</div>
      {delta_badge(d,"pct2")}
    </div>
    """, unsafe_allow_html=True)

with c3:
    d = (avg_spv - prev_spv) if not pd.isna(prev_spv) else np.nan
    show = eur0(avg_spv) if not pd.isna(avg_spv) else "n.v.t."
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-title">💸 Gem. SPV</div>
      <div class="kpi-value">{show}</div>
      {delta_badge(d,"eur")}
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ───────────────────────── Debug (veilig) ────────────────────────────
with st.expander("🔧 Debug — fetch"):
    st.write("HTTP:", s_cur, "rows:", len(df), "rows (uren gefilterd):", len(df_h))
    st.write("Params sample:", p_cur[:10])
    st.write("Pivot shape:", pivot.shape if 'pivot' in locals() else "(geen)")
