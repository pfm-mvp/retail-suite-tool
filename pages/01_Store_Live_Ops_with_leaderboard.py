import os, sys
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import pytz

# ---------- Imports / mapping ----------
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))
from shop_mapping import SHOP_NAME_MAP
from helpers_normalize import normalize_vemcount_response

st.set_page_config(page_title="Store Live Ops", page_icon="🛍️", layout="wide")
st.title("🛍️ Store Live Ops – Gisteren vs Eergisteren + Leaderboard (WTD)")

API_URL = st.secrets["API_URL"]

# ---------- PFM palette ----------
PFM_RED = "#F04438"
PFM_GREEN = "#22C55E"
PFM_PURPLE = "#6C4EE3"
PFM_GRAY = "#6B7280"

# ---------- Small CSS for cards ----------
st.markdown(f'''
<style>
.kpi-card {{
  border: 1px solid #EEE;
  border-radius: 14px;
  padding: 18px 18px 14px 18px;
}}
.kpi-title {{
  color:#0C111D; font-weight:600; font-size:16px; margin-bottom:8px;
}}
.kpi-value {{
  font-size:40px; font-weight:800; line-height:1.1; margin-bottom:6px;
}}
.kpi-delta {{ font-size:14px; font-weight:700; padding:4px 10px; border-radius:999px; display:inline-block; }}
.kpi-delta.up {{ color:{PFM_GREEN}; background: rgba(34,197,94,.10); }}
.kpi-delta.down {{ color:{PFM_RED}; background: rgba(240,68,56,.10); }}
.kpi-delta.flat {{ color:{PFM_GRAY}; background: rgba(107,114,128,.10); }}

.lb-card {{
  border: 1px dashed #DDD; border-radius: 12px; padding: 12px 14px; margin-bottom: 8px;
  background: #FAFAFC;
}}
.lb-title {{ font-size:14px; color:#0C111D; font-weight:600; }}
.lb-val {{ font-size:18px; font-weight:800; }}
</style>
''', unsafe_allow_html=True)

NAME_TO_ID = {v:k for k,v in SHOP_NAME_MAP.items()}
ID_TO_NAME = {k:v for k,v in SHOP_NAME_MAP.items()}

store_name = st.selectbox("Kies winkel", list(NAME_TO_ID.keys()), index=0)
store_id = NAME_TO_ID[store_name]

# Metrics we use
METRICS = ["count_in","conversion_rate","turnover","sales_per_visitor"]

TZ = pytz.timezone("Europe/Amsterdam")
TODAY = datetime.now(TZ).date()

def fetch(params):
    r = requests.post(API_URL, params=params, timeout=45)
    r.raise_for_status()
    return r.json()

def add_effective_date(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "date" not in d.columns:
        d["date"] = pd.Series([None]*len(d))
    ts = pd.to_datetime(d.get("timestamp"), errors="coerce")
    date_series = pd.to_datetime(d["date"], errors="coerce")
    d["date_eff"] = date_series.fillna(ts).dt.date
    return d

# ---------- Cards: gisteren vs eergisteren ----------
params_cards = [("data", store_id)]
params_cards += [("data_output", m) for m in METRICS]
params_cards += [("source","shops"), ("period","this_week"), ("step","day")]

def pick_yesterday_pairs(js):
    df = normalize_vemcount_response(js, SHOP_NAME_MAP, kpi_keys=METRICS)
    dfd = add_effective_date(df).dropna(subset=["date_eff"]).copy()
    if dfd.empty: return None, None, None, None, dfd
    # exclude today
    dfd = dfd[dfd["date_eff"] < TODAY]
    dates_sorted = sorted(dfd["date_eff"].unique())
    if len(dates_sorted) >= 2:
        y, b = dates_sorted[-1], dates_sorted[-2]
        g_y = dfd[dfd["date_eff"]==y].groupby("shop_id")[METRICS].sum(numeric_only=True).reset_index()
        g_b = dfd[dfd["date_eff"]==b].groupby("shop_id")[METRICS].sum(numeric_only=True).reset_index()
        return g_y, g_b, y, b, dfd
    return None, None, None, None, dfd

try:
    js_cards = fetch(params_cards)
    gy, gb, ydate, bdate, dfd_cards = pick_yesterday_pairs(js_cards)
    if gy is None:
        params_cards_fallback = [("data", store_id)]
        params_cards_fallback += [("data_output", m) for m in METRICS]
        params_cards_fallback += [("source","shops"), ("period","last_week"), ("step","day")]
        js_cards = fetch(params_cards_fallback)
        gy, gb, ydate, bdate, dfd_cards = pick_yesterday_pairs(js_cards)
except Exception as e:
    st.error(f"API fout (cards): {e}")
    st.stop()

if gy is None:
    st.warning("Geen voldoende dagdata om kaarten te tonen (na filter < vandaag).")
    st.stop()

ry = gy[gy["shop_id"]==store_id].iloc[0]
rb = gb[gb["shop_id"]==store_id].iloc[0]

def signed_text(value: float, kind: str) -> str:
    """Return '+ €1.234' or '- 0,45%' etc. kind in {'eur0','eur2','pct2','int'}"""
    sign = "+" if value >= 0 else "-"
    v = abs(value)
    if kind == "eur0":
        num = f"€{v:,.0f}".replace(",", ".")
    elif kind == "eur2":
        num = f"€{v:,.2f}".replace(",", ".")
    elif kind == "pct2":
        num = f"{v:.2f}%"
    elif kind == "int":
        num = f"{int(v):,}".replace(",", ".")
    else:
        num = f"{v}"
    return f"{sign} {num}"

def delta_badge(value: float, kind: str):
    """Colored badge with arrow & strong signed text followed by 't.o.v. dag ervoor'."""
    if value > 0:
        cls = "up"; arrow = "↑"
    elif value < 0:
        cls = "down"; arrow = "↓"
    else:
        cls = "flat"; arrow = "→"
    return f'<span class="kpi-delta {cls}">{arrow} <b>{signed_text(value, kind)}</b> t.o.v. dag ervoor</span>'

# Build four cards
c1,c2,c3,c4 = st.columns(4)

# Bezoekers
with c1:
    diff = float(ry['count_in'] - rb['count_in'])
    html = f'''
    <div class="kpi-card">
      <div class="kpi-title">Bezoekers <small>(gisteren)</small></div>
      <div class="kpi-value">{int(ry["count_in"]):,}</div>
      {delta_badge(diff, "int")}
    </div>
    '''.replace(",", ".")
    st.markdown(html, unsafe_allow_html=True)

# Conversie (% delta)
with c2:
    val = f"{ry['conversion_rate']:.2f}%"
    diff_pct = float(ry['conversion_rate'] - rb['conversion_rate'])
    html = f'''
    <div class="kpi-card">
      <div class="kpi-title">Conversie <small>(gisteren)</small></div>
      <div class="kpi-value">{val}</div>
      {delta_badge(diff_pct, "pct2")}
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)

# Omzet
with c3:
    val = f"€{ry['turnover']:,.0f}".replace(",", ".")
    diff_eur = float(ry['turnover'] - rb['turnover'])
    html = f'''
    <div class="kpi-card">
      <div class="kpi-title">Omzet <small>(gisteren)</small></div>
      <div class="kpi-value">{val}</div>
      {delta_badge(diff_eur, "eur0")}
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)

# Sales per visitor
with c4:
    val = f"€{ry['sales_per_visitor']:,.2f}".replace(",", ".")
    diff_spv = float(ry['sales_per_visitor'] - rb['sales_per_visitor'])
    html = f'''
    <div class="kpi-card">
      <div class="kpi-title">Sales per visitor <small>(gisteren)</small></div>
      <div class="kpi-value">{val}</div>
      {delta_badge(diff_spv, "eur2")}
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)

st.markdown("---")

# ---------- Leaderboard WTD (t/m gisteren) ----------
all_ids = list(SHOP_NAME_MAP.keys())

def fetch_week(period: str):
    params = []
    for sid in all_ids: params.append(("data", sid))
    for m in METRICS: params.append(("data_output", m))
    params += [("source","shops"), ("period", period), ("step","day")]
    js = fetch(params)
    df = normalize_vemcount_response(js, SHOP_NAME_MAP, kpi_keys=METRICS)
    d = add_effective_date(df)
    return d[d["date_eff"] < TODAY]

try:
    df_this = fetch_week("this_week")
    df_last = fetch_week("last_week")
except Exception as e:
    st.error(f"API fout (leaderboard): {e}")
    st.stop()

st.subheader("🏁 Leaderboard — huidige week (t/m gisteren)")
rank_choice = st.radio("Ranking op basis van", ["Conversie", "SPV"], horizontal=True, index=0,
                       help="Conversie is het meest actionabel; SPV voor upsell focus.")

def wtd_agg(d: pd.DataFrame) -> pd.DataFrame:
    if d is None or d.empty:
        return pd.DataFrame()
    g = d.groupby("shop_id", as_index=False).agg({"count_in":"sum","turnover":"sum"})
    g["sales_per_visitor"] = g.apply(lambda r: (r["turnover"]/r["count_in"]) if r["count_in"] else 0.0, axis=1)
    conv = d.groupby("shop_id").apply(
        lambda x: (x["conversion_rate"]*x["count_in"]).sum()/x["count_in"].sum() if x["count_in"].sum() else x["conversion_rate"].mean()
    ).reset_index()
    conv.columns = ["shop_id","conversion_rate"]
    g = g.merge(conv, on="shop_id", how="left")
    g["shop_name"] = g["shop_id"].map(SHOP_NAME_MAP)
    return g

agg_this = wtd_agg(df_this)
agg_last = wtd_agg(df_last)

if agg_this.empty:
    st.info("Geen WTD data beschikbaar.")
else:
    metric_map = {"Conversie":"conversion_rate", "SPV":"sales_per_visitor"}
    rank_metric = metric_map[rank_choice]
    ascending = False

    agg_this["rank_now"] = agg_this[rank_metric].rank(method="min", ascending=ascending).astype(int)
    tmp = agg_last[["shop_id", rank_metric]].copy()
    tmp["rank_last"] = tmp[rank_metric].rank(method="min", ascending=ascending).astype(int)
    agg = agg_this.merge(tmp[["shop_id","rank_last"]], on="shop_id", how="left")
    agg["rank_change"] = agg["rank_last"] - agg["rank_now"]

    # Leaderboard mini-card for this store
    me = agg[agg["shop_id"]==store_id].iloc[0]
    change = int(me["rank_change"]) if pd.notna(me["rank_change"]) else 0
    arrow = "🔺" if change>0 else ("🔻" if change<0 else "→")
    col = PFM_GREEN if change>0 else (PFM_RED if change<0 else PFM_GRAY)
    st.markdown(f'''
    <div class="lb-card">
      <div class="lb-title">Jouw positie op <b>{'Conversie' if rank_metric=='conversion_rate' else 'SPV'}</b></div>
      <div class="lb-val" style="color:{col}">#{int(me["rank_now"])} {arrow} {abs(change)} t.o.v. vorige week</div>
    </div>
    ''', unsafe_allow_html=True)

    agg = agg.sort_values("rank_now")
    def pos_delta(r):
        change = int(r["rank_change"]) if pd.notna(r["rank_change"]) else 0
        arrow = "🔺" if change>0 else ("🔻" if change<0 else "→")
        color = PFM_GREEN if change>0 else (PFM_RED if change<0 else PFM_GRAY)
        return f"<span style='color:{color};font-weight:600'>{int(r['rank_now'])} {arrow} {abs(change)}</span>"

    agg["positie (nu vs lw)"] = agg.apply(pos_delta, axis=1)

    show_cols = ["positie (nu vs lw)","shop_name","count_in","conversion_rate","sales_per_visitor","turnover"]
    fmt = agg[show_cols].copy()

    # EU notatie & units
    fmt["count_in"] = fmt["count_in"].map(lambda x: f"{int(x):,}".replace(",", "."))
    fmt["conversion_rate"] = fmt["conversion_rate"].map(lambda x: f"{x:.2f}%")
    fmt["sales_per_visitor"] = fmt["sales_per_visitor"].map(lambda x: f"€{x:,.2f}".replace(",", "."))
    fmt["turnover"] = fmt["turnover"].map(lambda x: f"€{x:,.0f}".replace(",", "."))

    try:
        highlight_idx = fmt.index[fmt["shop_name"] == store_name][0]
        def hl(s):
            return ["background-color: %s; color: white" % PFM_PURPLE if s.name==highlight_idx else "" for _ in s]
        styler = fmt.style.apply(hl, axis=1)
    except Exception:
        styler = fmt.style

    # Render HTML in first column
    styler = styler.format({"positie (nu vs lw)": lambda x: x}, escape=None)
    st.dataframe(styler, use_container_width=True)

with st.expander("🔧 Debug"):
    st.write("Vandaag (Europe/Amsterdam):", str(TODAY))
    st.write("Laatste kaart-datums (na filter < vandaag):", sorted(dfd_cards["date_eff"].unique())[-7:])
