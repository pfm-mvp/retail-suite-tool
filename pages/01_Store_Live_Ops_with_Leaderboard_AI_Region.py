# pages/01_Store_Live_Ops_with_Leaderboard_AI_Region.py
import os, sys, math, json
from datetime import datetime
import pytz
import requests
import numpy as np
import pandas as pd
import streamlit as st

# ────────────────────────────── Page config (exact 1x) ─────────────────────────
st.set_page_config(page_title="Store Live Ops — Gisteren vs Eergisteren + Leaderboard",
                   page_icon="🛍️", layout="wide")
st.title("🛍️ Store Live Ops — Gisteren vs Eergisteren + Leaderboard")

# ────────────────────────────── Imports / mapping ───────────────────────────────
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))
from helpers_shop import ID_TO_NAME, NAME_TO_ID   # robuuste maps (id->name / name->id)
from helpers_normalize import normalize_vemcount_response

API_URL = st.secrets["API_URL"]

# ────────────────────────────── Kleuren & CSS ───────────────────────────────────
PFM_RED = "#F04438"
PFM_GREEN = "#22C55E"
PFM_PURPLE = "#6C4EE3"
PFM_GRAY = "#6B7280"
PFM_GRAY_BG = "rgba(107,114,128,.10)"

st.markdown(f"""
<style>
.kpi-card {{ border: 1px solid #EEE; border-radius: 14px; padding: 18px 18px 14px 18px; }}
.kpi-title {{ color:#0C111D; font-weight:600; font-size:16px; margin-bottom:8px; }}
.kpi-value {{ font-size:40px; font-weight:800; line-height:1.1; margin-bottom:6px; }}
.kpi-delta {{ font-size:14px; font-weight:700; padding:4px 10px; border-radius:999px; display:inline-block; }}
.kpi-delta.up {{ color:{PFM_GREEN}; background: rgba(34,197,94,.10); }}
.kpi-delta.down {{ color:{PFM_RED}; background: rgba(240,68,56,.10); }}
.kpi-delta.flat {{ color:{PFM_GRAY}; background: {PFM_GRAY_BG}; }}
.lb-card {{ border: 1px dashed #DDD; border-radius: 12px; padding: 12px 14px; margin-bottom: 8px; background: #FAFAFC; }}
.lb-title {{ font-size:14px; color:#0C111D; font-weight:600; }}
.lb-val {{ font-size:18px; font-weight:800; }}
</style>
""", unsafe_allow_html=True)

# ────────────────────────────── Store picker (1x) ───────────────────────────────
if not NAME_TO_ID:
    st.error("Geen winkels geladen (NAME_TO_ID is leeg). Controleer helpers_shop.py of shop_mapping.py.")
    st.stop()

store_options = sorted(ID_TO_NAME.values())   # namen
store_name    = st.selectbox("Kies winkel", store_options, index=0, key="store_pick")
store_id      = NAME_TO_ID.get(store_name)
if store_id is None:
    st.error("Kon de geselecteerde winkel niet mappen naar een ID.")
    st.stop()

# ────────────────────────────── Tijd & KPI’s ────────────────────────────────────
METRICS = ["count_in","conversion_rate","turnover","sales_per_visitor"]
TZ = pytz.timezone("Europe/Amsterdam")
TODAY = datetime.now(TZ).date()

def post_report(params):
    r = requests.post(API_URL, params=params, timeout=45)
    r.raise_for_status()
    return r

def add_effective_date(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "date" not in d.columns:
        d["date"] = pd.Series([None]*len(d))
    ts = pd.to_datetime(d.get("timestamp"), errors="coerce")
    date_series = pd.to_datetime(d["date"], errors="coerce")
    d["date_eff"] = date_series.fillna(ts).dt.date
    return d

# ---------- util: veilige normalisatie + logging ----------
def norm_and_dateframe(js, metrics):
    df = normalize_vemcount_response(js, ID_TO_NAME, kpi_keys=metrics)  # verwacht {id->naam}
    df = add_effective_date(df)
    return df

# ---------- fetch: multi-call met fallback per winkel + uitgebreide debug ----------
def fetch_df(shop_ids, period, step, metrics, label=""):
    params = [("data", sid) for sid in shop_ids]
    params += [("data_output", m) for m in metrics]
    params += [("source","shops"), ("period", period), ("step", step)]
    resp = post_report(params)
    status = resp.status_code
    try:
        js = resp.json()
    except Exception:
        js = {"_non_json": True, "_text": resp.text[:800]}

    df = norm_and_dateframe(js, metrics)
    multi_rows = len(df)

    debug_info = {
        "label": label,
        "http_status": status,
        "payload_pairs": params[:10],  # eerste 10 voor leesbaarheid
        "json_keys": (list(js.keys())[:10] if isinstance(js, dict) else type(js).__name__),
        "rows_multi": multi_rows,
        "cols_multi": list(df.columns) if not df.empty else [],
        "date_eff_min": (str(df["date_eff"].min()) if "date_eff" in df.columns and not df.empty else None),
        "date_eff_max": (str(df["date_eff"].max()) if "date_eff" in df.columns and not df.empty else None),
    }

    # Fallback: als multi-call leeg is maar er zijn shops, probeer per shop (max 12 voor snelheid)
    fallback_frames = []
    fallback_logs = []
    if multi_rows == 0 and len(shop_ids) > 1:
        for sid in shop_ids[:12]:
            p1 = [("data", sid)] + [("data_output", m) for m in metrics] + [("source","shops"), ("period", period), ("step", step)]
            r1 = post_report(p1)
            try:
                js1 = r1.json()
            except Exception:
                js1 = {"_non_json": True, "_text": r1.text[:400]}
            d1 = norm_and_dateframe(js1, metrics)
            if not d1.empty:
                fallback_frames.append(d1)
            fallback_logs.append({"sid": sid, "status": r1.status_code, "rows": len(d1)})

    if fallback_frames:
        df_fb = pd.concat(fallback_frames, ignore_index=True)
        debug_info["fallback_used"] = True
        debug_info["fallback_samples"] = fallback_logs[:6]
        debug_info["rows_fallback"] = len(df_fb)
        # zet df naar fallbackresultaat
        df = df_fb
    else:
        debug_info["fallback_used"] = False
        debug_info["fallback_samples"] = fallback_logs[:6]

    return df, params, status, debug_info

# ─────────────────────── Cards (gisteren vs eergisteren) ───────────────────────
df_cards, p_cards, status_cards, dbg_cards = fetch_df([store_id], "this_week", "day", METRICS, label="cards:this_week")
# “t/m gisteren”-filter
pre_rows_cards = len(df_cards)
df_cards = df_cards[df_cards.get("date_eff") < TODAY]
post_rows_cards = len(df_cards)

dates = sorted(df_cards["date_eff"].unique()) if "date_eff" in df_cards.columns else []
if len(dates) < 2:
    df_cards, p_cards, status_cards, dbg_cards2 = fetch_df([store_id], "last_week", "day", METRICS, label="cards:last_week")
    pre_rows_cards2 = len(df_cards)
    df_cards = df_cards[df_cards.get("date_eff") < TODAY]
    post_rows_cards2 = len(df_cards)
    dates = sorted(df_cards["date_eff"].unique()) if "date_eff" in df_cards.columns else []
    dbg_cards = {"try2": dbg_cards2, "pre_rows": pre_rows_cards2, "post_rows": post_rows_cards2}

if len(dates) < 2:
    st.error("Niet genoeg dagdata om kaarten te tonen.")
    st.stop()

ydate, bdate = dates[-1], dates[-2]
gy = df_cards[(df_cards["date_eff"]==ydate) & (df_cards["shop_id"]==store_id)][METRICS].sum(numeric_only=True)
gb = df_cards[(df_cards["date_eff"]==bdate) & (df_cards["shop_id"]==store_id)][METRICS].sum(numeric_only=True)

def delta_badge(value: float, kind: str):
    if value > 0: cls, arrow = "up","↑"
    elif value < 0: cls, arrow = "down","↓"
    else: cls, arrow = "flat","→"
    if kind=="int":  disp = f"{abs(int(value)):,}".replace(",", ".")
    elif kind=="pct2": disp = f"{abs(value):.2f}%"
    elif kind=="eur0": disp = f"€{abs(value):,.0f}".replace(",", ".")
    elif kind=="eur2": disp = f"€{abs(value):,.2f}".replace(",", ".")
    else: disp = f"{abs(value)}"
    return f'<span class="kpi-delta {cls}">{arrow} <b>{disp}</b> t.o.v. dag ervoor</span>'

c1,c2,c3,c4 = st.columns(4)

with c1:
    val = int(gy.get("count_in", 0))
    diff = gy.get("count_in", 0) - gb.get("count_in", 0)
    st.markdown(f"""
<div class="kpi-card"><div class="kpi-title">Bezoekers <small>(gisteren)</small></div>
<div class="kpi-value">{val:,}</div>{delta_badge(float(diff),"int")}</div>
""".replace(",", "."), unsafe_allow_html=True)

with c2:
    conv_y = float(gy.get("conversion_rate", 0.0))
    conv_b = float(gb.get("conversion_rate", 0.0))
    st.markdown(f"""
<div class="kpi-card"><div class="kpi-title">Conversie <small>(gisteren)</small></div>
<div class="kpi-value">{conv_y:.2f}%</div>{delta_badge((conv_y-conv_b),"pct2")}</div>
""", unsafe_allow_html=True)

with c3:
    turn_y = float(gy.get("turnover", 0.0)); turn_b = float(gb.get("turnover", 0.0))
    st.markdown(f"""
<div class="kpi-card"><div class="kpi-title">Omzet <small>(gisteren)</small></div>
<div class="kpi-value">€{turn_y:,.0f}</div>{delta_badge((turn_y-turn_b),"eur0")}</div>
""".replace(",", "."), unsafe_allow_html=True)

with c4:
    spv_y = float(gy.get("sales_per_visitor", 0.0)); spv_b = float(gb.get("sales_per_visitor", 0.0))
    if spv_y==0 and val>0: spv_y = turn_y/val
    val_b = int(gb.get("count_in", 0))
    if spv_b==0 and val_b>0: spv_b = turn_b/val_b
    st.markdown(f"""
<div class="kpi-card"><div class="kpi-title">Sales per visitor <small>(gisteren)</small></div>
<div class="kpi-value">€{spv_y:,.2f}</div>{delta_badge((spv_y-spv_b),"eur2")}</div>
""".replace(",", "."), unsafe_allow_html=True)

st.markdown("---")

# ───────────────────── Leaderboard (WTD t/m gisteren) ──────────────────────────
all_ids = list(ID_TO_NAME.keys())

def fetch_wtd(period):
    df, p, s, dbg = fetch_df(all_ids, period, "day", METRICS, label=f"wtd:{period}")
    total_before = len(df)
    df = df[df.get("date_eff") < TODAY]
    return df, p, s, dbg, total_before, len(df)

df_this, p_this, s_this, dbg_this, rows_before_this, rows_after_this = fetch_wtd("this_week")
df_last, p_last, s_last, dbg_last, rows_before_last, rows_after_last = fetch_wtd("last_week")

st.subheader("🏁 Leaderboard — huidige week (t/m gisteren)")
rank_choice = st.radio("Ranking op basis van", ["Conversie", "SPV"], horizontal=True, index=0)

def wtd_agg(d: pd.DataFrame) -> pd.DataFrame:
    if d is None or d.empty: return pd.DataFrame()
    g = d.groupby("shop_id", as_index=False).agg({"count_in":"sum","turnover":"sum"})
    g["sales_per_visitor"] = g.apply(lambda r: (r["turnover"]/r["count_in"]) if r["count_in"] else 0.0, axis=1)
    conv = d.groupby("shop_id").apply(
        lambda x: (x["conversion_rate"]*x["count_in"]).sum()/x["count_in"].sum() if x["count_in"].sum()
        else float(x["conversion_rate"].mean())
    ).reset_index()
    conv.columns = ["shop_id","conversion_rate"]
    g = g.merge(conv, on="shop_id", how="left")
    g["shop_name"] = g["shop_id"].map(ID_TO_NAME)
    return g

agg_this = wtd_agg(df_this)
agg_last = wtd_agg(df_last)

if agg_this.empty:
    with st.expander("🔧 WTD Debug (openen bij lege tabel)"):
        st.write("this_week — HTTP / payload / json / vorm:", dbg_this)
        st.write("last_week  — HTTP / payload / json / vorm:", dbg_last)
        st.write("Rows before/after < TODAY filter — this_week:", rows_before_this, rows_after_this,
                 " last_week:", rows_before_last, rows_after_last)
        st.write("df_this head:", df_this.head() if not df_this.empty else "(leeg)")
        st.write("df_last head:", df_last.head() if not df_last.empty else "(leeg)")
    st.info("Geen WTD data beschikbaar.")
    st.stop()

metric_map = {"Conversie":"conversion_rate", "SPV":"sales_per_visitor"}
rank_metric = metric_map[rank_choice]

agg_this["rank_now"] = agg_this[rank_metric].rank(method="min", ascending=False).astype(int)
tmp = agg_last[["shop_id", rank_metric]].copy()
tmp["rank_last"] = tmp[rank_metric].rank(method="min", ascending=False)
agg = agg_this.merge(tmp[["shop_id","rank_last"]], on="shop_id", how="left")

def safe_int(x):
    try:
        return int(x) if pd.notna(x) else None
    except Exception:
        return None

agg["rank_last_int"] = agg["rank_last"].apply(safe_int)
agg["rank_change"] = agg.apply(lambda r: (r["rank_last_int"] - r["rank_now"]) if r["rank_last_int"] is not None else 0, axis=1)

me = agg[agg["shop_id"]==store_id].iloc[0]
ch = int(me["rank_change"]) if pd.notna(me["rank_change"]) else 0
arrow = "🔺" if ch>0 else ("🔻" if ch<0 else "→")
col = PFM_PURPLE if ch>0 else (PFM_RED if ch<0 else PFM_GRAY)
st.markdown(f"""
<div class="lb-card">
  <div class="lb-title">Jouw positie op <b>{'Conversie' if rank_metric=='conversion_rate' else 'SPV'}</b></div>
  <div class="lb-val" style="color:{col}">#{int(me["rank_now"])} {arrow} {abs(ch)} t.o.v. vorige week</div>
</div>
""", unsafe_allow_html=True)

agg = agg.sort_values("rank_now").reset_index(drop=True)
def pos_text(r):
    last = r["rank_last_int"]; now = int(r["rank_now"])
    last_txt = "–" if last is None else str(last)
    arrow = "⬆" if r["rank_change"]>0 else ("⬇" if r["rank_change"]<0 else "→")
    return f"{now} {arrow} {last_txt}"
agg["positie (nu vs lw)"] = agg.apply(pos_text, axis=1)

show_cols = ["positie (nu vs lw)","shop_name","count_in","conversion_rate","sales_per_visitor","turnover"]
fmt = agg[show_cols].copy()
fmt["count_in"] = fmt["count_in"].map(lambda x: f"{int(x):,}".replace(",", "."))
fmt["conversion_rate"] = fmt["conversion_rate"].map(lambda x: f"{float(x):.2f}%")
fmt["sales_per_visitor"] = fmt["sales_per_visitor"].map(lambda x: f"€{float(x):,.2f}".replace(",", "."))
fmt["turnover"] = fmt["turnover"].map(lambda x: f"€{float(x):,.0f}".replace(",", "."))

try:
    idx_me = fmt.index[agg["shop_name"] == store_name][0]
except Exception:
    idx_me = None

def highlight_rows(row):
    if idx_me is not None and row.name == idx_me:
        ch = int(agg.loc[row.name, "rank_change"]) if pd.notna(agg.loc[row.name, "rank_change"]) else 0
        if ch > 0: bg = PFM_PURPLE
        elif ch < 0: bg = PFM_RED
        else: bg = PFM_GRAY
        return [f"background-color: {bg}; color: white" for _ in row]
    return [""]*len(row)

styler = fmt.style.apply(highlight_rows, axis=1)

def color_pos_col(col):
    styles=[]
    for i,_ in enumerate(col):
        ch = int(agg.loc[fmt.index[i], "rank_change"]) if pd.notna(agg.loc[fmt.index[i], "rank_change"]) else 0
        if ch > 0: styles.append(f"color: {PFM_PURPLE}; font-weight: 700")
        elif ch < 0: styles.append(f"color: {PFM_RED}; font-weight: 700")
        else: styles.append(f"color: {PFM_GRAY}; font-weight: 700")
    return styles

styler = styler.apply(color_pos_col, subset=["positie (nu vs lw)"])
st.dataframe(styler, use_container_width=True)

# ────────────────────────────── AI Coach ────────────────────────────────────────
st.markdown("## 🤖 AI Coach")

def pct(x):
    try: return float(x)/100.0
    except: return 0.0
def eur(x, decimals=0): return f"€{x:,.{decimals}f}".replace(",", ".")
def safe_div(a, b): return (a/b) if (b and not np.isnan(b) and b!=0) else 0.0

vis_y   = float(gy.get("count_in", 0))
turn_y  = float(gy.get("turnover", 0.0))
conv_y  = float(gy.get("conversion_rate", 0.0))
spv_y   = float(gy.get("sales_per_visitor", 0.0))
if spv_y == 0 and vis_y > 0: spv_y = turn_y/vis_y

vis_b   = float(gb.get("count_in", 0))
turn_b  = float(gb.get("turnover", 0.0))
conv_b  = float(gb.get("conversion_rate", 0.0))
spv_b   = float(gb.get("sales_per_visitor", 0.0))
if spv_b == 0 and vis_b > 0: spv_b = turn_b/vis_b

conv_y_f = pct(conv_y)
conv_b_f = pct(conv_b)
atv_y    = safe_div(spv_y, conv_y_f)

peer = agg_this.copy()
peer_conv_med = float(peer["conversion_rate"].median())
peer_spv_med  = float(peer["sales_per_visitor"].median())

target_conv = max(peer_conv_med, conv_b)          # in %
target_spv  = max(peer_spv_med,  spv_b)

conv_target_f = pct(target_conv)
delta_conv_f  = max(0.0, conv_target_f - conv_y_f)
uplift_conv   = vis_y * delta_conv_f * atv_y

delta_spv     = max(0.0, target_spv - spv_y)
uplift_spv    = vis_y * delta_spv

rules = []
if vis_b > 0 and (vis_y - vis_b)/vis_b >= 0.15 and (conv_y - conv_b) <= -3.0:
    rules.append("**Traffic +15% vs dag ervoor maar conversie daalt −3pp** → verschuif 1 FTE naar vloer/paskamers, focus begroeten & hulp aanbieden.")
if conv_y < peer_conv_med:
    rules.append(f"**Conversie onder peermediaan ({peer_conv_med:.2f}%)** → korte floor-coaching: active greet, 3 vragen-regel, stuur op demo/proefpassen.")
if spv_y < peer_spv_med:
    rules.append(f"**SPV onder peermediaan ({eur(peer_spv_med,2)})** → push ‘2e artikel −20%’ of bundels; kassascripts voor add-ons.")

ch_here = int(me["rank_change"]) if not pd.isna(me["rank_change"]) else 0
if ch_here > 0:
    rules.append("**Gestegen in ranking** → hou ritme vast; cap drukte-uren met vaste vloer-rollen (greeter, paskamerhost, queue-busting).")
elif ch_here < 0:
    rules.append("**Gezakt in ranking** → check wachttijd aan kassa & passtijden; daily huddle met 1 micro-doel voor vandaag.")

# (optionele weer-hook hier weggelaten om focus te houden)

# ────────────────────────────── Debug (optioneel) ───────────────────────────────
with st.expander("🔧 Debug — API calls"):
    st.write("Cards call — payload (first 10):", p_cards[:10], "| http:", status_cards)
    st.write("Cards debug:", dbg_cards)
    st.write("WTD this_week payload (first 10):", p_this[:10], "| http:", s_this)
    st.write("WTD last_week  payload (first 10):", p_last[:10], "| http:", s_last)
    st.write("WTD this_week debug:", dbg_this)
    st.write("WTD last_week  debug:", dbg_last)
