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

NAME_TO_ID = {v:k for k,v in SHOP_NAME_MAP.items()}
ID_TO_NAME = {k:v for k,v in SHOP_NAME_MAP.items()}

store_name = st.selectbox("Kies winkel", list(NAME_TO_ID.keys()), index=0)
store_id = NAME_TO_ID[store_name]

metric_for_rank = st.selectbox("Leaderboard op basis van", ["conversion_rate","sales_per_visitor","turnover","count_in"], index=0)

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
    # parse timestamp if present (handles strings like 'Fri. Aug 8, 2025')
    ts = pd.to_datetime(d.get("timestamp"), errors="coerce")
    date_series = pd.to_datetime(d["date"], errors="coerce")
    # prefer 'date', fallback to parsed timestamp
    d["date_eff"] = date_series.fillna(ts).dt.date
    return d

# ---------- Cards: gisteren vs eergisteren ----------
params_cards = [("data", store_id)]
params_cards += [("data_output", m) for m in METRICS]
params_cards += [("source","shops"), ("period","this_week"), ("step","day")]

def pick_yesterday_pairs(js):
    df = normalize_vemcount_response(js, SHOP_NAME_MAP, kpi_keys=METRICS)
    dfd = add_effective_date(df).dropna(subset=["date_eff"]).copy()
    if dfd.empty:
        return None, None, None, None, dfd

    # Exclude TODAY to avoid zeros (sales posted next day)
    dfd = dfd[dfd["date_eff"] < TODAY]

    dates_sorted = sorted(dfd["date_eff"].unique())
    if len(dates_sorted) >= 2:
        y, b = dates_sorted[-1], dates_sorted[-2]
        g_y = dfd[dfd["date_eff"]==y].groupby("shop_id")[METRICS].sum(numeric_only=True).reset_index()
        g_b = dfd[dfd["date_eff"]==b].groupby("shop_id")[METRICS].sum(numeric_only=True).reset_index()
        return g_y, g_b, y.isoformat(), b.isoformat(), dfd
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
    st.warning("Geen voldoende dagdata om kaarten te tonen (na uitsluiten van vandaag).")
    st.stop()

ry = gy[gy["shop_id"]==store_id].iloc[0]
rb = gb[gb["shop_id"]==store_id].iloc[0]

c1,c2,c3,c4 = st.columns(4)
with c1:
    st.metric(f"Bezoekers ({ydate})", f"{int(ry['count_in']):,}".replace(",","."),
              delta=f"{int(ry['count_in']-rb['count_in']):,} vs {bdate}".replace(",","."))
with c2:
    st.metric(f"Conversie ({ydate})", f"{ry['conversion_rate']:.2f}%",
              delta=f"{(ry['conversion_rate']-rb['conversion_rate']):+.2f} pp")
with c3:
    st.metric(f"Omzet ({ydate})", f"€{ry['turnover']:,.0f}".replace(",","."),
              delta=f"€{(ry['turnover']-rb['turnover']):+,.0f}".replace(",","."))
with c4:
    st.metric(f"Sales/visitor ({ydate})", f"€{ry['sales_per_visitor']:,.2f}".replace(",","."),
              delta=f"{(ry['sales_per_visitor']-rb['sales_per_visitor']):+.2f}")

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
    # WTD t/m gisteren -> filter < TODAY
    return d[d["date_eff"] < TODAY]

try:
    df_this = fetch_week("this_week")
    df_last = fetch_week("last_week")
except Exception as e:
    st.error(f"API fout (leaderboard): {e}")
    st.stop()

if df_this.empty:
    st.warning("Geen data voor leaderboard.")
else:
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

    rank_metric = metric_for_rank
    ascending = False
    agg_this["rank_now"] = agg_this[rank_metric].rank(method="min", ascending=ascending).astype(int)

    tmp = agg_last[["shop_id", rank_metric]].copy()
    tmp["rank_last"] = tmp[rank_metric].rank(method="min", ascending=ascending).astype(int)
    agg = agg_this.merge(tmp[["shop_id","rank_last"]], on="shop_id", how="left")
    agg["rank_change"] = agg["rank_last"] - agg["rank_now"]

    agg = agg.sort_values("rank_now")
    def pos_delta(r):
        change = int(r["rank_change"]) if pd.notna(r["rank_change"]) else 0
        arrow = "🔺" if change>0 else ("🔻" if change<0 else "→")
        return f"{int(r['rank_now'])} {arrow} {abs(change)}"

    agg["positie (nu vs lw)"] = agg.apply(pos_delta, axis=1)

    show_cols = ["positie (nu vs lw)","shop_name","count_in","conversion_rate","sales_per_visitor","turnover"]
    fmt = agg[show_cols].copy()
    fmt["count_in"] = fmt["count_in"].map(lambda x: f"{int(x):,}".replace(",","."))
    fmt["conversion_rate"] = fmt["conversion_rate"].map(lambda x: f"{x:.2f}%")
    fmt["sales_per_visitor"] = fmt["sales_per_visitor"].map(lambda x: f"€{x:,.2f}".replace(",","."))
    fmt["turnover"] = fmt["turnover"].map(lambda x: f"€{x:,.0f}".replace(",","."))

    st.subheader("🏁 Leaderboard — huidige week (t/m gisteren)")
    st.caption(f"Ranking op: **{rank_metric}** · Jouw winkel gemarkeerd")

    try:
        highlight_idx = fmt.index[fmt["shop_name"] == store_name][0]
        style = fmt.style.apply(lambda s: ["background-color: #FFF3CD" if s.name==highlight_idx else "" for _ in s], axis=1)
        st.dataframe(style, use_container_width=True)
    except Exception:
        st.dataframe(fmt, use_container_width=True)

with st.expander("🔧 Debug"):
    st.write("Cards params:", params_cards)
    st.write("Vandaag (Europe/Amsterdam):", str(TODAY))
    st.write("Laatste kaart-datums (na filter < vandaag):", sorted(dfd_cards["date_eff"].unique())[-7:])
