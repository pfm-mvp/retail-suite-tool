
import os, sys
import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

# ---------- Imports / mapping ----------
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))
from shop_mapping import SHOP_NAME_MAP
from helpers_normalize import normalize_vemcount_response

st.set_page_config(page_title="Store Live Ops", page_icon="🛍️", layout="wide")
st.title("🛍️ Store Live Ops – Gisteren vs Eergisteren + Leaderboard (WTD)")

API_URL = st.secrets["API_URL"]

# ---------- UI: store + leaderboard metric ----------
NAME_TO_ID = {v:k for k,v in SHOP_NAME_MAP.items()}
ID_TO_NAME = {k:v for k,v in SHOP_NAME_MAP.items()}

store_name = st.selectbox("Kies winkel", list(NAME_TO_ID.keys()), index=0)
store_id = NAME_TO_ID[store_name]

metric_for_rank = st.selectbox("Leaderboard op basis van", ["conversion_rate","sales_per_visitor","turnover","count_in"], index=0)

# We gebruiken altijd day-level: period_step=day (anders geen dagrecords)
METRICS = ["count_in","conversion_rate","turnover","sales_per_visitor"]

def fetch(params):
    r = requests.post(API_URL, params=params, timeout=45)
    r.raise_for_status()
    return r.json()

# ---------- Cards: gisteren vs eergisteren ----------
params_cards = [("data", store_id)]
params_cards += [("data_output", m) for m in METRICS]
params_cards += [("source","shops"),("period","this_month"),("step","day")]  # step=day is essentieel

try:
    js_cards = fetch(params_cards)
except Exception as e:
    st.error(f"API fout (cards): {e}")
    st.stop()

df_cards = normalize_vemcount_response(js_cards, SHOP_NAME_MAP, kpi_keys=METRICS)
dfd = df_cards.dropna(subset=["date"]).copy()
if dfd.empty:
    st.warning("Geen data voor kaarten.")
    st.stop()

dates_sorted = sorted(dfd["date"].unique())
if len(dates_sorted) < 2:
    st.warning("Niet genoeg dagen data om te vergelijken.")
    st.dataframe(dfd.tail(5))
    st.stop()

yesterday, before = dates_sorted[-1], dates_sorted[-2]

g_y = dfd[dfd["date"]==yesterday].groupby("shop_id")[METRICS].sum(numeric_only=True).reset_index()
g_b = dfd[dfd["date"]==before].groupby("shop_id")[METRICS].sum(numeric_only=True).reset_index()
ry = g_y[g_y["shop_id"]==store_id].iloc[0]
rb = g_b[g_b["shop_id"]==store_id].iloc[0]

c1,c2,c3,c4 = st.columns(4)
with c1:
    st.metric("Bezoekers (gisteren)", f"{int(ry['count_in']):,}".replace(",","."),
              delta=f"{int(ry['count_in']-rb['count_in']):,} vs eergisteren".replace(",","."))
with c2:
    st.metric("Conversie (gisteren)", f"{ry['conversion_rate']:.2f}%",
              delta=f"{(ry['conversion_rate']-rb['conversion_rate']):+.2f} pp")
with c3:
    st.metric("Omzet (gisteren)", f"€{ry['turnover']:,.0f}".replace(",","."),
              delta=f"€{(ry['turnover']-rb['turnover']):+,.0f}".replace(",","."))
with c4:
    st.metric("Sales/visitor (gisteren)", f"€{ry['sales_per_visitor']:,.2f}".replace(",","."),
              delta=f"{(ry['sales_per_visitor']-rb['sales_per_visitor']):+.2f}")

st.markdown("---")

# ---------- Leaderboard: Week-To-Date (t/m gisteren) ----------
# We halen *this_week* en *last_week* op met step=day voor alle winkels
all_ids = list(SHOP_NAME_MAP.keys())

def fetch_week(period: str):
    params = []
    for sid in all_ids: params.append(("data", sid))
    for m in METRICS: params.append(("data_output", m))
    params += [("source","shops"), ("period", period), ("step","day")]
    js = fetch(params)
    df = normalize_vemcount_response(js, SHOP_NAME_MAP, kpi_keys=METRICS)
    return df

try:
    df_this = fetch_week("this_week")
    df_last = fetch_week("last_week")
except Exception as e:
    st.error(f"API fout (leaderboard): {e}")
    st.stop()

# We nemen voor "this_week" alleen data t/m gisteren (sales wordt vaak pas later volledig)
# Bepaal gisteren op basis van data aanwezig in df_this (laatste dag met records is gisteren)
if df_this.empty:
    st.warning("Geen data voor leaderboard.")
else:
    # bepaal 'gisteren' als max beschikbare datum in this_week
    dates_this = sorted([d for d in df_this["date"].dropna().unique()])
    cutoff = dates_this[-1] if dates_this else None

    def wtd_aggregate(df: pd.DataFrame, upto_date: str | None) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        d = df.dropna(subset=["date"]).copy()
        if upto_date:
            d = d[d["date"] <= upto_date]
        # sommen voor count_in en turnover; voor rates/averages doen we weighted
        g = d.groupby("shop_id", as_index=False).agg({
            "count_in":"sum",
            "turnover":"sum",
            "sales_per_visitor":"mean",   # we overschrijven zo met weighted
            "conversion_rate":"mean"      # idem
        })
        # weighted averages
        ci = g["count_in"].replace(0, pd.NA)
        # SPV weighted als turnover / count_in wanneer beschikbaar
        g["sales_per_visitor"] = (g["turnover"] / g["count_in"]).fillna(g["sales_per_visitor"])
        # conversie gewogen met count_in indien beschikbaar
        # API levert conversie in %, dus we nemen gewogen gemiddelde
        d_nonzero = d.copy()
        # merge som count_in per shop voor wgt
        w = d.groupby("shop_id", as_index=False)["count_in"].sum().rename(columns={"count_in":"wgt"})
        g = g.merge(w, on="shop_id", how="left")
        g["conversion_rate"] = (d.groupby("shop_id").apply(
            lambda x: (x["conversion_rate"]*x["count_in"]).sum() / x["count_in"].sum()
            if x["count_in"].sum() else x["conversion_rate"].mean()
        ).reset_index(drop=True))
        # shop_name erbij
        g["shop_name"] = g["shop_id"].map(SHOP_NAME_MAP)
        return g

    agg_this = wtd_aggregate(df_this, cutoff)
    agg_last = wtd_aggregate(df_last, None)  # hele last_week

    # Ranking nu (this week t/m gisteren)
    rank_metric = metric_for_rank
    ascending = False  # high is better
    if rank_metric in ["count_in","turnover","sales_per_visitor","conversion_rate"]:
        ascending = False
    agg_this["rank_now"] = agg_this[rank_metric].rank(method="min", ascending=ascending).astype(int)

    # Ranking vorige week
    agg_last = agg_last[["shop_id", rank_metric]].rename(columns={rank_metric:"metric_last"})
    # compute last rank
    tmp = agg_last.copy()
    tmp["rank_last"] = tmp["metric_last"].rank(method="min", ascending=ascending).astype(int)
    agg = agg_this.merge(tmp[["shop_id","rank_last"]], on="shop_id", how="left")
    agg["rank_change"] = agg["rank_last"] - agg["rank_now"]  # + = omhoog (verbetering)

    # sorteren op huidige rank
    agg = agg.sort_values("rank_now")

    # highlight eigen winkel
    def pos_delta(r):
        change = int(r["rank_change"]) if pd.notna(r["rank_change"]) else 0
        arrow = "🔺" if change>0 else ("🔻" if change<0 else "→")
        return f"{int(r['rank_now'])} {arrow} {abs(change)}"

    agg["positie (nu vs lw)"] = agg.apply(pos_delta, axis=1)

    show_cols = ["positie (nu vs lw)","shop_name","count_in","conversion_rate","sales_per_visitor","turnover"]
    # EU notatie
    fmt = agg[show_cols].copy()
    fmt["count_in"] = fmt["count_in"].map(lambda x: f"{int(x):,}".replace(",","."))
    fmt["conversion_rate"] = fmt["conversion_rate"].map(lambda x: f"{x:.2f}%")
    fmt["sales_per_visitor"] = fmt["sales_per_visitor"].map(lambda x: f"€{x:,.2f}".replace(",","."))
    fmt["turnover"] = fmt["turnover"].map(lambda x: f"€{x:,.0f}".replace(",","."))

    st.subheader("🏁 Leaderboard — huidige week (t/m gisteren)")
    st.caption(f"Ranking op: **{rank_metric}** · Jouw winkel gemarkeerd")
    # markeer eigen winkel
    fmt_style = fmt.style.apply(lambda s: ["background-color: #FFF3CD" if (s.name==fmt[fmt['shop_name']==store_name].index[0]) else "" for _ in s], axis=1)
    st.dataframe(fmt_style, use_container_width=True)

# ---------- Debug ----------
with st.expander("🔧 Debug (API params)"):
    st.write("Cards params:", params_cards)
    st.write("Week fetch uses step=day with periods this_week / last_week for all stores.")
