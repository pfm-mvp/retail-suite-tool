# pages/03_Retail_Portfolio_Benchmark_AI.py
import os, sys
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ───────── Imports uit project ─────────
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))
from helpers_shop import (
    ID_TO_NAME, REGIONS, get_ids_by_region
)
from helpers_normalize import normalize_vemcount_response, to_wide
from utils_pfmx import api_get_report, friendly_error, inject_css

# ───────── Page setup + CSS ─────────
st.set_page_config(page_title="Retail Portfolio Benchmark (+ uur-proto)", page_icon="🧩", layout="wide")
inject_css()

PFM_GRAY = "#6B7280"
PFM_PURPLE = "#6C4EE3"
PFM_RED = "#F04438"
PFM_GREEN = "#22C55E"

st.markdown("""
<style>
.kpi-card { border:1px solid #EEE; border-radius:14px; padding:18px 18px 14px 18px; }
.kpi-title { color:#0C111D; font-weight:600; font-size:16px; margin-bottom:8px; }
.kpi-value { font-size:40px; font-weight:800; line-height:1.1; margin-bottom:6px; }
.kpi-delta { font-size:14px; font-weight:700; padding:4px 10px; border-radius:999px; display:inline-block; }
.kpi-delta.up { color:%s; background: rgba(34,197,94,.10); }
.kpi-delta.down { color:%s; background: rgba(240,68,56,.10); }
.kpi-delta.flat { color:%s; background: rgba(107,114,128,.10); }
</style>
""" % (PFM_GREEN, PFM_RED, PFM_GRAY), unsafe_allow_html=True)

# ───────── Helpers ─────────
TZ = pytz.timezone("Europe/Amsterdam")

KPI_KEYS_BASE = ["count_in", "conversion_rate", "turnover", "sales_per_visitor", "sq_meter"]
KPI_KEYS_HOURLY = KPI_KEYS_BASE[:]  # idem, we vragen dezelfde velden per uur

def post_report(params):
    r = api_get_report(params)
    return r

def add_effective_date(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "date" not in d.columns:
        d["date"] = pd.NaT
    ts = pd.to_datetime(d.get("timestamp"), errors="coerce")
    d["date_eff"] = pd.to_datetime(d["date"], errors="coerce").fillna(ts)
    # bewaar zowel date als hour
    d["date_only"] = d["date_eff"].dt.date
    d["hour"] = d["date_eff"].dt.hour
    return d

def fetch_df(shop_ids, period, step, metrics):
    params = [("data", sid) for sid in shop_ids]
    params += [("data_output", m) for m in metrics]
    params += [("source","shops"), ("period", period), ("step", step)]
    js = post_report(params)
    if friendly_error(js, period):
        return pd.DataFrame(), params, 200
    df = normalize_vemcount_response(js, ID_TO_NAME, kpi_keys=metrics)
    dfe = add_effective_date(df)
    return dfe, params, 200

def weighted_avg(series, weights):
    try:
        w = weights.fillna(0.0)
        s = series.fillna(0.0)
        d = w.sum()
        return (s*w).sum()/d if d else np.nan
    except Exception:
        return np.nan

def fmt_eur0(x): 
    try: return f"€{float(x):,.0f}".replace(",", ".")
    except: return "€0"

def fmt_eur2(x): 
    try: return f"€{float(x):,.2f}".replace(",", ".")
    except: return "€0.00"

def fmt_pct2(x):
    try: return f"{float(x):.2f}%"
    except: return "0.00%"

def delta_txt(diff):
    if pd.isna(diff): 
        cls = "flat"; label = "n.v.t."
    else:
        cls = "up" if diff>0 else ("down" if diff<0 else "flat")
        sign = "+" if diff>0 else ""
        label = f"{sign}{fmt_eur0(diff)}"
    return f'<span class="kpi-delta {cls}">{label} vs vorige</span>'

# ───────── Inputs ─────────
st.title("🧩 Retail Portfolio Benchmark")

PERIODS = ["this_week","last_week","this_month","last_month","this_quarter","last_quarter","this_year","last_year"]
c0, c1, c2 = st.columns([1.2, 1, 1])
with c0:
    period = st.selectbox("Periode", PERIODS, index=1, key="p03_period")
with c1:
    regio_choice = st.selectbox("Regio", ["All"] + REGIONS, index=0, key="p03_region")
with c2:
    kpi_main = st.selectbox("KPI voor tabellen / kaarten", 
                            ["sales_per_sqm","conversion_rate","sales_per_visitor"], 
                            index=0, key="p03_main_kpi")

# Bepaal shops op basis van regio
if regio_choice == "All":
    ALL_IDS = list(ID_TO_NAME.keys())
else:
    r_ids = get_ids_by_region(regio_choice)
    ALL_IDS = r_ids if r_ids else list(ID_TO_NAME.keys())

if not ALL_IDS:
    st.warning("Geen winkels gevonden (mapping leeg).")
    st.stop()

# ───────── Data ophalen (dag) ─────────
df_day, p_day, s_day = fetch_df(ALL_IDS, period, "day", KPI_KEYS_BASE)

if df_day.empty:
    st.warning("Geen data voor deze periode/regio.")
    with st.expander("🔧 Debug"):
        st.write("Params (day):", p_day)
    st.stop()

# Alleen t/m gisteren voor this_*
TODAY = datetime.now(TZ).date()
if period.startswith("this_"):
    df_day = df_day[df_day["date_only"] < TODAY]

# ───────── Aggregatie / kaarten ─────────
# Baseline per winkel (gewogen gemiddelden voor conv & spv)
g_sum = df_day.groupby("shop_id", as_index=False).agg({"count_in":"sum","turnover":"sum"})
g_w = df_day.groupby("shop_id").apply(
    lambda x: pd.Series({
        "conversion_rate": weighted_avg(x["conversion_rate"], x["count_in"]),
        "sales_per_visitor": weighted_avg(x["sales_per_visitor"], x["count_in"]),
    })
).reset_index()
base = g_sum.merge(g_w, on="shop_id", how="left")
# sq_meter = laatste bekende
sqm = (
    df_day.sort_values("date_eff")
          .groupby("shop_id")["sq_meter"]
          .apply(lambda s: float(s.dropna().iloc[-1]) if s.dropna().size else np.nan)
          .reset_index()
)
base = base.merge(sqm, on="shop_id", how="left")
base["sales_per_sqm"] = base.apply(
    lambda r: (r["turnover"]/r["sq_meter"]) if (pd.notna(r["sq_meter"]) and r["sq_meter"]>0) else np.nan,
    axis=1
)
base["shop_name"] = base["shop_id"].map(ID_TO_NAME)

# Regio totalen + deltas (vs "vorige" periode als je een last_* of this_* kiest)
TOTAL_TURN = base["turnover"].sum()
TOTAL_VIS = base["count_in"].sum()
AVG_CONV = weighted_avg(base["conversion_rate"], base["count_in"])
AVG_SPV = weighted_avg(base["sales_per_visitor"], base["count_in"])
TOTAL_SQM = base["sq_meter"].fillna(0).sum()
AVG_SPSQM = (TOTAL_TURN / TOTAL_SQM) if TOTAL_SQM>0 else np.nan

def prev_period_of(p):
    mapping = {
        "this_week":"last_week", "this_month":"last_month",
        "this_quarter":"last_quarter", "this_year":"last_year",
        "last_week":"last_week", "last_month":"last_month",
        "last_quarter":"last_quarter", "last_year":"last_year"
    }
    return mapping.get(p)

prevP = prev_period_of(period)
if prevP:
    df_prev, _, _ = fetch_df(ALL_IDS, prevP, "day", KPI_KEYS_BASE)
    if prevP.startswith("this_"):
        df_prev = df_prev[df_prev["date_only"] < TODAY]

    if not df_prev.empty:
        g_sum_p = df_prev.groupby("shop_id", as_index=False).agg({"count_in":"sum","turnover":"sum"})
        g_w_p = df_prev.groupby("shop_id").apply(
            lambda x: pd.Series({
                "conversion_rate": weighted_avg(x["conversion_rate"], x["count_in"]),
                "sales_per_visitor": weighted_avg(x["sales_per_visitor"], x["count_in"]),
            })
        ).reset_index()
        base_p = g_sum_p.merge(g_w_p, on="shop_id", how="left")
        sqm_p = (
            df_prev.sort_values("date_eff")
                  .groupby("shop_id")["sq_meter"]
                  .apply(lambda s: float(s.dropna().iloc[-1]) if s.dropna().size else np.nan)
                  .reset_index()
        )
        base_p = base_p.merge(sqm_p, on="shop_id", how="left")
        base_p["sales_per_sqm"] = base_p.apply(
            lambda r: (r["turnover"]/r["sq_meter"]) if (pd.notna(r["sq_meter"]) and r["sq_meter"]>0) else np.nan,
            axis=1
        )

        TOTAL_TURN_P = base_p["turnover"].sum()
        AVG_SPSQM_P = (TOTAL_TURN_P / base_p["sq_meter"].fillna(0).sum()) if base_p["sq_meter"].fillna(0).sum()>0 else np.nan
        DIFF_TURN = TOTAL_TURN - TOTAL_TURN_P
        DIFF_SPSQM = (AVG_SPSQM - AVG_SPSQM_P) if (pd.notna(AVG_SPSQM) and pd.notna(AVG_SPSQM_P)) else np.nan
    else:
        DIFF_TURN = np.nan
        DIFF_SPSQM = np.nan
else:
    DIFF_TURN = np.nan
    DIFF_SPSQM = np.nan

# ───────── KPI-kaarten ─────────
k1,k2,k3,k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-title">💶 Totale omzet</div>
      <div class="kpi-value">{fmt_eur0(TOTAL_TURN)}</div>
      {delta_txt(DIFF_TURN)}
    </div>
    """, unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-title">🛒 Gem. conversie</div>
      <div class="kpi-value">{fmt_pct2(AVG_CONV)}</div>
      <span class="kpi-delta flat">gewogen</span>
    </div>
    """, unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-title">💸 Gem. SPV</div>
      <div class="kpi-value">{fmt_eur2(AVG_SPV)}</div>
      <span class="kpi-delta flat">gewogen</span>
    </div>
    """, unsafe_allow_html=True)
with k4:
    spsqm_txt = "n.v.t." if pd.isna(AVG_SPSQM) else fmt_eur2(AVG_SPSQM)
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-title">🏁 Gem. Sales/m²</div>
      <div class="kpi-value">{spsqm_txt}</div>
      {('<span class="kpi-delta %s">%s vs vorige</span>' %
         (("up" if DIFF_SPSQM>0 else ("down" if DIFF_SPSQM<0 else "flat")),
          f"{'+' if DIFF_SPSQM>0 else ''}{fmt_eur2(abs(DIFF_SPSQM))}")) if not pd.isna(DIFF_SPSQM) else '<span class="kpi-delta flat">n.v.t.</span>'}
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ───────── Leaderboard t.o.v. regio-gemiddelde ─────────
st.subheader("🏁 Leaderboard — t.o.v. regio-gemiddelde")
metric_map_lbl = {
    "sales_per_sqm":"Sales/m²",
    "conversion_rate":"Conversie",
    "sales_per_visitor":"SPV"
}
metric_label = metric_map_lbl[kpi_main]

comp = base[["shop_id","shop_name","count_in","turnover","sales_per_visitor","conversion_rate","sq_meter","sales_per_sqm"]].copy()
region_avg = comp[kpi_main].mean(skipna=True)
comp["delta"] = comp[kpi_main] - region_avg
sort_best_first = st.toggle("Beste afwijking eerst", value=True, key="p03_toggle_best")
comp = comp.sort_values("delta", ascending=not sort_best_first)

show = comp[["shop_name",kpi_main,"delta","turnover","count_in"]].rename(columns={
    "shop_name":"winkel", kpi_main: metric_label, "delta":"Δ vs gem.", "turnover":"omzet", "count_in":"bezoekers"
})
def color_delta(series):
    styles=[]
    for v in series:
        if pd.isna(v) or v==0: styles.append("color: #6B7280;")
        elif v>0: styles.append("color: #22C55E; font-weight:700;")
        else: styles.append("color: #F04438; font-weight:700;")
    return styles
styler = (
    show.style
        .format({metric_label: ("€{:.2f}" if kpi_main!="conversion_rate" else "{:.2f}%"),
                 "Δ vs gem.": ("€{:+.2f}" if kpi_main!="conversion_rate" else "{:+.2f}%"),
                 "omzet":"€{:.0f}", "bezoekers":"{:.0f}"})
        .apply(color_delta, subset=["Δ vs gem."])
)
st.dataframe(styler, use_container_width=True)

# ───────── UUR-HEATMAP (proto in dezelfde pagina) ─────────
with st.expander("🔥 Uur-heatmap (proto) — vergelijk max 30 winkels", expanded=False):
    cA, cB, cC = st.columns([1, 1, 1.2])
    with cA:
        kpi_heat = st.selectbox("KPI voor heatmap", 
                                ["sales_per_sqm","conversion_rate","sales_per_visitor"],
                                index=0, key="p03_heat_kpi")
    with cB:
        hour_min, hour_max = st.slider("Toon uren (openingstijden)", 0, 23, (10, 18), step=1, key="p03_hour_range")
    with cC:
        # standaard de “beste” 6 winkels op basis van omzet voor overzicht
        top_default = base.sort_values("turnover", ascending=False)["shop_name"].head(6).tolist()
        sel_names = st.multiselect("Vergelijk winkels", 
                                   base["shop_name"].tolist(),
                                   default=top_default, 
                                   max_selections=30,
                                   key="p03_heat_shops")
    if not sel_names:
        st.info("Kies ≥1 winkel om de heatmap te tonen.")
        st.stop()

    sel_ids = base[base["shop_name"].isin(sel_names)]["shop_id"].tolist()

    # Hourly pull (alleen voor de gekozen winkels om het lichter te houden)
    df_hr, p_hr, s_hr = fetch_df(sel_ids, period, "hour", KPI_KEYS_HOURLY)
    if period.startswith("this_"):
        df_hr = df_hr[df_hr["date_only"] < TODAY]

    if df_hr.empty:
        st.info("Geen uurdataset voor de gekozen instellingen.")
        st.stop()

    # fix hour kolom
    if "hour" not in df_hr.columns:
        df_hr["hour"] = pd.to_datetime(df_hr["date_eff"], errors="coerce").dt.hour

    # filter uren
    df_hr = df_hr[(df_hr["hour"] >= hour_min) & (df_hr["hour"] <= hour_max)]

    # voeg sq_meter per shop toe en bereken sales_per_sqm per uur
    sqm_map = base.set_index("shop_id")["sq_meter"]
    df_hr["sq_meter"] = df_hr["shop_id"].map(sqm_map)
    df_hr["sales_per_sqm"] = np.where(df_hr["sq_meter"]>0, df_hr["turnover"]/df_hr["sq_meter"], np.nan)

    # maak de te plotten waarde
    val_col = kpi_heat
    # normaliseer per winkel naar % van eigen piek: goed om "dead hours" te spotten
    def per_shop_pct(g: pd.DataFrame, col: str):
        v = g[col].astype(float)
        mx = v.max(skipna=True)
        return (v / mx * 100.0) if (mx and not pd.isna(mx) and mx>0) else np.zeros(len(g))
    df_hr["val_pct"] = df_hr.groupby("shop_id").apply(lambda g: per_shop_pct(g, val_col)).reset_index(level=0, drop=True)

    # alleen geselecteerde winkels en kolommen
    plot_df = df_hr[df_hr["shop_id"].isin(sel_ids)][["shop_id","shop_name","hour","val_pct"]].copy()
    # Sorteer shops op totale omzet (volgorde in heatmap)
    order_names = base.sort_values("turnover", ascending=False)["shop_name"].tolist()
    plot_df["shop_name"] = pd.Categorical(plot_df["shop_name"], categories=order_names, ordered=True)

    # categorievolgorde voor uren (beperk tot gekozen range)
    hour_ticks = list(range(hour_min, hour_max+1))
    plot_df["hour"] = pd.Categorical(plot_df["hour"], categories=hour_ticks, ordered=True)

    fig = px.imshow(
        plot_df.pivot_table(index="shop_name", columns="hour", values="val_pct", aggfunc="mean").sort_index(),
        color_continuous_scale="Magma",
        origin="lower",
        aspect="auto",
        labels=dict(color="%")
    )
    fig.update_layout(height=540, margin=dict(l=10,r=10,t=10,b=10), coloraxis_colorbar_title="%")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Tip: kleur = % van eigen piek in de gekozen periode. Donker = ‘dead hours’ voor die winkel. "
               "Schakel KPI om te kijken of het zit in traffic → conversie of in SPV/ATV.")

# ───────── Debug (optioneel) ─────────
with st.expander("🔧 Debug — API params"):
    st.write("Day params:", p_day[:10], "…")
    if prevP: st.write("Prev period:", prevP)