# pages/03_AI_Advisor_Weather_CBS.py
import os
import sys
import pandas as pd
import streamlit as st

# ── Page setup
st.set_page_config(page_title="AI Advisor — Weer + CBS", page_icon="🧭", layout="wide")
st.title("🧭 AI Advisor — Weer + CBS (v1)")

# ── Import helpers from project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from helpers_normalize import normalize_vemcount_response
from helpers_shop import ID_TO_NAME, get_ids_by_region, REGIONS

# ── Services (package 'services' moet een __init__.py hebben)
from services.weather_service import get_daily_forecast
from services.cbs_service import (
    get_consumer_confidence,
    get_cci_series,
    get_retail_index,
    list_retail_branches,
)
from services.advisor import build_advice

# ── Secrets (platte keys zoals jouw Streamlit Settings → Secrets)
def _get_secret(key: str, env_fallback: str = "") -> str:
    val = st.secrets.get(key) or os.getenv(env_fallback or key.upper()) or ""
    return (val or "").strip()

OPENWEATHER_KEY = _get_secret("openweather_api_key", "OPENWEATHER_API_KEY")
CBS_DATASET     = _get_secret("cbs_dataset", "CBS_DATASET")
API_URL         = _get_secret("API_URL").rstrip("/")

# Controle op vereiste secrets
missing = []
if not OPENWEATHER_KEY: missing.append("openweather_api_key")
if not CBS_DATASET:     missing.append("cbs_dataset")
if not API_URL:         missing.append("API_URL")
if missing:
    st.error("Missing secrets: " + ", ".join(missing) + "\n\nCheck Streamlit → Settings → Secrets.")
    st.stop()

# ── Controls
region = st.selectbox("Regio", options=["ALL"] + list(REGIONS), index=0)
lat = st.number_input("Latitude", value=52.37)
lon = st.number_input("Longitude", value=4.90)
days_ahead = st.slider("Dagen vooruit", 1, 7, 5)
period_hist = st.selectbox("Historische periode", ["last_month", "this_year", "last_year"], index=0)

st.subheader("Macro-context (CBS)")
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    months_back = st.slider("Maanden terug (CBS)", 6, 36, 18)
with col2:
    use_retail = st.checkbox("Toon detailhandel-index (85828NED)", value=True)
with col3:
    dim_name, branch_items = list_retail_branches("85828NED")
    # Bouw mapping: Title -> Key (fallbacks voor als API niets teruggeeft)
    if branch_items:
        title_to_key = {b["title"]: str(b["key"]) for b in branch_items}
        titles = list(title_to_key.keys())
        # probeer "Totaal detailhandel" als default; anders eerste item
        default_idx = 0
        for i, t in enumerate(titles):
            if "totaal" in t.lower():
                default_idx = i
                break
        branch_title = st.selectbox("Branche (CBS)", titles, index=default_idx)
        branch_key = title_to_key[branch_title]
    else:
        # Fallback zonder dimensielijst
        branch_title = st.selectbox("Branche (CBS)", ["DH_TOTAAL","DH_FOOD","DH_NONFOOD"], index=0)
        branch_key = branch_title  # hier gebruiken we de code zelf als 'key'

# Ophalen macro-reeksen (buiten de knop zodat tiles meteen renderen)
try:
    cci_series = get_cci_series(months_back=months_back, dataset=CBS_DATASET)
except Exception as e:
    cci_series = []
    st.info(f"CCI niet beschikbaar: {e}")

try:
    retail_series = []
    if use_retail:
        # 1) Probeer met de KEY uit de dropdown
        retail_series = get_retail_index(branch_code_or_title=branch_key, months_back=months_back)
        # 2) Als leeg, probeer met de TITLE (sommige datasets matchen daarop)
        if not retail_series:
            retail_series = get_retail_index(branch_code_or_title=branch_title, months_back=months_back)
        # 3) Als nog leeg, probeer totale branche als harde fallback
        if not retail_series:
            retail_series = get_retail_index(branch_code_or_title="DH_TOTAAL", months_back=months_back)
except Exception as e:
    retail_series = []
    if use_retail:
        st.info(f"Detailhandelreeks (85828NED) niet beschikbaar: {e}")

# ── Fetch historical KPIs (je bestaande FastAPI/Vemcount-laag)
def fetch_hist_kpis(shop_ids, period: str):
    params = [("source", "shops")]
    for sid in shop_ids:
        params.append(("data", int(sid)))
    for k in ["count_in", "conversion_rate", "turnover", "sales_per_visitor"]:
        params.append(("data_output", k))
    params.append(("period", period))
    params.append(("period_step", "day"))
    import requests
    r = requests.get(API_URL, params=params, timeout=45)
    r.raise_for_status()
    return r.json()

# ── Baselines per weekdag (per winkel) voor forecast
def build_weekday_baselines(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    d = df.copy()
    d["date_eff"] = pd.to_datetime(d["date"], errors="coerce")
    d["weekday"] = d["date_eff"].dt.weekday  # 0=Mon
    out = {}
    for wd, g in d.groupby("weekday"):
        stores = {}
        for sid, gs in g.groupby("shop_id"):
            name = ID_TO_NAME.get(int(sid), f"Shop {sid}")
            if "sales_per_visitor" in gs.columns:
                spv_series = gs["sales_per_visitor"]
            else:
                denom = gs["count_in"].replace(0, pd.NA)
                spv_series = (gs["turnover"] / denom).fillna(0)
            stores[name] = {
                "visitors": float(gs["count_in"].mean()),
                "conversion": float(gs["conversion_rate"].mean()),
                "spv": float(spv_series.mean()),
                "spv_median": float(spv_series.median()),
                "visitors_p30": float(gs["count_in"].quantile(0.30)),
            }
        out[wd] = stores
    return out

# ==== ANALYTICS HELPERS =======================================================
def _eur(x):
    try: return "€{:,.0f}".format(x).replace(",", ".")
    except: return "–"

def _fmt_pct(x):
    return "—" if (x is None or pd.isna(x)) else f"{x:+.1f}%"

def monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date_eff"] = pd.to_datetime(d["date"], errors="coerce")
    d["ym"] = d["date_eff"].dt.to_period("M").astype(str)
    g = d.groupby(["ym"], as_index=False).agg(
        visitors=("count_in", "sum"),
        turnover=("turnover", "sum"),
    )
    d["weighted_conv"] = d["conversion_rate"] * d["count_in"]
    conv = d.groupby("ym", as_index=False).agg(conv_num=("weighted_conv","sum"), conv_den=("count_in","sum"))
    conv["conversion"] = (conv["conv_num"]/conv["conv_den"]).fillna(0)
    out = g.merge(conv[["ym","conversion"]], on="ym", how="left")
    out["spv"] = (out["turnover"] / out["visitors"]).replace([float("inf")], 0).fillna(0)
    return out

def mom_yoy(dfm: pd.DataFrame):
    if dfm is None or dfm.empty:
        return {}
    m = dfm.copy().sort_values("ym").reset_index(drop=True)
    m["ym_dt"] = pd.to_datetime(m["ym"].astype(str) + "-01", errors="coerce")
    if m["ym_dt"].isna().all(): return {}
    last = m.iloc[-1]; prev = m.iloc[-2] if len(m) > 1 else None
    yoy_dt = last["ym_dt"] - pd.DateOffset(years=1)
    yoy_row = m.loc[m["ym_dt"] == yoy_dt]
    yoy_row = yoy_row.iloc[0] if not yoy_row.empty else None
    def pct(a,b):
        if b in [0,None] or pd.isna(b): return None
        try: return (float(a)/float(b) - 1) * 100
        except: return None
    return {
        "last_ym": last["ym_dt"].strftime("%Y-%m"),
        "visitors": float(last.get("visitors", 0)),
        "turnover": float(last.get("turnover", 0)),
        "conversion": float(last.get("conversion", 0)),
        "spv": float(last.get("spv", 0)),
        "mom": {
            "turnover": pct(last.get("turnover", 0), prev.get("turnover", 0)) if prev is not None else None,
            "visitors": pct(last.get("visitors", 0), prev.get("visitors", 0)) if prev is not None else None,
            "conversion": pct(last.get("conversion", 0), prev.get("conversion", 0)) if prev is not None else None,
            "spv": pct(last.get("spv", 0), prev.get("spv", 0)) if prev is not None else None,
        },
        "yoy": {
            "turnover": pct(last.get("turnover", 0), yoy_row.get("turnover", 0)) if yoy_row is not None else None,
            "visitors": pct(last.get("visitors", 0), yoy_row.get("visitors", 0)) if yoy_row is not None else None,
            "conversion": pct(last.get("conversion", 0), yoy_row.get("conversion", 0)) if yoy_row is not None else None,
            "spv": pct(last.get("spv", 0), yoy_row.get("spv", 0)) if yoy_row is not None else None,
        },
    }

def estimate_weather_uplift(baseline_day: dict, forecast_days: list[dict]) -> dict:
    """
    Per dag:
    - base = bezoekers/spv uit baseline (op weekday), zonder weer-correctie.
    - adj  = base * (1 - 0.20*POP) * (1 + 0.01*(temp-15))
    Retourneert daily + total + delta t.o.v. base.
    """
    daily = []
    for f in forecast_days:
        wd = pd.to_datetime(f["date"]).weekday()
        base = baseline_day.get(wd, {"visitors": 0.0, "spv": 0.0})
        base_vis = float(base["visitors"]); spv = float(base["spv"])
        pop  = float(f.get("pop", 0.0))
        temp = float(f.get("temp", 15.0))
        adj_vis = base_vis * (1 - 0.20*pop) * (1 + 0.01*(temp-15.0))
        row = {
            "date": f["date"],
            "weekday": wd,
            "temp": temp,
            "pop": pop,
            "base_visitors": base_vis,
            "adj_visitors": adj_vis,
            "spv": spv,
            "base_turnover": base_vis * spv,
            "adj_turnover": adj_vis * spv
        }
        row["delta_turnover"] = row["adj_turnover"] - row["base_turnover"]
        daily.append(row)
    base_total = sum(x["base_turnover"] for x in daily)
    adj_total  = sum(x["adj_turnover"]  for x in daily)
    delta_total = adj_total - base_total
    delta_pct = (delta_total/base_total*100) if base_total else None
    return {"daily": daily, "base_total": base_total, "adj_total": adj_total, "delta_total": delta_total, "delta_pct": delta_pct}

# ── Action
st.caption("Selecteer regio en druk op de knop om aanbevelingen te genereren.")
shop_ids = get_ids_by_region(region)
st.write(f"{len(shop_ids)} winkels geselecteerd in regio: **{region}**")

if st.button("Genereer aanbevelingen"):
    # 1) Historische data
    js = fetch_hist_kpis(shop_ids, period_hist)
    df = normalize_vemcount_response(
        js, ID_TO_NAME,
        kpi_keys=["count_in", "conversion_rate", "turnover", "sales_per_visitor"]
    )

    # 2) Baselines + regiotrend
    baseline = build_weekday_baselines(df)
    dfm = monthly_agg(df)
    trend = mom_yoy(dfm)

    # 3) Weer + CBS
    forecast = get_daily_forecast(lat, lon, OPENWEATHER_KEY, days_ahead)
    try:
        cci_info = get_consumer_confidence(CBS_DATASET)
        cci = cci_info["value"]; cci_period = cci_info["period"]
    except Exception as e:
        st.warning(f"Kon CBS Consumentenvertrouwen niet ophalen (gebruik standaard 0). Details: {e}")
        cci, cci_period = 0.0, "n/a"

    # 4) Adviesregels (dag/winkel)
    advice = build_advice("Your Company", baseline, forecast, cci)

    # 5) Regionale baseline (gemiddelde over winkels per weekday)
    baseline_day = {}
    for wd, storemap in baseline.items():
        if not storemap: continue
        visitors = pd.Series([v["visitors"] for v in storemap.values()]).mean()
        spv = pd.Series([v["spv"] for v in storemap.values()]).mean()
        baseline_day[wd] = {"visitors": float(visitors), "spv": float(spv)}

    # 6) 7-daagse verwachting o.b.v. weer (met base vs adjusted)
    wx = estimate_weather_uplift(baseline_day, forecast)

    # ── Silver-platter samenvatting (regio)
    st.subheader("🔎 Silver-platter samenvatting (regio)")
    if trend:
        colA, colB, colC, colD = st.columns(4)
        colA.metric(
            f"Omzet {trend.get('last_ym','-')}",
            _eur(trend.get('turnover', 0)),
            _fmt_pct(trend['mom'].get('turnover'))
        )
        colB.metric(
            "Bezoekers", f"{trend.get('visitors', 0):,.0f}".replace(",", "."),
            _fmt_pct(trend['mom'].get('visitors'))
        )
        colC.metric(
            "Conversie", f"{trend.get('conversion', 0)*100:.2f}%",
            _fmt_pct(trend['mom'].get('conversion'))
        )
        colD.metric(
            "SPV", f"€{trend.get('spv', 0):.2f}",
            _fmt_pct(trend['mom'].get('spv'))
        )
        st.caption(
            "YoY: "
            + (f"Omzet {trend['yoy']['turnover']:.1f}%  " if trend['yoy']['turnover'] is not None else "")
            + (f"• Bezoekers {trend['yoy']['visitors']:.1f}%  " if trend['yoy']['visitors'] is not None else "")
            + (f"• Conversie {trend['yoy']['conversion']:.1f}%  " if trend['yoy']['conversion'] is not None else "")
            + (f"• SPV {trend['yoy']['spv']:.1f}%" if trend['yoy']['spv'] is not None else "")
        )

    # ── Macro duiding (CCI)
    st.metric("Consumentenvertrouwen (CBS)", f"{cci}", help=f"Periode: {cci_period}")
    cci_text = "positief (kooplustiger)" if cci >= 0 else "negatief (voorzichtiger)"
    st.info(
        f"**CCI ({cci_period}) = {cci:.1f}** → sentiment {cci_text}. "
        "Bij positief sentiment: mik op hogere SPV (premium/upsell). Bij negatief: bundels/waarde benadrukken."
    )

    # ── Komende dagen: verwachting & acties
    st.subheader("📅 Komende dagen — verwachting & acties")
    for d in advice["days"]:
        with st.expander(f'📆 {d["date"]} — temp {d["weather"]["temp"]:.1f}°C • POP {int(d["weather"]["pop"]*100)}%'):
            for s in d["stores"]:
                st.markdown(f"**{s['store']}**")
                st.write("— Storemanager:", " • ".join(s["store_actions"]))
                st.write("— Regiomanager:", " • ".join(s["regional_actions"]))

    # ── NIEUW: Weerimpact per dag (base vs adjusted)
    st.subheader("🌦️ Weerimpact per dag (t.o.v. normale weekdag)")
    if wx and wx["daily"]:
        wx_df = pd.DataFrame(wx["daily"]).copy()
        wx_df["date"] = pd.to_datetime(wx_df["date"])
        wx_df = wx_df.sort_values("date")
        wx_df["impact_pct"] = (wx_df["delta_turnover"] / wx_df["base_turnover"] * 100).replace([pd.NA, pd.NaT], 0)
        # Bar chart: extra/minder omzet door weer
        st.bar_chart(wx_df.set_index("date")["delta_turnover"].rename("Δ omzet vs baseline"))
        # Tabel met context
        show_cols = ["date","temp","pop","base_turnover","adj_turnover","delta_turnover","impact_pct"]
        nice = wx_df[show_cols].rename(columns={
            "date":"Datum","temp":"Temp (°C)","pop":"Regenkans",
            "base_turnover":"Baseline omzet","adj_turnover":"Verwachte omzet",
            "delta_turnover":"Δ Omzet","impact_pct":"Δ %"
        })
        nice["Regenkans"] = (nice["Regenkans"]*100).round(0).astype(int).astype(str) + "%"
        nice["Baseline omzet"] = nice["Baseline omzet"].map(_eur)
        nice["Verwachte omzet"] = nice["Verwachte omzet"].map(_eur)
        nice["Δ Omzet"] = nice["Δ Omzet"].map(_eur)
        nice["Δ %"] = nice["Δ %"].apply(lambda v: "—" if pd.isna(v) else f"{v:+.1f}%")
        st.dataframe(nice, use_container_width=True)

        # Tile samenvatting
        delta_pct_txt = "—" if (wx["delta_pct"] is None) else f"{wx['delta_pct']:+.1f}%"
        st.success(
            f"🔮 **7-daagse verwachting**: { _eur(wx['adj_total']) } "
            f"({delta_pct_txt} t.o.v. normale week). Heuristische weer-correctie toegepast.",
            icon="🔮"
        )
    else:
        st.info("Geen forecast-gegevens beschikbaar.")

    # ── Visuals (compact)
    st.subheader("📊 Regiotrend (maandelijks)")
    if not dfm.empty:
        left, right = st.columns(2)
        left.line_chart(dfm.set_index("ym")[["turnover"]].rename(columns={"turnover": "Omzet"}))
        right.line_chart(dfm.set_index("ym")[["visitors"]].rename(columns={"visitors": "Bezoekers"}))

    # ── Macro tiles (context)
    if cci_series:
        st.metric("CCI (laatste maand)", f"{cci_series[-1]['cci']:.1f}")
        with st.expander("📈 CCI reeks (CBS)"):
            st.line_chart({"CCI": [x["cci"] for x in cci_series]})

    if retail_series:
        last_r = retail_series[-1]
        st.metric(f"Detailhandel ({last_r['branch']}) — {last_r['series']}", f"{last_r['retail_value']:.1f}")
        with st.expander("🛍️ Detailhandel reeks (CBS 85828NED)"):
            st.line_chart({"Retail": [x["retail_value"] for x in retail_series]})
    elif use_retail:
        st.info("Geen detailhandelreeks gevonden voor deze branche/periode (85828NED). Probeer 'DH_TOTAAL', 'DH_FOOD' of 'DH_NONFOOD'.")
