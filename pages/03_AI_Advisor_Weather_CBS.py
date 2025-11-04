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

# ── Secrets (werkt met platte keys zoals in jouw screenshot)
def _get_secret(key: str, env_fallback: str = "") -> str:
    """Zoekt secret key op in st.secrets of omgeving."""
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
    branch_options = [b["title"] for b in branch_items] if branch_items else ["DH_TOTAAL", "DH_FOOD", "DH_NONFOOD"]
    branch_label = st.selectbox("Branche (CBS)", branch_options, index=0)

# Ophalen macro-reeksen (buiten de knop zodat tiles altijd beschikbaar zijn)
try:
    cci_series = get_cci_series(months_back=months_back, dataset=CBS_DATASET)
except Exception as e:
    cci_series = []
    st.info(f"CCI niet beschikbaar: {e}")

try:
    retail_series = get_retail_index(branch_code_or_title=branch_label, months_back=months_back) if use_retail else []
except Exception as e:
    retail_series = []
    if use_retail:
        st.info(f"Detailhandelreeks (85828NED) niet beschikbaar: {e}")

# ── Fetch historical KPIs from your existing API (no changes server-side)
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

# ── Build weekday baselines per store (simple & robust)
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
            # Robuuste SPV: kolom gebruiken als die bestaat, anders turnover/count_in
            if "sales_per_visitor" in gs.columns:
                spv_series = gs["sales_per_visitor"]
            else:
                denom = gs["count_in"].replace(0, pd.NA)
                spv_series = (gs["turnover"] / denom).fillna(0)

            visitors = float(gs["count_in"].mean())
            conv = float(gs["conversion_rate"].mean())
            spv = float(spv_series.mean())
            spv_median = float(spv_series.median())
            visitors_p30 = float(gs["count_in"].quantile(0.30))

            stores[name] = {
                "visitors": visitors,
                "conversion": conv,
                "spv": spv,
                "spv_median": spv_median,
                "visitors_p30": visitors_p30,
                "temps": []  # TODO v2: historische weerdata koppelen (timemachine)
            }
        out[wd] = stores
    return out

# ==== ANALYTICS HELPERS =======================================================

def _pct(x):
    try:
        return f"{x*100:,.1f}%".replace(",", ".")
    except Exception:
        return "–"

def _eur(x):
    try:
        return "€{:,.0f}".format(x).replace(",", ".")
    except Exception:
        return "–"

def monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Maandtotalen (regio): sum visitors/turnover, gewogen conversie, mean SPV."""
    d = df.copy()
    d["date_eff"] = pd.to_datetime(d["date"], errors="coerce")
    d["ym"] = d["date_eff"].dt.to_period("M").astype(str)

    g = d.groupby(["ym"], as_index=False).agg(
        visitors=("count_in", "sum"),
        turnover=("turnover", "sum"),
    )

    d["weighted_conv"] = d["conversion_rate"] * d["count_in"]
    conv = d.groupby("ym", as_index=False).agg(conv_num=("weighted_conv", "sum"), conv_den=("count_in", "sum"))
    conv["conversion"] = (conv["conv_num"] / conv["conv_den"]).fillna(0)

    out = g.merge(conv[["ym", "conversion"]], on="ym", how="left")
    out["spv"] = (out["turnover"] / out["visitors"]).replace([float("inf")], 0).fillna(0)
    return out

def mom_yoy(dfm: pd.DataFrame):
    """
    Robuuste MoM/YoY op regiomaand-aggregaten.
    Werkt ook als ym geen string is of er maar 1 maand aanwezig is.
    """
    if dfm is None or dfm.empty:
        return {}

    m = dfm.copy().sort_values("ym").reset_index(drop=True)

    # Forceer ym -> datetime (1e dag vd maand). Als dit faalt, stop netjes.
    try:
        m["ym_dt"] = pd.to_datetime(m["ym"].astype(str) + "-01", errors="coerce")
    except Exception:
        m["ym_dt"] = pd.NaT

    if m["ym_dt"].isna().all():
        return {}

    last = m.iloc[-1]
    prev = m.iloc[-2] if len(m) > 1 else None

    # Zoek dezelfde maand vorig jaar
    yoy_dt = last["ym_dt"] - pd.DateOffset(years=1)
    yoy_match = m.loc[m["ym_dt"] == yoy_dt]
    yoy_row = yoy_match.iloc[0] if not yoy_match.empty else None

    def pct(a, b):
        if b in [0, None] or pd.isna(b):
            return None
        try:
            return (float(a) / float(b) - 1) * 100
        except Exception:
            return None

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

def weather_effect(df: pd.DataFrame, hist_weather: pd.DataFrame | None) -> dict:
    """Simpele correlatie tussen dagelijks weer en bezoekers (optioneel, v2 voor echte timemachine)."""
    if df is None or df.empty or hist_weather is None or hist_weather.empty:
        return {"corr_temp_visitors": None, "corr_pop_visitors": None}
    d = df.copy()
    d["date_eff"] = pd.to_datetime(d["date"])
    m = d.groupby(d["date_eff"].dt.date).agg(visitors=("count_in", "sum")).reset_index()
    hw = hist_weather.copy()
    hw["date"] = pd.to_datetime(hw["date"]).dt.date
    j = m.merge(hw, on="date", how="inner")
    if j.empty:
        return {"corr_temp_visitors": None, "corr_pop_visitors": None}
    return {
        "corr_temp_visitors": float(j["temp"].corr(j["visitors"])),
        "corr_pop_visitors": float(j["pop"].corr(j["visitors"])),
    }

def forecast_next_week(baseline_day: dict, forecast_days: list[dict]) -> dict:
    """Ruwe schatting bezoekers & omzet komende dagen vanuit baseline per weekdag + weerfactoren."""
    out = []
    for f in forecast_days:
        wd = pd.to_datetime(f["date"]).weekday()
        b = baseline_day.get(wd, {"visitors": 0, "spv": 0})
        visitors = b["visitors"]
        visitors *= (1 - 0.20 * f.get("pop", 0))     # -20% bij 100% regen
        temp = f.get("temp", 0)
        visitors *= (1 + 0.01 * (temp - 15))        # +/-1% per °C tov 15
        spv = b["spv"]
        turnover = visitors * spv
        out.append(
            {
                "date": f["date"],
                "visitors": round(visitors),
                "spv": spv,
                "turnover": round(turnover, 2),
                "pop": f.get("pop", 0),
                "temp": temp,
            }
        )
    return {"daily": out, "sum_turnover": round(sum(x["turnover"] for x in out), 2)}

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

    # 2) Baselines per weekdag (voor forecast-schatting) + maandaggregatie voor trend
    baseline = build_weekday_baselines(df)
    dfm = monthly_agg(df)
    trend = mom_yoy(dfm)

    # 3) Weer + CBS
    forecast = get_daily_forecast(lat, lon, OPENWEATHER_KEY, days_ahead)
    wfx = weather_effect(df, hist_weather=None)  # hist weather nog niet gekoppeld

    try:
        cci_info = get_consumer_confidence(CBS_DATASET)
        cci = cci_info["value"]
        cci_period = cci_info["period"]
    except Exception as e:
        st.warning(f"Kon CBS Consumentenvertrouwen niet ophalen (gebruik standaard 0). Details: {e}")
        cci, cci_period = 0.0, "n/a"

    # 4) Adviesregels op dagniveau
    advice = build_advice("Your Company", baseline, forecast, cci)

    # 5) Forecast op weekniveau (regio-breed)
    baseline_day = {}
    for wd, storemap in baseline.items():
        if not storemap:
            continue
        visitors = pd.Series([v["visitors"] for v in storemap.values()]).mean()
        spv = pd.Series([v["spv"] for v in storemap.values()]).mean()
        baseline_day[wd] = {"visitors": float(visitors), "spv": float(spv)}
    week_forecast = forecast_next_week(baseline_day, forecast)

    # ── Silver-platter regio
    st.subheader("🔎 Silver-platter samenvatting (regio)")
    if trend:
        colA, colB, colC, colD = st.columns(4)
        colA.metric(
            f"Omzet {trend.get('last_ym','-')}",
            _eur(trend.get('turnover', 0)),
            (f"{trend['mom']['turnover']:.1f}% m/m" if trend['mom']['turnover'] is not None else "—"),
        )
        colB.metric(
            "Bezoekers",
            f"{trend.get('visitors', 0):,.0f}".replace(",", "."),
            (f"{trend['mom']['visitors']:.1f}% m/m" if trend['mom']['visitors'] is not None else "—"),
        )
        colC.metric(
            "Conversie",
            f"{trend.get('conversion', 0)*100:.2f}%",
            (f"{trend['mom']['conversion']:.1f}% m/m" if trend['mom']['conversion'] is not None else "—"),
        )
        colD.metric(
            "SPV",
            f"€{trend.get('spv', 0):.2f}",
            (f"{trend['mom']['spv']:.1f}% m/m" if trend['mom']['spv'] is not None else "—"),
        )

        st.caption(
            "YoY: "
            + (f"Omzet {trend['yoy']['turnover']:.1f}%  " if trend['yoy']['turnover'] is not None else "")
            + (f"• Bezoekers {trend['yoy']['visitors']:.1f}%  " if trend['yoy']['visitors'] is not None else "")
            + (f"• Conversie {trend['yoy']['conversion']:.1f}%  " if trend['yoy']['conversion'] is not None else "")
            + (f"• SPV {trend['yoy']['spv']:.1f}%" if trend['yoy']['spv'] is not None else "")
        )

    # Uitleg + macro duiding
    st.metric("Consumentenvertrouwen (CBS)", f"{cci}", help=f"Periode: {cci_period}")
    cci_text = "positief (kooplustiger)" if cci >= 0 else "negatief (voorzichtiger)"
    if abs(cci) > 100:
        st.warning("Let op: CCI lijkt buiten de gebruikelijke bandbreedte (verwacht ~-60..+40). Controleer dataset/veldselectie.")
    st.info(
        f"**CCI ({cci_period}) = {cci:.1f}** → sentiment {cci_text}. "
        "Gebruik hogere/betere SPV-doelen bij positief sentiment; focus op bundel/waarde bij negatief."
    )

    # Dagverwachting + aanbevelingen (weer)
    st.subheader("📅 Komende dagen — verwachting & acties")
    for d in advice["days"]:
        with st.expander(f'📆 {d["date"]} — temp {d["weather"]["temp"]:.1f}°C • POP {int(d["weather"]["pop"]*100)}%'):
            for s in d["stores"]:
                st.markdown(f"**{s['store']}**")
                st.write("— Storemanager:", " • ".join(s["store_actions"]))
                st.write("— Regiomanager:", " • ".join(s["regional_actions"]))

    # Verwachting 7 dagen (regio)
    corr_t = wfx["corr_temp_visitors"]; corr_p = wfx["corr_pop_visitors"]
    weer_line = []
    if corr_t is not None: weer_line.append(f"temp↔bezoekers corr = {corr_t:.2f}")
    if corr_p is not None: weer_line.append(f"regen↔bezoekers corr = {corr_p:.2f}")
    weer_line = " | ".join(weer_line) if weer_line else "historische weerkoppeling nog niet geactiveerd"
    st.success(
        "🔮 **Verwachting komende 7 dagen**\n"
        f"• Geschatte omzet: **{_eur(week_forecast['sum_turnover'])}**\n"
        f"• {weer_line}\n"
        "• Tip: plan inzet rond dagen met lage regen-kans en zet queue-buster in op nattere piekuren."
    )

    # Visuals (compact)
    st.subheader("📊 Regiotrend (maandelijks)")
    if not dfm.empty:
        left, right = st.columns(2)
        left.line_chart(dfm.set_index("ym")[["turnover"]].rename(columns={"turnover": "Omzet"}))
        right.line_chart(dfm.set_index("ym")[["visitors"]].rename(columns={"visitors": "Bezoekers"}))

    # Macro tiles (context)
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
        st.info("Geen detailhandelreeks gevonden voor deze branche-code en periode (85828NED). Probeer 'DH_TOTAAL', 'DH_FOOD' of 'DH_NONFOOD'.")
