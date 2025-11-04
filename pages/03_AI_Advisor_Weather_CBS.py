# pages/03_AI_Advisor_Weather_CBS.py
import os, sys
from datetime import date
import pandas as pd
import streamlit as st

# ── Page setup
st.set_page_config(page_title="AI Advisor — Weer + CBS", page_icon="🧭", layout="wide")
st.title("🧭 AI Advisor — Weer + CBS (v1)")

# ── Import helpers from project root
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))
from helpers_normalize import normalize_vemcount_response
from helpers_shop import ID_TO_NAME, get_ids_by_region, REGIONS
from advisor import build_advice

# import services uit de submap 'services'
from services.weather_service import get_daily_forecast
from services.cbs_service import get_consumer_confidence, get_cci_series, get_retail_index  # nieuwe imports

# ── Secrets (werkt met platte keys zoals in jouw screenshot)
import streamlit as st, os

def _get_secret(key: str, env_fallback: str = "") -> str:
    """Zoekt secret key op in st.secrets of omgeving"""
    val = st.secrets.get(key) or os.getenv(env_fallback or key.upper()) or ""
    return val.strip()

OPENWEATHER_KEY = _get_secret("openweather_api_key", "OPENWEATHER_API_KEY")
CBS_DATASET     = _get_secret("cbs_dataset", "CBS_DATASET")
API_URL         = _get_secret("API_URL")

# controle
if not OPENWEATHER_KEY or not CBS_DATASET or not API_URL:
    st.error(f"Missing secrets: "
             f"{'openweather_api_key ' if not OPENWEATHER_KEY else ''}"
             f"{'cbs_dataset ' if not CBS_DATASET else ''}"
             f"{'API_URL' if not API_URL else ''}\n\n"
             "Check Streamlit > Settings > Secrets.")
    st.stop()

# ── Controls
region = st.selectbox("Regio", options=["ALL"] + list(REGIONS), index=0)
lat = st.number_input("Latitude", value=52.37)
lon = st.number_input("Longitude", value=4.90)
days_ahead = st.slider("Dagen vooruit", 1, 7, 5)
period_hist = st.selectbox("Historische periode", ["last_month","this_year","last_year"], index=0)
st.subheader("Macro-context (CBS)")
col1, col2, col3 = st.columns([1,1,2])
with col1:
    months_back = st.slider("Maanden terug (CBS)", 6, 36, 18)
with col2:
    use_retail = st.checkbox("Toon detailhandel-index (85828NED)", value=True)
with col3:
    branch_code = st.text_input("Branchecode (CBS)", value="DH_TOTAAL")  # later dropdown maken

# Ophalen macro-reeksen
cci_series = get_cci_series(months_back=months_back, dataset=CBS_DATASET)
retail_series = get_retail_index(branch_code=branch_code, months_back=months_back) if use_retail else []

# ── Fetch historical KPIs from your existing API (no changes server-side)
def fetch_hist_kpis(shop_ids, period: str):
    params = [("source","shops")]
    for sid in shop_ids:
        params.append(("data", int(sid)))
    for k in ["count_in","conversion_rate","turnover","sales_per_visitor"]:
        params.append(("data_output", k))
    params.append(("period", period))
    params.append(("period_step", "day"))
    import requests
    r = requests.get(f"{API_URL}", params=params, timeout=45)
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
            visitors = float(gs["count_in"].mean())
            conv = float(gs["conversion_rate"].mean())
            spv = float(gs.get("sales_per_visitor", gs["turnover"]/gs["count_in"].replace(0, pd.NA)).fillna(0).mean())
            spv_median = float(gs["sales_per_visitor"].median())
            visitors_p30 = float(gs["count_in"].quantile(0.30))
            stores[name] = {
                "visitors": visitors,
                "conversion": conv,
                "spv": spv,
                "spv_median": spv_median,
                "visitors_p30": visitors_p30,
                "temps": []  # TODO v2: historische weerdata (timemachine) koppelen
            }
        out[wd] = stores
    return out

# ── Action
st.caption("Selecteer regio en druk op de knop om aanbevelingen te genereren.")
shop_ids = get_ids_by_region(region)
st.write(f"{len(shop_ids)} winkels geselecteerd in regio: **{region}**")

if st.button("Genereer aanbevelingen"):
    # 1) historische data
    js = fetch_hist_kpis(shop_ids, period_hist)
    df = normalize_vemcount_response(js, ID_TO_NAME, kpi_keys=["count_in","conversion_rate","turnover","sales_per_visitor"])
    # 2) baselines per weekdag
    baseline = build_weekday_baselines(df)
    # 3) weer + cbs
    forecast = get_daily_forecast(lat, lon, OPENWEATHER_KEY, days_ahead)
    try:
        cci_info = get_consumer_confidence(CBS_DATASET)
        cci = cci_info["value"]
        cci_period = cci_info["period"]
    except Exception as e:
    st.warning(f"Kon CBS Consumentenvertrouwen niet ophalen (gebruik standaard 0). Details: {e}")
        cci, cci_period = 0.0, "n/a"

    # 4) advies
    advice = build_advice("Your Company", baseline, forecast, cci)

    st.metric("Consumentenvertrouwen (CBS)", f"{cci}", help=f"Periode: {cci_period}")
    for d in advice["days"]:
        with st.expander(f'📅 {d["date"]} — temp {d["weather"]["temp"]:.1f}°C • neerslagkans {int(d["weather"]["pop"]*100)}%'):
            for s in d["stores"]:
                st.markdown(f"**{s['store']}**")
                st.write("— Storemanager:", " • ".join(s["store_actions"]))
                st.write("— Regiomanager:", " • ".join(s["regional_actions"]))

    # Macro tiles
    if cci_series:
        st.metric("CCI (laatste maand)", f"{cci_series[-1]['cci']:.1f}")
        with st.expander("📈 CCI reeks (CBS)"):
            st.line_chart({ "CCI": [x["cci"] for x in cci_series] })

    if retail_series:
        last = retail_series[-1]
        st.metric(f"Detailhandel ({last['branch']}) — {last['series']}", f"{last['retail_value']:.1f}")
        with st.expander("🛍️ Detailhandel reeks (CBS 85828NED)"):
            st.line_chart({ "Retail": [x["retail_value"] for x in retail_series] })

