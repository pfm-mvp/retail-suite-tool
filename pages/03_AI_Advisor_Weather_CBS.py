import streamlit as st
from services.weather import get_daily_forecast
from services.cbs import get_consumer_confidence
from services.advisor import build_advice
from utils_pfmx import get_config  # of je eigen config loader
# importeer je bestaande helpers om hist baselines per weekdag te maken
from helpers_normalize import make_hist_by_weekday  # voorbeeldnaam

st.set_page_config(page_title="AI Advisor (Weer + CBS)", layout="wide")

cfg = st.secrets.get("external", {}) or {}
lat = st.number_input("Latitude", value=52.37)
lon = st.number_input("Longitude", value=4.90)
days = st.slider("Days ahead", 1, 7, 5)

if st.button("Genereer aanbevelingen"):
    # 1) haal historische weekdagprofielen uit jouw bestaande data pipelines
    stores_hist_by_weekday = make_hist_by_weekday()  # gebruikt jouw Vemcount-normalisatie (reeds werkend)
    # 2) run advisor
    advice = build_advice(
        company_name="Your Company",
        stores_hist_by_weekday=stores_hist_by_weekday,
        lat=lat, lon=lon,
        api_key=cfg.get("openweather_api_key",""),
        days=days
    )
    # 3) Toon compact: tiles + expander per dag
    st.metric("Consumentenvertrouwen (CBS)", f'{advice["cci"]}')
    for d in advice["days"]:
        with st.expander(f'📅 {d["date"]} — temp {d["weather"]["temp"]:.1f}°C • pop {int(d["weather"]["pop"]*100)}%'):
            for s in d["stores"]:
                st.markdown(f"**{s['store']}**")
                st.write("— Storemanager:", " • ".join(s["store_actions"]))
                st.write("— Regiomanager:", " • ".join(s["regional_actions"]))
