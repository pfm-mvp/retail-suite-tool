# pages/01_Store_Live_Ops.py
# Streamlit page: Store Live Ops (visitors today vs. yesterday, same window)
# Requirements:
# - utils_pfmx.api_get_report(params: list[tuple]) -> dict (POST with query params, no [] in keys)
# - utils_pfmx.shop_mapping: Dict[int, str] (id -> readable name)  [optional but recommended]
# - streamlit secrets hold API_URL if your helper needs it

import streamlit as st
import datetime as dt
from zoneinfo import ZoneInfo

# ---- Local imports (existing in your project) ----
try:
    from utils_pfmx import api_get_report, shop_mapping
except Exception:
    # Fallback for local dev if imports differ
    from utils_pfmx import api_get_report
    shop_mapping = {}

st.set_page_config(page_title="Store Live Ops", page_icon="👣", layout="wide")

st.title("👣 Store Live Ops")
st.caption("Bezoekers *vandaag tot nu* vs. *gisteren in hetzelfde tijdvenster*.")

# ---------- Store selector ----------
def pick_store():
    if shop_mapping:
        # order by readable name
        items = sorted([(name, sid) for sid, name in shop_mapping.items()], key=lambda x: x[0].lower())
        names = [n for n, _ in items]
        sname = st.selectbox("Winkel", names, index=0)
        sid = dict(items)[sname]
        return sid, sname
    else:
        sid = st.number_input("Shop ID", min_value=1, step=1, value=1)
        return int(sid), f"Shop {sid}"

shop_id, shop_name = pick_store()

# ---------- Openingstijden als slider ----------
st.markdown("#### Openingstijden (venster voor vergelijking)")
open_time, close_time = st.slider(
    "Kies het venster voor vandaag & gisteren",
    value=(dt.time(9, 0), dt.time(18, 0)),
    format="HH:mm"
)

# ---------- Huidige tijd (Europe/Amsterdam) ----------
now_local = dt.datetime.now(ZoneInfo("Europe/Amsterdam"))
today = now_local.date()

# Begrenzen van het 'tot-nu' moment
if now_local.time() < open_time:
    now_cut = open_time
else:
    now_cut = min(now_local.time(), close_time)

from_start_from = open_time.strftime("%H:%M")
from_start_to   = now_cut.strftime("%H:%M")

# ---------- Helper: robuuste parser ----------
def sum_count_in(resp_json) -> int:
    """
    Extract day total (or partial window) for count_in from /get-report.
    Supports both day-level and timestamp-split structures.
    """
    try:
        data_block = resp_json.get("data", {})
        # find a date_* key
        date_key = next((k for k in data_block.keys() if str(k).startswith("date_")), None)
        if not date_key:
            return 0
        shop_block = data_block.get(date_key, {})
        # first shop-id key
        if not isinstance(shop_block, dict) or not shop_block:
            return 0
        shop_id_key = next(iter(shop_block.keys()))
        shop_data = shop_block.get(shop_id_key, {})
        # direct day-level
        if isinstance(shop_data.get("data"), dict) and "count_in" in shop_data["data"]:
            return int(round(float(shop_data["data"]["count_in"])))
        # fallback per timestamp
        total = 0.0
        for node in shop_data.get("dates", {}).values():
            if "data" in node and "count_in" in node["data"]:
                try:
                    total += float(node["data"]["count_in"])
                except Exception:
                    pass
        return int(round(total))
    except Exception:
        return 0

# ---------- Queries ----------
params_today = [
    ("source", "shops"),
    ("period", "day"),
    ("data",   int(shop_id)),
    ("data_output", "count_in"),
    ("from_start_from", from_start_from),
    ("from_start_to",   from_start_to),
]
params_yest = [
    ("source", "shops"),
    ("period", "yesterday"),
    ("data",   int(shop_id)),
    ("data_output", "count_in"),
    ("from_start_from", from_start_from),
    ("from_start_to",   from_start_to),
]

with st.spinner("Data ophalen..."):
    resp_today = api_get_report(params_today)
    resp_yest  = api_get_report(params_yest)

today_count = sum_count_in(resp_today)
yesterday_count = sum_count_in(resp_yest)
delta = today_count - yesterday_count
pct = (delta / yesterday_count * 100.0) if yesterday_count > 0 else 0.0

# ---------- Metrics ----------
def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")

c1, c2, c3 = st.columns(3)
c1.metric("Vandaag (tot nu)", fmt_int(today_count))
c2.metric("Gisteren (zelfde venster)", fmt_int(yesterday_count))
c3.metric("Verschil", f"{delta:+}".replace("+", "+ ").replace("-", "− "), f"{pct:.1f}%")

# ---------- Context hints ----------
st.divider()
colA, colB = st.columns([1,2])
with colA:
    st.subheader("Instellingen")
    st.write(f"**Winkel:** {shop_name} (ID: {shop_id})")
    st.write(f"**Venster:** {from_start_from} → {from_start_to}")
with colB:
    if now_local.time() < open_time:
        st.info("De winkel is nog niet open volgens de ingestelde openingstijd; ‘Vandaag’ telt vanaf opening.")
    elif now_local.time() > close_time:
        st.caption("Tip: pas de sluitingstijd aan als je vanavond langer open bent.")

# ---------- Debug (optioneel) ----------
with st.expander("Debug (API responses)"):
    st.json({"params_today": params_today, "resp_today": resp_today})
    st.json({"params_yesterday": params_yest, "resp_yesterday": resp_yest})
