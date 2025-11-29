# pages/05_Retail_AI_Copilot.py

import numpy as np
import pandas as pd
import requests
import streamlit as st

from datetime import datetime, timedelta

from helpers_clients import load_clients
from helpers_normalize import normalize_vemcount_response
from services.cbs_service import get_cbs_stats_for_postcode4
from services.pathzz_service import fetch_monthly_street_traffic

st.set_page_config(
    page_title="PFM Retail Performance Copilot",
    layout="wide"
)

# ----------------------
# API URL / secrets setup
# ----------------------

# In jouw setup wijst API_URL naar /get-report, omdat andere tools daar direct op praten.
# Voor deze Copilot splitsen we 'm op:
# - REPORT_URL = volledige /get-report URL (voor metrics)
# - FASTAPI_BASE_URL = root zonder /get-report (voor /company/{company_id}/location)
raw_api_url = st.secrets["API_URL"].rstrip("/")

if raw_api_url.endswith("/get-report"):
    REPORT_URL = raw_api_url
    FASTAPI_BASE_URL = raw_api_url.rsplit("/get-report", 1)[0]
else:
    FASTAPI_BASE_URL = raw_api_url
    REPORT_URL = raw_api_url + "/get-report"

VISUALCROSSING_KEY = st.secrets.get("visualcrossing_key", None)

# -------------
# Format helpers
# -------------

def fmt_eur(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"€ {x:,.0f}".replace(",", ".")


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:.1f}%".replace(".", ",")


def fmt_int(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:,.0f}".replace(",", ".")


# -------------
# API helpers
# -------------

@st.cache_data(ttl=600)
def get_locations_by_company(company_id: int) -> pd.DataFrame:
    """
    Wrapper rond /company/{company_id}/location van de vemcount-agent.
    """
    url = f"{FASTAPI_BASE_URL.rstrip('/')}/company/{company_id}/location"

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict) and "locations" in data:
        df = pd.DataFrame(data["locations"])
    else:
        df = pd.DataFrame(data)

    return df

@st.cache_data(ttl=600)
def get_report(
    shop_ids,
    data_outputs,
    period: str,
    step: str = "day",
    source: str = "shops",
    company_id: int | None = None,  # voor toekomst, nu niet gebruikt
):
    """
    Wrapper rond /get-report (POST) van de vemcount-agent, met querystring zonder [].

    We gebruiken hier REPORT_URL, die in jouw setup gelijk is aan
    https://vemcount-agent.onrender.com/get-report
    """
    params: list[tuple[str, str]] = []

    for sid in shop_ids:
        params.append(("data", str(sid)))

    for dout in data_outputs:
        params.append(("data_output", dout))

    params.append(("period", period))
    params.append(("step", step))
    params.append(("source", source))

    # LET OP: geen 'company' meesturen — Vemcount heeft genoeg aan data=shop_ids

    resp = requests.post(REPORT_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()

# -------------
# Weather helper (Visual Crossing)
# -------------

@st.cache_data(ttl=3600)
def fetch_visualcrossing_history(location_str: str, start_date, end_date) -> pd.DataFrame:
    """
    Haalt historische daily weather data op via Visual Crossing.

    location_str:
    - "52.3702,4.8952" (lat,lon) of
    - "Amsterdam,NL"
    """
    if not VISUALCROSSING_KEY:
        return pd.DataFrame()

    start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end = pd.to_datetime(end_date).strftime("%Y-%m-%d")

    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location_str}/{start}/{end}"
    params = {
        "unitGroup": "metric",
        "key": VISUALCROSSING_KEY,
        "include": "days",
    }

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if "days" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["days"])
    df["date"] = pd.to_datetime(df["datetime"])
    return df[["date", "temp", "precip", "windspeed"]]


# -------------
# KPI helpers
# -------------

def compute_daily_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "turnover" in df.columns and "footfall" in df.columns:
        df["sales_per_visitor"] = np.where(
            df["footfall"] > 0,
            df["turnover"] / df["footfall"],
            np.nan,
        )
    if "transactions" in df.columns and "footfall" in df.columns:
        df["conversion_rate"] = np.where(
            df["footfall"] > 0,
            df["transactions"] / df["footfall"] * 100,
            np.nan,
        )
    return df


def aggregate_monthly(df: pd.DataFrame, sqm: float | None) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    agg = df.groupby("month").agg(
        {
            "footfall": "sum",
            "turnover": "sum",
            "sales_per_visitor": "mean",
            "sales_per_sqm": "mean" if "sales_per_sqm" in df.columns else "mean",
        }
    ).reset_index()

    if sqm and sqm > 0:
        agg["turnover_per_sqm"] = agg["turnover"] / sqm
        agg["footfall_per_sqm"] = agg["footfall"] / sqm
    else:
        agg["turnover_per_sqm"] = np.nan
        agg["footfall_per_sqm"] = np.nan

    return agg


def compute_capture_rate(store_monthly: pd.DataFrame, street_monthly: pd.DataFrame) -> pd.DataFrame:
    s = store_monthly.copy()
    st_df = street_monthly.copy()

    s["month"] = pd.to_datetime(s["month"])
    st_df["month"] = pd.to_datetime(st_df["month"])

    merged = pd.merge(s, st_df, on="month", how="left")

    if "street_footfall" in merged.columns and "footfall" in merged.columns:
        merged["capture_rate"] = np.where(
            merged["street_footfall"] > 0,
            merged["footfall"] / merged["street_footfall"] * 100,
            np.nan,
        )
    else:
        merged["capture_rate"] = np.nan

    return merged


def compute_yoy_monthly(df_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    YoY op maandniveau, op basis van laatste jaar vs jaar ervoor.
    """
    df = df_monthly.copy()
    df["year"] = df["month"].dt.year
    df["month_num"] = df["month"].dt.month

    cur_year = df["year"].max()
    prev_year = cur_year - 1

    cur = df[df["year"] == cur_year]
    prev = df[df["year"] == prev_year]

    if cur.empty or prev.empty:
        return df_monthly

    merged = pd.merge(
        cur,
        prev,
        on="month_num",
        how="left",
        suffixes=("_cur", "_prev"),
    )

    for col in ["footfall", "turnover", "capture_rate", "turnover_per_sqm"]:
        cur_col = f"{col}_cur"
        prev_col = f"{col}_prev"
        yoy_col = f"{col}_yoy_pct"
        if cur_col in merged.columns and prev_col in merged.columns:
            merged[yoy_col] = np.where(
                merged[prev_col] > 0,
                (merged[cur_col] - merged[prev_col]) / merged[prev_col] * 100,
                np.nan,
            )

    return merged


def generate_insights(yoy_df: pd.DataFrame) -> list[str]:
    insights: list[str] = []
    if yoy_df.empty:
        insights.append("Nog onvoldoende maanddata voor een YoY-analyse.")
        return insights

    # Omzet YoY
    if "turnover_cur" in yoy_df.columns and "turnover_prev" in yoy_df.columns:
        cur_total = yoy_df["turnover_cur"].sum()
        prev_total = yoy_df["turnover_prev"].sum()
        if prev_total > 0:
            yoy = (cur_total - prev_total) / prev_total * 100
            insights.append(f"Totale omzet over de gekozen maanden: {yoy:+.1f}% vs vorig jaar.")

    # Footfall YoY
    if "footfall_cur" in yoy_df.columns and "footfall_prev" in yoy_df.columns:
        cur_f = yoy_df["footfall_cur"].sum()
        prev_f = yoy_df["footfall_prev"].sum()
        if prev_f > 0:
            yoyf = (cur_f - prev_f) / prev_f * 100
            insights.append(f"Totale footfall over de gekozen maanden: {yoyf:+.1f}% vs vorig jaar.")

    # Capture rate YoY
    if "capture_rate_cur" in yoy_df.columns and "capture_rate_prev" in yoy_df.columns:
        cr_cur = yoy_df["capture_rate_cur"].mean()
        cr_prev = yoy_df["capture_rate_prev"].mean()
        if pd.notna(cr_cur) and pd.notna(cr_prev) and cr_prev > 0:
            cr_yoy = (cr_cur - cr_prev) / cr_prev * 100
            if cr_yoy < -5:
                insights.append(
                    f"Capture rate is ~{cr_yoy:.1f}% gedaald vs vorig jaar – straatdrukte groeit harder dan winkeltraffic."
                )
            elif cr_yoy > 5:
                insights.append(
                    f"Capture rate is ~{cr_yoy:.1f}% gestegen vs vorig jaar – je trekt relatief meer passanten naar binnen."
                )

    # m²-index YoY
    if "turnover_per_sqm_cur" in yoy_df.columns and "turnover_per_sqm_prev" in yoy_df.columns:
        m2_cur = yoy_df["turnover_per_sqm_cur"].mean()
        m2_prev = yoy_df["turnover_per_sqm_prev"].mean()
        if m2_prev and m2_prev > 0:
            m2_yoy = (m2_cur - m2_prev) / m2_prev * 100
            if m2_yoy < -5:
                insights.append(
                    f"Omzet per m² ligt gemiddeld {m2_yoy:.1f}% lager dan vorig jaar – ruimteproductiviteit is afgenomen."
                )
            elif m2_yoy > 5:
                insights.append(
                    f"Omzet per m² ligt gemiddeld {m2_yoy:.1f}% hoger dan vorig jaar – sterke verbetering in ruimteproductiviteit."
                )

    if not insights:
        insights.append("De prestaties zijn redelijk stabiel; geen opvallende afwijkingen op maandniveau.")
    return insights

# -------------
# MAIN UI
# -------------

def main():
    st.title("PFM Retail Performance Copilot – Fase 1")

    # --- Retailer selectie via clients.json ---
    clients = load_clients("clients.json")
    clients_df = pd.DataFrame(clients)
    clients_df["label"] = clients_df.apply(
        lambda r: f"{r['brand']} – {r['name']} (company_id {r['company_id']})",
        axis=1,
    )

    st.sidebar.header("Selecteer retailer & winkel")

    client_label = st.sidebar.selectbox("Retailer", clients_df["label"].tolist())
    selected_client = clients_df[clients_df["label"] == client_label].iloc[0].to_dict()
    company_id = int(selected_client["company_id"])

    # --- Winkels ophalen via FastAPI ---
    locations_df = get_locations_by_company(company_id)
    if locations_df.empty:
        st.error("Geen winkels gevonden voor deze retailer.")
        return

    if "name" not in locations_df.columns:
        locations_df["name"] = locations_df["id"].astype(str)

    locations_df["label"] = locations_df.apply(
        lambda r: f"{r['name']} (ID: {r['id']})", axis=1
    )

    shop_label = st.sidebar.selectbox("Winkel", locations_df["label"].tolist())
    shop_row = locations_df[locations_df["label"] == shop_label].iloc[0].to_dict()
    shop_id = int(shop_row["id"])
    sqm = float(shop_row.get("sqm", 0) or 0)
    postcode = shop_row.get("postcode", "")
    lat = float(shop_row.get("lat", 0) or 0)
    lon = float(shop_row.get("lon", 0) or 0)

    # --- Periode selectie (huidige vs vorige periode) ---
    period_choice = st.sidebar.selectbox(
        "Periode",
        [
            "Deze week",
            "Laatste week",
            "Deze maand",
            "Laatste maand",
            "Dit kwartaal",
            "Laatste kwartaal",
        ],
        index=2,  # default: Deze maand
    )

    today = datetime.today().date()

    def get_week_range(base_date):
        """Maandag–zondag week van base_date."""
        wd = base_date.weekday()  # 0=ma
        start = base_date - timedelta(days=wd)
        end = start + timedelta(days=6)
        return start, end

    def get_month_range(year, month):
        start = datetime(year, month, 1).date()
        if month == 12:
            next_start = datetime(year + 1, 1, 1).date()
        else:
            next_start = datetime(year, month + 1, 1).date()
        end = next_start - timedelta(days=1)
        return start, end

    def get_quarter_range(year, month):
        q = (month - 1) // 3 + 1
        start_month = 3 * (q - 1) + 1
        start = datetime(year, start_month, 1).date()
        if start_month == 10:
            next_start = datetime(year + 1, 1, 1).date()
        else:
            next_start = datetime(year, start_month + 3, 1).date()
        end = next_start - timedelta(days=1)
        return start, end

    # Bereken huidige + vorige periode
    if period_choice == "Deze week":
        start_cur, end_cur = get_week_range(today)
        start_prev, end_prev = start_cur - timedelta(days=7), start_cur - timedelta(days=1)

    elif period_choice == "Laatste week":
        # Laatste volledige week vóór deze week
        this_week_start, _ = get_week_range(today)
        end_cur = this_week_start - timedelta(days=1)
        start_cur = end_cur - timedelta(days=6)
        start_prev = start_cur - timedelta(days=7)
        end_prev = start_cur - timedelta(days=1)

    elif period_choice == "Deze maand":
        start_cur, end_cur = get_month_range(today.year, today.month)
        # vorige maand
        if today.month == 1:
            prev_y, prev_m = today.year - 1, 12
        else:
            prev_y, prev_m = today.year, today.month - 1
        start_prev, end_prev = get_month_range(prev_y, prev_m)

    elif period_choice == "Laatste maand":
        # vorige maand = huidige periode
        if today.month == 1:
            cur_y, cur_m = today.year - 1, 12
        else:
            cur_y, cur_m = today.year, today.month - 1
        start_cur, end_cur = get_month_range(cur_y, cur_m)
        # maand daar weer voor
        if cur_m == 1:
            prev_y, prev_m = cur_y - 1, 12
        else:
            prev_y, prev_m = cur_y, cur_m - 1
        start_prev, end_prev = get_month_range(prev_y, prev_m)

    elif period_choice == "Dit kwartaal":
        start_cur, end_cur = get_quarter_range(today.year, today.month)
        # vorige kwartaal
        cur_q = (today.month - 1) // 3 + 1
        if cur_q == 1:
            prev_y, prev_q = today.year - 1, 4
        else:
            prev_y, prev_q = today.year, cur_q - 1
        prev_start_month = 3 * (prev_q - 1) + 1
        start_prev, end_prev = get_quarter_range(prev_y, prev_start_month)

    else:  # "Laatste kwartaal"
        cur_q = (today.month - 1) // 3 + 1
        # huidige periode = vorige kwartaal
        if cur_q == 1:
            cur_y, cur_q_eff = today.year - 1, 4
        else:
            cur_y, cur_q_eff = today.year, cur_q - 1
        cur_start_month = 3 * (cur_q_eff - 1) + 1
        start_cur, end_cur = get_quarter_range(cur_y, cur_start_month)

        # periode daar weer voor
        if cur_q_eff == 1:
            prev_y, prev_q = cur_y - 1, 4
        else:
            prev_y, prev_q = cur_y, cur_q_eff - 1
        prev_start_month = 3 * (prev_q - 1) + 1
        start_prev, end_prev = get_quarter_range(prev_y, prev_start_month)

    # --- Weather & CBS input ---
    default_city = shop_row.get("city") or "Amsterdam"
    weather_location = st.sidebar.text_input(
        "Weerlocatie",
        value=f"{default_city},NL",
    )
    postcode4 = st.sidebar.text_input(
        "CBS postcode (4-cijferig)",
        value=postcode[:4] if postcode else "",
    )

    run_btn = st.sidebar.button("Analyseer", type="primary")

    if not run_btn:
        st.info("Selecteer retailer & winkel, kies een periode en klik op **Analyseer**.")
        return

    # --- Data ophalen uit FastAPI ---
    with st.spinner("Data ophalen uit Storescan / FastAPI..."):
        metric_map = {
            "count_in": "footfall",
            "turnover": "turnover",
            "sales_per_sqm": "sales_per_sqm",
        }

        # Haal heel het jaar op, snij zelf de periodes eruit
        resp_all = get_report(
            [shop_id],
            list(metric_map.keys()),
            period="this_year",
            step="day",
            source="shops",
        )
        df_all_raw = normalize_vemcount_response(
            resp_all,
            kpi_keys=metric_map.keys(),
        )
        df_all_raw = df_all_raw.rename(columns=metric_map)

    if df_all_raw.empty:
        st.warning("Geen data gevonden voor dit jaar voor deze winkel.")
        return

    # Zorg dat 'date' datetime is
    df_all_raw["date"] = pd.to_datetime(df_all_raw["date"], errors="coerce")

    # Slice naar huidige + vorige periode
    start_cur_ts = pd.Timestamp(start_cur)
    end_cur_ts = pd.Timestamp(end_cur)
    start_prev_ts = pd.Timestamp(start_prev)
    end_prev_ts = pd.Timestamp(end_prev)

    df_cur = df_all_raw[
        (df_all_raw["date"] >= start_cur_ts)
        & (df_all_raw["date"] <= end_cur_ts)
    ].copy()

    df_prev = df_all_raw[
        (df_all_raw["date"] >= start_prev_ts)
        & (df_all_raw["date"] <= end_prev_ts)
    ].copy()

    if df_cur.empty and df_prev.empty:
        st.warning("Geen data gevonden in de gekozen periodes.")
        return

    # KPI's berekenen
    df_cur = compute_daily_kpis(df_cur)
    if not df_prev.empty:
        df_prev = compute_daily_kpis(df_prev)

    # --- Weerdata via Visual Crossing ---
    weather_df = pd.DataFrame()
    if weather_location and VISUALCROSSING_KEY:
        weather_df = fetch_visualcrossing_history(weather_location, start_cur, end_cur)

    # --- Pathzz street traffic (maandniveau) ---
    pathzz_monthly = pd.DataFrame()
    if lat and lon:
        pathzz_monthly = fetch_monthly_street_traffic(
            lat=lat,
            lon=lon,
            start_date=datetime.combine(start_cur, datetime.min.time()),
            end_date=datetime.combine(end_cur, datetime.min.time()),
            radius_m=100,
        )

    # --- CBS context ---
    cbs_stats = {}
    if postcode4:
        cbs_stats = get_cbs_stats_for_postcode4(postcode4)

    # --- KPI-cards met vergelijking vorige periode ---
    st.subheader(f"{selected_client['brand']} – {shop_row['name']}")

    foot_cur = df_cur["footfall"].sum() if "footfall" in df_cur.columns else 0
    foot_prev = df_prev["footfall"].sum() if ("footfall" in df_prev.columns and not df_prev.empty) else 0
    foot_delta = None
    if foot_prev > 0:
        foot_delta_val = (foot_cur - foot_prev) / foot_prev * 100
        foot_delta = f"{foot_delta_val:+.1f}%"

    turn_cur = df_cur["turnover"].sum() if "turnover" in df_cur.columns else 0
    turn_prev = df_prev["turnover"].sum() if ("turnover" in df_prev.columns and not df_prev.empty) else 0
    turn_delta = None
    if turn_prev > 0:
        turn_delta_val = (turn_cur - turn_prev) / turn_prev * 100
        turn_delta = f"{turn_delta_val:+.1f}%"

    spv_cur = df_cur["sales_per_visitor"].mean() if "sales_per_visitor" in df_cur.columns else np.nan
    spv_prev = df_prev["sales_per_visitor"].mean() if ("sales_per_visitor" in df_prev.columns and not df_prev.empty) else np.nan
    spv_delta = None
    if pd.notna(spv_cur) and pd.notna(spv_prev) and spv_prev > 0:
        spv_delta_val = (spv_cur - spv_prev) / spv_prev * 100
        spv_delta = f"{spv_delta_val:+.1f}%"

    conv_cur = df_cur["conversion_rate"].mean() if "conversion_rate" in df_cur.columns else np.nan
    conv_prev = df_prev["conversion_rate"].mean() if ("conversion_rate" in df_prev.columns and not df_prev.empty) else np.nan
    conv_delta = None
    if pd.notna(conv_cur) and pd.notna(conv_prev) and conv_prev > 0:
        conv_delta_val = (conv_cur - conv_prev) / conv_prev * 100
        conv_delta = f"{conv_delta_val:+.1f}%"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Footfall (periode)", fmt_int(foot_cur), delta=foot_delta)
    with col2:
        st.metric("Omzet (periode)", fmt_eur(turn_cur), delta=turn_delta)
    with col3:
        if "sales_per_visitor" in df_cur.columns:
            value = f"€ {spv_cur:.2f}".replace(".", ",") if pd.notna(spv_cur) else "-"
            st.metric("Gem. besteding/visitor", value, delta=spv_delta)
    with col4:
        if "conversion_rate" in df_cur.columns:
            st.metric("Gem. conversie", fmt_pct(conv_cur) if pd.notna(conv_cur) else "-", delta=conv_delta)

    # --- Dagelijkse grafiek ---
    st.markdown("### Dagelijkse footfall & omzet")
    if "footfall" in df_cur.columns and "turnover" in df_cur.columns:
        daily_chart = df_cur.set_index("date")[["footfall", "turnover"]]
        st.line_chart(daily_chart)

    # --- Weer vs footfall (optioneel) ---
    if not weather_df.empty:
        st.markdown("#### Weer vs footfall (indicatief)")
        m = pd.merge(
            df_cur[["date", "footfall"]],
            weather_df[["date", "temp", "precip"]],
            on="date",
            how="left",
        ).set_index("date")
        # toont 3 lijnen: footfall, temperatuur, neerslag
        st.line_chart(m)

    # Voor nu: maandniveau / capture-rate laten we zoals eerder (of later verfijnen),
    # maar je hebt nu in elk geval:
    # - Periode-selectie
    # - Vergelijking met vorige periode
    # - KPI-delta's in de cards

    # --- Debug ---
    with st.expander("🔧 Debug"):
        st.write("Periode keuze:", period_choice)
        st.write("Huidige periode:", start_cur, "→", end_cur)
        st.write("Vorige periode:", start_prev, "→", end_prev)
        st.write("Shop row:", shop_row)
        st.write("Dagdata ALL (head):", df_all_raw.head())
        st.write("Dagdata (cur):", df_cur.head())
        st.write("Dagdata (prev):", df_prev.head())
        st.write("Pathzz monthly:", pathzz_monthly.head())
        st.write("CBS stats:", cbs_stats)
        st.write("Weather df:", weather_df.head())

if __name__ == "__main__":
    main()
