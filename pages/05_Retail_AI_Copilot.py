# pages/05_Retail_AI_Copilot.py

import numpy as np
import pandas as pd
import requests
import streamlit as st

from datetime import datetime, timedelta

from helpers_clients import load_clients
from helpers_normalize import normalize_vemcount_response
from services.cbs_service import get_cbs_stats_for_postcode4
from services.pathzz_service import fetch_weekly_street_traffic  # ← weekly import

st.set_page_config(
    page_title="PFM Retail Performance Copilot",
    layout="wide"
)

# ----------------------
# API URL / secrets setup
# ----------------------

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
    """
    params: list[tuple[str, str]] = []

    for sid in shop_ids:
        params.append(("data", str(sid)))

    for dout in data_outputs:
        params.append(("data_output", dout))

    params.append(("period", period))
    params.append(("step", step))
    params.append(("source", source))

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


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregeer store-data naar weekniveau (week_start = maandag).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["week_start"] = df["date"].dt.to_period("W").dt.start_time

    agg = df.groupby("week_start").agg(
        {
            "footfall": "sum",
            "turnover": "sum",
            "sales_per_visitor": "mean",
            "conversion_rate": "mean",
        }
    ).reset_index()

    return agg


def aggregate_monthly(df: pd.DataFrame, sqm: float | None) -> pd.DataFrame:
    """
    Aggregatie naar maandniveau (voor eventuele andere analyses).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    agg_dict: dict[str, str] = {}

    if "footfall" in df.columns:
        agg_dict["footfall"] = "sum"
    if "turnover" in df.columns:
        agg_dict["turnover"] = "sum"
    if "sales_per_visitor" in df.columns:
        agg_dict["sales_per_visitor"] = "mean"
    if "sales_per_sqm" in df.columns:
        agg_dict["sales_per_sqm"] = "mean"

    if not agg_dict:
        return df[["month"]].drop_duplicates().reset_index(drop=True)

    agg = df.groupby("month", as_index=False).agg(agg_dict)

    if sqm and sqm > 0:
        if "turnover" in agg.columns:
            agg["turnover_per_sqm"] = agg["turnover"] / sqm
        else:
            agg["turnover_per_sqm"] = np.nan

        if "footfall" in agg.columns:
            agg["footfall_per_sqm"] = agg["footfall"] / sqm
        else:
            agg["footfall_per_sqm"] = np.nan
    else:
        agg["turnover_per_sqm"] = np.nan
        agg["footfall_per_sqm"] = np.nan

    return agg


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
        this_week_start, _ = get_week_range(today)
        end_cur = this_week_start - timedelta(days=1)
        start_cur = end_cur - timedelta(days=6)
        start_prev = start_cur - timedelta(days=7)
        end_prev = start_cur - timedelta(days=1)

    elif period_choice == "Deze maand":
        start_cur, end_cur = get_month_range(today.year, today.month)
        if today.month == 1:
            prev_y, prev_m = today.year - 1, 12
        else:
            prev_y, prev_m = today.year, today.month - 1
        start_prev, end_prev = get_month_range(prev_y, prev_m)

    elif period_choice == "Laatste maand":
        if today.month == 1:
            cur_y, cur_m = today.year - 1, 12
        else:
            cur_y, cur_m = today.year, today.month - 1
        start_cur, end_cur = get_month_range(cur_y, cur_m)
        if cur_m == 1:
            prev_y, prev_m = cur_y - 1, 12
        else:
            prev_y, prev_m = cur_y, cur_m - 1
        start_prev, end_prev = get_month_range(prev_y, prev_m)

    elif period_choice == "Dit kwartaal":
        start_cur, end_cur = get_quarter_range(today.year, today.month)
        cur_q = (today.month - 1) // 3 + 1
        if cur_q == 1:
            prev_y, prev_q = today.year - 1, 4
        else:
            prev_y, prev_q = today.year, cur_q - 1
        prev_start_month = 3 * (prev_q - 1) + 1
        start_prev, end_prev = get_quarter_range(prev_y, prev_start_month)

    else:  # "Laatste kwartaal"
        cur_q = (today.month - 1) // 3 + 1
        if cur_q == 1:
            cur_y, cur_q_eff = today.year - 1, 4
        else:
            cur_y, cur_q_eff = today.year, cur_q - 1
        cur_start_month = 3 * (cur_q_eff - 1) + 1
        start_cur, end_cur = get_quarter_range(cur_y, cur_start_month)

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

    # --- Pathzz street traffic (weekly demo) ---
    pathzz_weekly = fetch_weekly_street_traffic(
        start_date=start_cur,
        end_date=end_cur,
    )

    cap_weekly = pd.DataFrame()
    avg_capture_cur = None
    avg_capture_prev = None  # nog niet gebruikt, maar placeholder voor toekomst

    if not pathzz_weekly.empty:
        # Weekly store data over heel jaar, daarna gefilterd op geselecteerde weken
        store_weekly_all = aggregate_weekly(df_all_raw)

        cap_weekly = pd.merge(
            store_weekly_all,
            pathzz_weekly,
            on="week_start",
            how="inner",
        )

        if not cap_weekly.empty and "street_footfall" in cap_weekly.columns:
            cap_weekly["capture_rate"] = np.where(
                cap_weekly["street_footfall"] > 0,
                cap_weekly["footfall"] / cap_weekly["street_footfall"] * 100,
                np.nan,
            )

            st.markdown("### Weekly straatdrukte vs winkeltraffic (Pathzz demo)")

            chart_df = cap_weekly.set_index("week_start")[["footfall", "street_footfall"]]
            chart_df = chart_df.rename(
                columns={
                    "footfall": "store_footfall",
                    "street_footfall": "street_footfall",
                }
            )
            st.line_chart(chart_df)

    # --- CBS context (data ophalen) ---
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
        # Voor nu: altijd conversie tonen (capture rate komt later netjes met weekly logica)
        if "conversion_rate" in df_cur.columns:
            st.metric(
                "Gem. conversie",
                fmt_pct(conv_cur) if pd.notna(conv_cur) else "-",
                delta=conv_delta,
            )

    # --- Dagelijkse grafiek ---
    st.markdown("### Dagelijkse footfall & omzet")
    if "footfall" in df_cur.columns and "turnover" in df_cur.columns:
        daily_chart = df_cur.set_index("date")[["footfall", "turnover"]]
        st.line_chart(daily_chart)

    # --- Weer vs footfall (optioneel) ---
    if not weather_df.empty:
        st.markdown("### Weer vs footfall (indicatief)")

        m = pd.merge(
            df_cur[["date", "footfall"]],
            weather_df[["date", "temp", "precip"]],
            on="date",
            how="left",
        ).set_index("date")

        m_plot = m.copy()
        max_foot = m_plot["footfall"].max() if "footfall" in m_plot.columns else None

        if max_foot and not np.isnan(max_foot) and max_foot > 0:
            if "temp" in m_plot.columns:
                max_temp = m_plot["temp"].abs().max()
                if max_temp and not np.isnan(max_temp) and max_temp > 0:
                    m_plot["temp (index)"] = m_plot["temp"] / max_temp * max_foot
            if "precip" in m_plot.columns:
                max_precip = m_plot["precip"].abs().max()
                if max_precip and not np.isnan(max_precip) and max_precip > 0:
                    m_plot["precip (index)"] = m_plot["precip"] / max_precip * max_foot

        cols = ["footfall"]
        for c in ["temp (index)", "precip (index)"]:
            if c in m_plot.columns:
                cols.append(c)

        st.line_chart(m_plot[cols])

    # --- Weekly capture tabel (optioneel) ---
    if not cap_weekly.empty and "street_footfall" in cap_weekly.columns:
        st.markdown("### Weekly straatdrukte & capture rate")
        cap_view = cap_weekly[["week_start", "street_footfall", "footfall", "capture_rate"]].copy()
        cap_view["week_start"] = pd.to_datetime(cap_view["week_start"]).dt.strftime("%Y-%m-%d")
        cap_view = cap_view.rename(
            columns={
                "street_footfall": "Street footfall",
                "footfall": "Store footfall",
                "capture_rate": "Capture rate (%)",
            }
        )
        st.dataframe(cap_view, use_container_width=True)

    # --- AI Insights (lichte interpretatie huidige vs vorige periode) ---
    st.markdown("### AI Insights (huidige vs vorige periode)")
    insights = []
    if foot_delta is not None:
        insights.append(f"Footfall veranderde met {foot_delta} vs de vorige periode.")
    if turn_delta is not None:
        insights.append(f"Omzet veranderde met {turn_delta} vs de vorige periode.")
    if spv_delta is not None:
        insights.append(f"Gemiddelde besteding per bezoeker veranderde met {spv_delta}.")
    if conv_delta is not None:
        insights.append(f"Gemiddelde conversie veranderde met {conv_delta}.")
    if not insights:
        insights.append("Nog onvoldoende data om een goede vergelijking te maken met de vorige periode.")
    for txt in insights:
        st.markdown(f"- {txt}")

    # --- CBS context ---
    if cbs_stats:
        st.markdown("### CBS context (postcodegebied)")
        c1, c2 = st.columns(2)
        if "avg_income_index" in cbs_stats:
            c1.metric("Inkomensindex (NL = 100)", cbs_stats["avg_income_index"])
        if "population_density_index" in cbs_stats:
            c2.metric("Bevolkingsdichtheid-index", cbs_stats["population_density_index"])
        if "note" in cbs_stats:
            st.caption(cbs_stats["note"])

    # --- Debug ---
    with st.expander("🔧 Debug"):
        st.write("Periode keuze:", period_choice)
        st.write("Huidige periode:", start_cur, "→", end_cur)
        st.write("Vorige periode:", start_prev, "→", end_prev)
        st.write("Shop row:", shop_row)
        st.write("Dagdata ALL (head):", df_all_raw.head())
        st.write("Dagdata (cur):", df_cur.head())
        st.write("Dagdata (prev):", df_prev.head())
        st.write("Pathzz weekly:", pathzz_weekly.head())
        st.write("Capture weekly:", cap_weekly.head())
        st.write("CBS stats:", cbs_stats)
        st.write("Weather df:", weather_df.head())

if __name__ == "__main__":
    main()
