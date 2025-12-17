# pages/05_Retail_AI_Store_Copilot.py

import os
import time  # ✅ for retry backoff
import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

try:
    from openai import OpenAI
    _OPENAI_INSTALLED = True
except Exception:
    OpenAI = None
    _OPENAI_INSTALLED = False

from datetime import datetime, timedelta

from helpers_clients import load_clients
from helpers_normalize import normalize_vemcount_response
from services.cbs_service import get_cbs_stats_for_postcode4
from services.forecast_service import (
    build_simple_footfall_turnover_forecast,
    build_pro_footfall_turnover_forecast,
)

# ✅ REUSE EXISTING SVI SERVICE (single source of truth)
from services.svi_service import build_store_vitality

# Probeer de service-import; als die er niet is, gebruik een lokale CSV-loader
try:
    from services.pathzz_service import fetch_monthly_street_traffic
except Exception:
    def fetch_monthly_street_traffic(start_date, end_date):
        csv_path = "data/pathzz_sample_weekly.csv"
        try:
            df = pd.read_csv(csv_path, sep=";", dtype={"Visits": "string"})
        except Exception:
            return pd.DataFrame()

        df = df.rename(columns={"Week": "week", "Visits": "street_footfall"})
        df["street_footfall"] = (
            df["street_footfall"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )

        def _parse_week_start(s):
            if isinstance(s, str) and "To" in s:
                return pd.to_datetime(s.split("To")[0].strip(), errors="coerce")
            return pd.NaT

        df["week_start"] = df["week"].apply(_parse_week_start)
        df = df.dropna(subset=["week_start"])

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        df = df[(df["week_start"] >= start) & (df["week_start"] <= end)]

        return df[["week_start", "street_footfall"]].reset_index(drop=True)

st.set_page_config(
    page_title="PFM Retail Performance Copilot",
    layout="wide"
)

# ----------------------
# OpenAI helper
# ----------------------
def _get_openai_client():
    if not _OPENAI_INSTALLED:
        return None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        api_key = None
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None

_OPENAI_CLIENT = _get_openai_client()

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
if VISUALCROSSING_KEY:
    os.environ["VISUALCROSSING_API_KEY"] = VISUALCROSSING_KEY

# ----------------------
# PFM brand colors
# ----------------------
PFM_PURPLE = "#762181"
PFM_RED = "#F04438"
PFM_PEACH = "#FDBA8C"
PFM_PINK = "#F973A6"
PFM_BLUE = "#38BDF8"
PFM_GRAY = "#6B7280"
PFM_LINE = "#E5E7EB"
PFM_DARK = "#111827"

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

# ----------------------
# Region mapping (same as Region tool)
# ----------------------
@st.cache_data(ttl=600)
def load_region_mapping(path: str = "data/regions.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=";")
    except Exception:
        return pd.DataFrame()

    if "shop_id" not in df.columns or "region" not in df.columns:
        return pd.DataFrame()

    df["shop_id"] = pd.to_numeric(df["shop_id"], errors="coerce").astype("Int64")
    df["region"] = df["region"].astype(str)

    if "sqm_override" in df.columns:
        df["sqm_override"] = pd.to_numeric(df["sqm_override"], errors="coerce")
    else:
        df["sqm_override"] = np.nan

    if "store_label" in df.columns:
        df["store_label"] = df["store_label"].astype(str)
    else:
        df["store_label"] = np.nan

    df = df.dropna(subset=["shop_id"])
    return df

# -------------
# API helpers
# -------------
@st.cache_data(ttl=600)
def get_locations_by_company(company_id: int) -> pd.DataFrame:
    url = f"{FASTAPI_BASE_URL.rstrip('/')}/company/{company_id}/location"

    last_err = None
    for attempt, timeout_s in enumerate([45, 120], start=1):
        try:
            resp = requests.get(url, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, dict) and "locations" in data:
                df = pd.DataFrame(data["locations"])
            else:
                df = pd.DataFrame(data)

            return df

        except requests.exceptions.ReadTimeout as e:
            last_err = e
            time.sleep(0.8 * attempt)
            continue

    raise requests.exceptions.ReadTimeout(
        f"Timeout bij ophalen locaties voor company {company_id} via {url}"
    ) from last_err

@st.cache_data(ttl=600)
def get_report(
    shop_ids,
    data_outputs,
    period: str,
    step: str = "day",
    source: str = "shops",
    company_id: int | None = None,
    form_date_from: str | None = None,
    form_date_to: str | None = None,
):
    params: list[tuple[str, str]] = []

    for sid in shop_ids:
        params.append(("data", str(sid)))

    for dout in data_outputs:
        params.append(("data_output", dout))

    params.append(("period", period))
    params.append(("step", step))
    params.append(("source", source))

    if period == "date" and form_date_from and form_date_to:
        params.append(("form_date_from", form_date_from))
        params.append(("form_date_to", form_date_to))

    resp = requests.post(REPORT_URL, params=params, timeout=90)
    resp.raise_for_status()
    return resp.json()

# -------------
# Weather helper
# -------------
@st.cache_data(ttl=3600)
def fetch_visualcrossing_history(location_str: str, start_date, end_date) -> pd.DataFrame:
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
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["week_start"] = df["date"].dt.to_period("W-SAT").dt.start_time

    agg_dict: dict[str, str] = {}
    if "footfall" in df.columns:
        agg_dict["footfall"] = "sum"
    if "turnover" in df.columns:
        agg_dict["turnover"] = "sum"
    if "sales_per_visitor" in df.columns:
        agg_dict["sales_per_visitor"] = "mean"
    if "conversion_rate" in df.columns:
        agg_dict["conversion_rate"] = "mean"

    if not agg_dict:
        return df[["week_start"]].drop_duplicates().reset_index(drop=True)

    return df.groupby("week_start", as_index=False).agg(agg_dict)

# -------------
# AI Store Coach helper
# -------------
def build_store_ai_coach_text(
    store_name: str,
    recent_foot: float,
    recent_turn: float | float,
    fut_foot: float,
    fut_turn: float | float,
    spv_cur: float | float,
    conv_cur: float | float,
    fc: pd.DataFrame,
) -> str:
    recent_foot = float(recent_foot or 0)
    fut_foot = float(fut_foot or 0)
    recent_turn = float(recent_turn or 0)
    fut_turn = float(fut_turn or 0)

    has_recent_foot = recent_foot > 0
    has_recent_turn = recent_turn > 0
    has_spv_cur = pd.notna(spv_cur) and (spv_cur or 0) > 0
    has_conv_cur = pd.notna(conv_cur) and (conv_cur or 0) > 0

    foot_msg = ""
    if has_recent_foot and fut_foot > 0:
        diff_foot = fut_foot - recent_foot
        diff_foot_pct = (diff_foot / recent_foot) * 100
        direction = "more" if diff_foot > 0 else "less"
        foot_msg = (
            f"- **Footfall outlook**: ~{diff_foot_pct:+.1f}% {direction} visitors vs last 14 days "
            f"(~{fmt_int(abs(diff_foot))} difference).\n"
        )
    elif fut_foot > 0:
        foot_msg = (
            f"- **Footfall outlook**: ~{fmt_int(fut_foot)} visitors expected in the next 14 days, "
            "but we lack enough history for a solid comparison.\n"
        )
    else:
        foot_msg = "- **Footfall outlook**: insufficient data for a reliable forecast.\n"

    omzet_msg = ""
    if has_recent_turn and fut_turn > 0:
        diff_turn = fut_turn - recent_turn
        diff_turn_pct = (diff_turn / recent_turn) * 100
        direction = "more" if diff_turn > 0 else "less"
        omzet_msg = (
            f"- **Turnover outlook**: ~{diff_turn_pct:+.1f}% {direction} vs last 14 days "
            f"(~{fmt_eur(abs(diff_turn))} difference).\n"
        )
    elif fut_turn > 0:
        omzet_msg = (
            f"- **Turnover outlook**: forecast turnover ~{fmt_eur(fut_turn)}, "
            "but we lack enough history for a solid comparison.\n"
        )
    else:
        omzet_msg = "- **Turnover outlook**: insufficient data for a reliable turnover forecast.\n"

    scenario_msg = ""
    if fut_foot > 0 and fut_turn > 0:
        spv_forecast = fut_turn / fut_foot
        uplift_pct = 5.0
        extra_turn_spv = fut_foot * spv_forecast * uplift_pct / 100.0
        scenario_msg += (
            f"- **Scenario SPV +{uplift_pct:.0f}%**: ~{fmt_eur(extra_turn_spv)} extra turnover "
            f"on top of the forecast (next 14 days).\n"
        )

    if fut_foot > 0 and has_spv_cur and has_conv_cur:
        conv_baseline = float(conv_cur)
        spv_baseline = float(spv_cur)
        atv_est = spv_baseline * 100.0 / conv_baseline
        conv_new = conv_baseline + 1.0
        extra_trans = fut_foot * (conv_new - conv_baseline) / 100.0
        extra_turn_conv = extra_trans * atv_est
        scenario_msg += (
            f"- **Scenario +1pp conversion**: ~{fmt_int(extra_trans)} extra transactions and "
            f"~{fmt_eur(extra_turn_conv)} extra turnover (next 14 days).\n"
        )

    peak_msg = ""
    if isinstance(fc, pd.DataFrame) and not fc.empty and "footfall_forecast" in fc.columns:
        fc_local = fc.copy()
        fc_local["date"] = pd.to_datetime(fc_local["date"], errors="coerce")
        fc_local = fc_local.dropna(subset=["date"])
        if not fc_local.empty:
            top_days = fc_local.sort_values("footfall_forecast", ascending=False).head(3)
            low_days = fc_local.sort_values("footfall_forecast", ascending=True).head(2)

            def _fmt_day(row):
                d = row["date"]
                return f"{d.strftime('%a %d-%m')} (~{fmt_int(row['footfall_forecast'])})"

            top_str = ", ".join(_fmt_day(r) for _, r in top_days.iterrows())
            low_str = ", ".join(_fmt_day(r) for _, r in low_days.iterrows())

            peak_msg = (
                f"- **Peak days to win**: {top_str}. Staff up + push active selling.\n"
                f"- **Quieter days to optimize**: {low_str}. Use for training + setup.\n"
            )

    action_msg = (
        "- **Focus for this period**: nail staffing on peak days + run one conversion/SPV experiment "
        "(greeting script, bundles, sharper promo placement). Share results back to the region team.\n"
    )

    header = "### AI Store Coach — next 14 days\n\n"
    intro = (
        f"For **{store_name}**, we combine recent performance with the next 14-day forecast. "
        "Here’s what to act on:\n\n"
    )

    return header + intro + foot_msg + omzet_msg + scenario_msg + peak_msg + action_msg

# -------------
# MAIN UI
# -------------
def main():
    st.title("PFM Retail Performance Copilot – Store Manager")

    clients = load_clients("clients.json")
    clients_df = pd.DataFrame(clients)
    clients_df["label"] = clients_df.apply(
        lambda r: f"{r['brand']} – {r['name']} (company_id {r['company_id']})",
        axis=1,
    )

    st.sidebar.header("Select retailer & store")

    client_label = st.sidebar.selectbox("Retailer", clients_df["label"].tolist())
    selected_client = clients_df[clients_df["label"] == client_label].iloc[0].to_dict()
    company_id = int(selected_client["company_id"])

    # --- Locations ---
    try:
        locations_df = get_locations_by_company(company_id)
    except requests.exceptions.ReadTimeout:
        st.error(
            "FastAPI location endpoint timed out (cold start / many locations). "
            "Click **Analyse** again or try later."
        )
        st.stop()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching locations: {e}")
        st.stop()

    if locations_df.empty:
        st.error("No stores found for this retailer.")
        return

    if "name" not in locations_df.columns:
        locations_df["name"] = locations_df["id"].astype(str)

    locations_df["label"] = locations_df.apply(
        lambda r: f"{r['name']} (ID: {r['id']})", axis=1
    )

    shop_label = st.sidebar.selectbox("Store", locations_df["label"].tolist())
    shop_row = locations_df[locations_df["label"] == shop_label].iloc[0].to_dict()
    shop_id = int(shop_row["id"])
    sqm = float(shop_row.get("sqm", 0) or 0)
    postcode = shop_row.get("zip") or shop_row.get("postcode", "")
    lat = float(shop_row.get("lat", 0) or 0)
    lon = float(shop_row.get("lon", 0) or 0)

    # --- Period selector ---
    period_choice = st.sidebar.selectbox(
        "Period",
        ["This week", "Last week", "This month", "Last month", "This quarter", "Last quarter"],
        index=2,
    )

    today = datetime.today().date()

    default_hist_start = today - timedelta(days=365)
    hist_start_input = st.sidebar.date_input(
        "Forecast history from",
        value=default_hist_start,
        help="This date controls how much history we use to train the forecast model.",
    )

    def get_week_range(base_date):
        wd = base_date.weekday()
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

    if period_choice == "This week":
        start_cur, end_cur = get_week_range(today)
        start_prev, end_prev = start_cur - timedelta(days=7), start_cur - timedelta(days=1)

    elif period_choice == "Last week":
        this_week_start, _ = get_week_range(today)
        end_cur = this_week_start - timedelta(days=1)
        start_cur = end_cur - timedelta(days=6)
        start_prev = start_cur - timedelta(days=7)
        end_prev = start_cur - timedelta(days=1)

    elif period_choice == "This month":
        start_cur, end_cur = get_month_range(today.year, today.month)
        if today.month == 1:
            prev_y, prev_m = today.year - 1, 12
        else:
            prev_y, prev_m = today.year, today.month - 1
        start_prev, end_prev = get_month_range(prev_y, prev_m)

    elif period_choice == "Last month":
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

    elif period_choice == "This quarter":
        start_cur, end_cur = get_quarter_range(today.year, today.month)
        cur_q = (today.month - 1) // 3 + 1
        if cur_q == 1:
            prev_y, prev_q = today.year - 1, 4
        else:
            prev_y, prev_q = today.year, cur_q - 1
        prev_start_month = 3 * (prev_q - 1) + 1
        start_prev, end_prev = get_quarter_range(prev_y, prev_start_month)

    else:  # Last quarter
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

    default_city = shop_row.get("city") or "Amsterdam"
    weather_location = st.sidebar.text_input("Weather location", value=f"{default_city},NL")
    postcode4 = st.sidebar.text_input("CBS postcode (4 digits)", value=postcode[:4] if postcode else "")

    forecast_mode = st.sidebar.radio(
        "Forecast mode",
        ["Simple (DoW)", "Pro (LightGBM beta)"],
        index=0,
    )

    run_btn = st.sidebar.button("Analyse", type="primary")

    if not run_btn:
        st.info("Select retailer & store, pick a period and click **Analyse**.")
        return

    # --- Fetch store data (this year, then slice locally) ---
    with st.spinner("Fetching data via FastAPI..."):
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
        df_all_raw = normalize_vemcount_response(resp_all, kpi_keys=metric_map.keys()).rename(columns=metric_map)

    if df_all_raw.empty:
        st.warning("No data found for this year for this store.")
        return

    df_all_raw["date"] = pd.to_datetime(df_all_raw["date"], errors="coerce")
    df_all_raw = df_all_raw.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # --- Forecast history (date range) ---
    hist_end = datetime.today().date()
    hist_start = min(hist_start_input, hist_end)

    try:
        resp_hist = get_report(
            [shop_id],
            list(metric_map.keys()),
            period="date",
            step="day",
            source="shops",
            form_date_from=hist_start.strftime("%Y-%m-%d"),
            form_date_to=hist_end.strftime("%Y-%m-%d"),
        )
        df_hist_raw = normalize_vemcount_response(resp_hist, kpi_keys=metric_map.keys()).rename(columns=metric_map)
        df_hist_raw["date"] = pd.to_datetime(df_hist_raw["date"], errors="coerce")
        df_hist_raw = df_hist_raw.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        if df_hist_raw.empty:
            df_hist_raw = df_all_raw.copy()
    except Exception:
        df_hist_raw = df_all_raw.copy()

    start_cur_ts = pd.Timestamp(start_cur)
    end_cur_ts = pd.Timestamp(end_cur)
    start_prev_ts = pd.Timestamp(start_prev)
    end_prev_ts = pd.Timestamp(end_prev)

    df_cur = df_all_raw[(df_all_raw["date"] >= start_cur_ts) & (df_all_raw["date"] <= end_cur_ts)].copy()
    df_prev = df_all_raw[(df_all_raw["date"] >= start_prev_ts) & (df_all_raw["date"] <= end_prev_ts)].copy()

    if df_cur.empty and df_prev.empty:
        st.warning("No data found in the selected periods.")
        return

    df_cur = compute_daily_kpis(df_cur)
    if not df_prev.empty:
        df_prev = compute_daily_kpis(df_prev)

    # --- Weather ---
    weather_df = pd.DataFrame()
    if weather_location and VISUALCROSSING_KEY:
        weather_df = fetch_visualcrossing_history(weather_location, start_cur, end_cur)

    # --- Pathzz demo (weekly) ---
    pathzz_weekly = fetch_monthly_street_traffic(start_date=start_prev, end_date=end_cur)

    capture_weekly = pd.DataFrame()
    avg_capture_cur = None
    avg_capture_prev = None

    if not pathzz_weekly.empty:
        df_range = df_all_raw[(df_all_raw["date"] >= start_prev_ts) & (df_all_raw["date"] <= end_cur_ts)].copy()
        df_range = compute_daily_kpis(df_range)
        store_weekly = aggregate_weekly(df_range)

        pathzz_weekly["week_start"] = pd.to_datetime(pathzz_weekly["week_start"])
        street_weekly = pathzz_weekly.groupby("week_start", as_index=False)["street_footfall"].mean()

        capture_weekly = pd.merge(
            store_weekly[["week_start", "footfall", "turnover"]],
            street_weekly,
            on="week_start",
            how="inner",
        )

        if not capture_weekly.empty:
            capture_weekly["capture_rate"] = np.where(
                capture_weekly["street_footfall"] > 0,
                capture_weekly["footfall"] / capture_weekly["street_footfall"] * 100,
                np.nan,
            )

            capture_weekly = capture_weekly.sort_values("week_start")
            capture_weekly["period"] = np.where(
                (capture_weekly["week_start"] >= start_cur_ts) & (capture_weekly["week_start"] <= end_cur_ts),
                "current",
                "previous",
            )

            avg_capture_cur = capture_weekly.loc[capture_weekly["period"] == "current", "capture_rate"].mean()
            avg_capture_prev = capture_weekly.loc[capture_weekly["period"] == "previous", "capture_rate"].mean()

    # --- CBS context ---
    cbs_stats = {}
    if postcode4:
        cbs_stats = get_cbs_stats_for_postcode4(postcode4)

    # --- KPI cards ---
    st.subheader(f"{selected_client['brand']} – {shop_row['name']}")

    foot_cur = df_cur["footfall"].sum() if "footfall" in df_cur.columns else 0
    foot_prev = df_prev["footfall"].sum() if ("footfall" in df_prev.columns and not df_prev.empty) else 0
    foot_delta = None
    if foot_prev > 0:
        foot_delta = f"{(foot_cur - foot_prev) / foot_prev * 100:+.1f}%"

    turn_cur = df_cur["turnover"].sum() if "turnover" in df_cur.columns else 0
    turn_prev = df_prev["turnover"].sum() if ("turnover" in df_prev.columns and not df_prev.empty) else 0
    turn_delta = None
    if turn_prev > 0:
        turn_delta = f"{(turn_cur - turn_prev) / turn_prev * 100:+.1f}%"

    spv_cur = df_cur["sales_per_visitor"].mean() if "sales_per_visitor" in df_cur.columns else np.nan
    spv_prev = df_prev["sales_per_visitor"].mean() if ("sales_per_visitor" in df_prev.columns and not df_prev.empty) else np.nan
    spv_delta = None
    if pd.notna(spv_cur) and pd.notna(spv_prev) and spv_prev > 0:
        spv_delta = f"{(spv_cur - spv_prev) / spv_prev * 100:+.1f}%"

    conv_cur = df_cur["conversion_rate"].mean() if "conversion_rate" in df_cur.columns else np.nan
    conv_prev = df_prev["conversion_rate"].mean() if ("conversion_rate" in df_prev.columns and not df_prev.empty) else np.nan
    conv_delta = None
    if pd.notna(conv_cur) and pd.notna(conv_prev) and conv_prev > 0:
        conv_delta = f"{(conv_cur - conv_prev) / conv_prev * 100:+.1f}%"

    # ---------------------------
    # ✅ SVI (reuse services/svi_service.py)
    # ---------------------------
    store_svi_score = None
    store_svi_rank = None
    store_svi_peer_n = None
    store_region = None

    region_map = load_region_mapping()

    if not region_map.empty:
        region_map["shop_id"] = pd.to_numeric(region_map["shop_id"], errors="coerce").astype("Int64")
        this_map_row = region_map[region_map["shop_id"] == shop_id]
        if not this_map_row.empty:
            store_region = str(this_map_row["region"].iloc[0])

    if store_region:
        # peers = all shops in same region (for this retailer)
        region_peer_ids = region_map[region_map["region"].astype(str) == store_region]["shop_id"].dropna().astype(int).unique().tolist()

        # limit peers to stores that exist in locations_df (this company)
        loc_ids = pd.to_numeric(locations_df["id"], errors="coerce").dropna().astype(int).unique().tolist()
        peer_ids = [sid for sid in region_peer_ids if sid in set(loc_ids)]
        if shop_id not in peer_ids:
            peer_ids.append(shop_id)

        # pull peers data for selected current period range
        try:
            resp_peers = get_report(
                peer_ids,
                ["count_in", "turnover"],
                period="date",
                step="day",
                source="shops",
                form_date_from=start_cur.strftime("%Y-%m-%d"),
                form_date_to=end_cur.strftime("%Y-%m-%d"),
            )
            df_peers = normalize_vemcount_response(resp_peers, kpi_keys=["count_in", "turnover"]).rename(
                columns={"count_in": "footfall", "turnover": "turnover"}
            )
            if not df_peers.empty:
                df_peers["date"] = pd.to_datetime(df_peers["date"], errors="coerce")
                df_peers = df_peers.dropna(subset=["date"])

                # detect store key col
                store_key_col = None
                for cand in ["id", "shop_id", "location_id"]:
                    if cand in df_peers.columns:
                        store_key_col = cand
                        break

                if store_key_col:
                    # add sqm_effective to df_peers using locations + override
                    loc_meta = locations_df.copy()
                    loc_meta["id"] = pd.to_numeric(loc_meta["id"], errors="coerce").astype("Int64")
                    loc_meta["sqm"] = pd.to_numeric(loc_meta.get("sqm", np.nan), errors="coerce")

                    # region overrides for sqm / labels
                    reg = region_map.copy()
                    reg["shop_id"] = pd.to_numeric(reg["shop_id"], errors="coerce").astype("Int64")
                    reg["sqm_override"] = pd.to_numeric(reg.get("sqm_override", np.nan), errors="coerce")

                    meta = loc_meta.merge(reg, left_on="id", right_on="shop_id", how="left")

                    meta["sqm_effective"] = np.where(
                        meta["sqm_override"].notna(),
                        meta["sqm_override"],
                        meta["sqm"],
                    )

                    meta["store_display"] = np.where(
                        meta.get("store_label", pd.Series([np.nan] * len(meta))).notna(),
                        meta["store_label"],
                        meta.get("name", meta["id"].astype(str)),
                    )

                    # df_peers join
                    df_peers[store_key_col] = pd.to_numeric(df_peers[store_key_col], errors="coerce").astype("Int64")
                    df_peers = df_peers.merge(
                        meta[["id", "store_display", "sqm_effective"]],
                        left_on=store_key_col,
                        right_on="id",
                        how="left",
                    )
                    df_peers["sqm_effective"] = pd.to_numeric(df_peers["sqm_effective"], errors="coerce")

                    # compute KPIs needed by SVI service
                    df_peers = compute_daily_kpis(df_peers)

                    # build vitality (one row per store)
                    svi_df = build_store_vitality(
                        df_period=df_peers,
                        region_shops=meta.rename(columns={"id": "id"}),  # expects 'id' and 'store_display'
                        store_key_col=store_key_col,
                    )

                    if svi_df is not None and not svi_df.empty:
                        # rank inside region peers
                        svi_df["svi_score"] = pd.to_numeric(svi_df["svi_score"], errors="coerce")
                        svi_df = svi_df.dropna(subset=["svi_score"]).sort_values("svi_score", ascending=False).reset_index(drop=True)
                        svi_df["rank"] = np.arange(1, len(svi_df) + 1)

                        row = svi_df[svi_df[store_key_col].astype("Int64") == pd.Series([shop_id], dtype="Int64").iloc[0]]
                        if not row.empty:
                            store_svi_score = float(np.clip(row["svi_score"].iloc[0], 0, 100))
                            store_svi_rank = int(row["rank"].iloc[0])
                            store_svi_peer_n = int(len(svi_df))

        except Exception:
            # keep SVI as None (UI will show placeholders)
            pass

    # ---------------------------
    # Row: KPI cards (now includes SVI card)
    # ---------------------------
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1.2])

    with col1:
        st.metric("Footfall (period)", fmt_int(foot_cur), delta=foot_delta)
    with col2:
        st.metric("Turnover (period)", fmt_eur(turn_cur), delta=turn_delta)
    with col3:
        value = f"€ {spv_cur:.2f}".replace(".", ",") if pd.notna(spv_cur) else "-"
        st.metric("Avg spend / visitor", value, delta=spv_delta)
    with col4:
        if avg_capture_cur is not None and not pd.isna(avg_capture_cur) and avg_capture_prev not in (None, 0):
            delta_val = None
            if avg_capture_prev and avg_capture_prev > 0:
                delta_val = (avg_capture_cur - avg_capture_prev) / avg_capture_prev * 100
            st.metric(
                "Avg capture rate",
                fmt_pct(avg_capture_cur),
                delta=f"{delta_val:+.1f}%" if delta_val is not None else None,
            )
        else:
            st.metric(
                "Avg conversion",
                fmt_pct(conv_cur) if pd.notna(conv_cur) else "-",
                delta=conv_delta,
            )

    with col5:
        # ✅ SVI Scorecard (exactly what you asked)
        st.markdown(
            f"""
            <div style="border:1px solid {PFM_LINE}; border-radius:14px; padding:0.85rem 1rem; background:white;">
              <div style="color:{PFM_GRAY}; font-size:0.85rem; font-weight:700;">Store Vitality (SVI)</div>
              <div style="color:{PFM_DARK}; font-size:1.45rem; font-weight:900; margin-top:0.2rem;">
                {f"{store_svi_score:.0f}" if store_svi_score is not None else "—"} / 100
              </div>
              <div style="color:{PFM_GRAY}; font-size:0.85rem; margin-top:0.25rem;">
                Rank: {f"{store_svi_rank} / {store_svi_peer_n}" if (store_svi_rank and store_svi_peer_n) else "—"}
                · benchmark vs regional peers
                {f" · Region: {store_region}" if store_region else ""}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Weekly chart: street vs store + turnover + capture ---
    st.markdown("### Street traffic vs store traffic (weekly demo)")

    if not capture_weekly.empty:
        chart_df = capture_weekly[["week_start", "footfall", "street_footfall", "turnover", "capture_rate"]].copy()
        iso_cal = chart_df["week_start"].dt.isocalendar()
        chart_df["week_label"] = iso_cal.week.apply(lambda w: f"W{int(w):02d}")
        week_order = chart_df.sort_values("week_start")["week_label"].unique().tolist()

        fig_week = make_subplots(specs=[[{"secondary_y": True}]])

        fig_week.add_bar(x=chart_df["week_label"], y=chart_df["footfall"], name="Footfall (store)", marker_color=PFM_PURPLE, offsetgroup=0)
        fig_week.add_bar(x=chart_df["week_label"], y=chart_df["street_footfall"], name="Street traffic", marker_color=PFM_PEACH, opacity=0.7, offsetgroup=1)
        fig_week.add_bar(x=chart_df["week_label"], y=chart_df["turnover"], name="Turnover (€)", marker_color=PFM_PINK, opacity=0.7, offsetgroup=2)

        fig_week.add_trace(
            go.Scatter(x=chart_df["week_label"], y=chart_df["capture_rate"], name="Capture rate (%)", mode="lines+markers",
                       line=dict(color=PFM_RED, width=2)),
            secondary_y=True,
        )

        fig_week.update_xaxes(title_text="Week", categoryorder="array", categoryarray=week_order)
        fig_week.update_yaxes(title_text="Footfall / street / turnover", secondary_y=False)
        fig_week.update_yaxes(title_text="Capture rate (%)", secondary_y=True)
        fig_week.update_layout(barmode="group", height=350,
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                               margin=dict(l=40, r=40, t=40, b=40))

        st.plotly_chart(fig_week, use_container_width=True)
    else:
        st.info("No Pathzz weekly demo data available for this period.")

    # --- Daily chart ---
    st.markdown("### Daily footfall & turnover")
    if "footfall" in df_cur.columns and "turnover" in df_cur.columns:
        daily_df = df_cur[["date", "footfall", "turnover"]].copy()

        fig_daily = make_subplots(specs=[[{"secondary_y": True}]])
        fig_daily.add_bar(x=daily_df["date"], y=daily_df["footfall"], name="Footfall", marker_color=PFM_PURPLE)

        fig_daily.add_trace(
            go.Scatter(x=daily_df["date"], y=daily_df["turnover"], name="Turnover (€)",
                       mode="lines+markers", line=dict(color=PFM_PEACH, width=2)),
            secondary_y=True,
        )

        fig_daily.update_yaxes(title_text="Footfall", secondary_y=False)
        fig_daily.update_yaxes(title_text="Turnover (€)", secondary_y=True)
        fig_daily.update_layout(height=350,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                                margin=dict(l=40, r=40, t=20, b=40))
        st.plotly_chart(fig_daily, use_container_width=True)

    # --- Weather vs footfall ---
    if not weather_df.empty:
        st.markdown("### Weather vs footfall (indicative)")

        weather_merge = pd.merge(
            df_cur[["date", "footfall"]],
            weather_df[["date", "temp", "precip"]],
            on="date",
            how="left",
        )

        fig_weather = make_subplots(specs=[[{"secondary_y": True}]])
        fig_weather.add_bar(x=weather_merge["date"], y=weather_merge["footfall"], name="Footfall", marker_color=PFM_PURPLE)

        fig_weather.add_trace(
            go.Scatter(x=weather_merge["date"], y=weather_merge["temp"], name="Temp (°C)", mode="lines+markers",
                       line=dict(color=PFM_RED, width=2)),
            secondary_y=True,
        )

        fig_weather.add_trace(
            go.Scatter(x=weather_merge["date"], y=weather_merge["precip"], name="Precip (mm)", mode="lines+markers",
                       line=dict(color=PFM_BLUE, width=2, dash="dot")),
            secondary_y=True,
        )

        fig_weather.update_yaxes(title_text="Footfall", secondary_y=False)
        fig_weather.update_yaxes(title_text="Temp / precip", secondary_y=True)
        fig_weather.update_layout(height=350,
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                                  margin=dict(l=40, r=40, t=20, b=40))
        st.plotly_chart(fig_weather, use_container_width=True)

    # --- Forecast ---
    st.markdown("### Forecast: footfall & turnover (next 14 days)")

    weather_cfg = None
    if VISUALCROSSING_KEY and weather_location:
        parts = weather_location.split(",")
        city_part = parts[0].strip() if parts else weather_location
        country = "Netherlands"
        if len(parts) >= 2:
            cc = parts[1].strip().upper()
            country_map = {"NL": "Netherlands", "BE": "Belgium", "DE": "Germany", "FR": "France", "UK": "United Kingdom", "GB": "United Kingdom"}
            country = country_map.get(cc, cc)

        weather_cfg = {"mode": "city_country", "city": city_part, "country": country, "api_key": VISUALCROSSING_KEY}

    try:
        if forecast_mode == "Pro (LightGBM beta)":
            fc_res = build_pro_footfall_turnover_forecast(
                df_hist_raw,
                horizon=14,
                min_history_days=60,
                weather_cfg=weather_cfg,
                use_weather=bool(weather_cfg),
            )
        else:
            fc_res = build_simple_footfall_turnover_forecast(df_hist_raw)

        if not fc_res.get("enough_history", False):
            st.info("Not enough historical data for a reliable forecast.")
        else:
            hist_recent = fc_res["hist_recent"]
            fc = fc_res["forecast"]

            recent_foot = fc_res["recent_footfall_sum"]
            recent_turn = fc_res["recent_turnover_sum"]
            fut_foot = fc_res["forecast_footfall_sum"]
            fut_turn = fc_res["forecast_turnover_sum"]

            c_model = st.columns([3, 1])[1]
            with c_model:
                st.caption(f"Model: **{fc_res['model_type']}**")
                if fc_res.get("used_simple_fallback", False):
                    st.caption("Fallback → Simple DoW")

            c1, c2 = st.columns(2)
            with c1:
                delta_foot = f"{(fut_foot - recent_foot) / recent_foot * 100:+.1f}%" if recent_foot > 0 else None
                st.metric("Expected visitors (14d)", fmt_int(fut_foot), delta=delta_foot)

            with c2:
                delta_turn = f"{(fut_turn - recent_turn) / recent_turn * 100:+.1f}%" if (not pd.isna(recent_turn) and recent_turn > 0) else None
                st.metric("Expected turnover (14d)", fmt_eur(fut_turn), delta=delta_turn)

            fig_fc = make_subplots(specs=[[{"secondary_y": True}]])
            fig_fc.add_bar(x=hist_recent["date"], y=hist_recent["footfall"], name="Footfall (hist)", marker_color=PFM_PURPLE)
            fig_fc.add_bar(x=fc["date"], y=fc["footfall_forecast"], name="Footfall (fc)", marker_color=PFM_PEACH)

            if "turnover" in hist_recent.columns:
                fig_fc.add_trace(
                    go.Scatter(x=hist_recent["date"], y=hist_recent["turnover"], name="Turnover (hist)", mode="lines",
                               line=dict(color=PFM_PINK, width=2)),
                    secondary_y=True,
                )

            fig_fc.add_trace(
                go.Scatter(x=fc["date"], y=fc["turnover_forecast"], name="Turnover (fc)", mode="lines+markers",
                           line=dict(color=PFM_RED, width=2, dash="dash")),
                secondary_y=True,
            )

            fig_fc.update_yaxes(title_text="Footfall", secondary_y=False)
            fig_fc.update_yaxes(title_text="Turnover (€)", secondary_y=True)
            fig_fc.update_layout(height=350,
                                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                                 margin=dict(l=40, r=40, t=20, b=40))
            st.plotly_chart(fig_fc, use_container_width=True)

            coach_text = build_store_ai_coach_text(
                store_name=shop_row["name"],
                recent_foot=recent_foot,
                recent_turn=recent_turn,
                fut_foot=fut_foot,
                fut_turn=fut_turn,
                spv_cur=spv_cur,
                conv_cur=conv_cur,
                fc=fc,
            )
            st.markdown(coach_text)

    except Exception as e:
        st.info("Forecast could not be computed (missing data / model / weather issue).")
        st.exception(e)

    # --- Debug ---
    with st.expander("🔧 Debug"):
        st.write("Period choice:", period_choice)
        st.write("Current period:", start_cur, "→", end_cur)
        st.write("Previous period:", start_prev, "→", end_prev)
        st.write("Shop row:", shop_row)
        st.write("Store region:", store_region)
        st.write("SVI score:", store_svi_score, "Rank:", store_svi_rank, "Peers:", store_svi_peer_n)
        st.write("All daily (head):", df_all_raw.head())
        st.write("Current df (head):", df_cur.head())
        st.write("Prev df (head):", df_prev.head())
        st.write("Pathzz weekly (head):", pathzz_weekly.head())
        st.write("Capture weekly (head):", capture_weekly.head() if isinstance(capture_weekly, pd.DataFrame) else capture_weekly)
        st.write("CBS stats:", cbs_stats)
        st.write("Weather df (head):", weather_df.head())
        try:
            st.write("Forecast model_type:", fc_res.get("model_type"))
            st.write("Forecast used_simple_fallback:", fc_res.get("used_simple_fallback"))
            st.write("Forecast head:", fc_res["forecast"].head())
        except Exception:
            st.write("Forecast object not available in this run.")

if __name__ == "__main__":
    main()
