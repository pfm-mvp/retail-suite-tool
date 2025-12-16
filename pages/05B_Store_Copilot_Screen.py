# pages/05_Retail_AI_Store_Copilot.py

import os
import time  # ✅ for retry backoff
import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt  # eventueel later nog voor andere grafieken
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# OpenAI (optional)
try:
    from openai import OpenAI
    _OPENAI_INSTALLED = True
except Exception:
    OpenAI = None
    _OPENAI_INSTALLED = False

from services.event_service import build_event_flags_for_dates
from datetime import datetime, timedelta

from helpers_clients import load_clients
from helpers_normalize import normalize_vemcount_response
from services.cbs_service import get_cbs_stats_for_postcode4
from services.forecast_service import (
    build_simple_footfall_turnover_forecast,
    build_pro_footfall_turnover_forecast,
)

# Pathzz service import with fallback
try:
    from services.pathzz_service import fetch_monthly_street_traffic  # gebruikt sample weekly CSV
except Exception:
    def fetch_monthly_street_traffic(start_date, end_date):
        """
        Fallback: read demo street traffic from data/pathzz_sample_weekly.csv

        CSV structure:
        Week;Visits
        2025-10-05 To 2025-10-11;11.613  (→ 11613)
        ...
        Return columns:
        - week_start (datetime)
        - street_footfall (float)
        """
        csv_path = "data/pathzz_sample_weekly.csv"
        try:
            df = pd.read_csv(csv_path, sep=";", dtype={"Visits": "string"})
        except Exception:
            return pd.DataFrame()

        df = df.rename(columns={"Week": "week", "Visits": "street_footfall"})

        df["street_footfall"] = (
            df["street_footfall"]
            .astype(str)
            .str.replace(".", "", regex=False)   # dot as thousand separator
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

# ----------------------
# UI (new layout / alignment)
# ----------------------
PFM_DARK = "#0F172A"
PFM_GRAY = "#6B7280"
PFM_LINE = "#E5E7EB"
PFM_BG = "#F8FAFC"

st.markdown(
    f"""
    <style>
      .block-container {{
        padding-top: 0.9rem;
        padding-bottom: 2rem;
        max-width: 1320px;
      }}

      /* Header */
      .pfm-header {{
        display:flex;
        align-items:flex-end;
        justify-content:space-between;
        gap: 1rem;
        padding: 0.85rem 1rem;
        border: 1px solid {PFM_LINE};
        border-radius: 16px;
        background: white;
        margin-bottom: 0.65rem;
      }}
      .pfm-title {{
        font-size: 1.35rem;
        font-weight: 900;
        color: {PFM_DARK};
        line-height: 1.15;
      }}
      .pfm-sub {{
        color: {PFM_GRAY};
        font-size: 0.92rem;
        margin-top: 0.2rem;
      }}

      /* Panels */
      .panel {{
        border: 1px solid {PFM_LINE};
        border-radius: 16px;
        background: white;
        padding: 0.85rem 1rem;
        margin-top: 0.35rem;
      }}
      .panel-title {{
        font-weight: 900;
        color: {PFM_DARK};
        margin-bottom: 0.55rem;
      }}
      .panel-sub {{
        color: {PFM_GRAY};
        font-size: 0.85rem;
        margin-top: -0.2rem;
        margin-bottom: 0.35rem;
      }}

      /* KPI cards */
      .kpi-card {{
        border: 1px solid {PFM_LINE};
        border-radius: 16px;
        background: white;
        padding: 0.9rem 1rem;
        height: 100%;
      }}
      .kpi-label {{
        color: {PFM_GRAY};
        font-size: 0.86rem;
        font-weight: 700;
      }}
      .kpi-value {{
        color: {PFM_DARK};
        font-size: 1.55rem;
        font-weight: 950;
        margin-top: 0.25rem;
        letter-spacing: -0.02em;
      }}
      .kpi-delta {{
        margin-top: 0.15rem;
        font-size: 0.88rem;
        font-weight: 800;
      }}
      .kpi-help {{
        color: {PFM_GRAY};
        font-size: 0.82rem;
        margin-top: 0.25rem;
      }}

      /* Chips */
      .chip {{
        display:inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        font-size: 0.80rem;
        font-weight: 800;
        border: 1px solid {PFM_LINE};
        color: {PFM_DARK};
        background: {PFM_BG};
        margin-right: 0.35rem;
        margin-bottom: 0.25rem;
      }}

      /* Buttons */
      div.stButton > button {{
        background: {PFM_RED} !important;
        color: white !important;
        border: 0px !important;
        border-radius: 12px !important;
        padding: 0.65rem 1rem !important;
        font-weight: 900 !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi_card(label: str, value: str, delta: str | None = None, help_text: str = "", delta_color: str | None = None):
    if delta is None:
        delta_html = ""
    else:
        color = delta_color or PFM_PURPLE
        delta_html = f'<div class="kpi-delta" style="color:{color};">{delta}</div>'

    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          {delta_html}
          <div class="kpi-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_start(title: str, sub: str = ""):
    st.markdown(f'<div class="panel"><div class="panel-title">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="panel-sub">{sub}</div>', unsafe_allow_html=True)


def panel_end():
    st.markdown("</div>", unsafe_allow_html=True)


def chip(text: str):
    st.markdown(f'<span class="chip">{text}</span>', unsafe_allow_html=True)


def fmt_score_100(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "— / 100"
    return f"{int(round(float(x)))} / 100"


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
    ✅ Robust wrapper around /company/{company_id}/location

    - Render/cold start friendly
    - Retry on ReadTimeout
    - Longer timeouts
    """
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
        f"Timeout while fetching locations for company {company_id} via {url}"
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
    """
    Wrapper around /get-report (POST), with query params WITHOUT [].
    """
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

    resp = requests.post(REPORT_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


# -------------
# Weather helper (Visual Crossing direct for chart)
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


def aggregate_monthly(df: pd.DataFrame, sqm: float | None) -> pd.DataFrame:
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
# AI Store Coach helper (existing)
# -------------

def build_ai_store_coach_text(
    shop_name: str,
    brand: str,
    fc_res: dict,
    events_future_df: pd.DataFrame | None = None,
    weather_cfg: dict | None = None,
) -> str:
    recent_foot = fc_res.get("recent_footfall_sum", 0.0) or 0.0
    fut_foot = fc_res.get("forecast_footfall_sum", 0.0) or 0.0
    recent_turn = fc_res.get("recent_turnover_sum", np.nan)
    fut_turn = fc_res.get("forecast_turnover_sum", np.nan)

    model_type = fc_res.get("model_type", "unknown")
    used_fallback = fc_res.get("used_simple_fallback", False)

    event_lines = []
    if events_future_df is not None and not events_future_df.empty:
        ef = events_future_df.copy()
        ef["date"] = pd.to_datetime(ef["date"]).dt.date
        for _, row in ef.iterrows():
            labels = []
            if row.get("is_black_friday", 0) == 1:
                labels.append("Black Friday-like moment")
            if row.get("is_december_trade", 0) == 1:
                labels.append("December peak")
            if row.get("is_summer_sale", 0) == 1:
                labels.append("Summer sale period")
            if row.get("is_school_holiday", 0) == 1:
                labels.append("School holiday")
            if labels:
                event_lines.append(f"- {row['date']}: " + ", ".join(labels))

    base_summary = (
        f"Store: {shop_name} (brand: {brand}). "
        f"Last 14 days footfall: {recent_foot:.0f}, forecast next 14 days: {fut_foot:.0f}. "
    )
    if not pd.isna(recent_turn) and not pd.isna(fut_turn):
        base_summary += f"Turnover last 14 days ~ {recent_turn:.0f}, expected ~ {fut_turn:.0f}. "

    if used_fallback:
        base_summary += "Model fell back to simple DoW. "
    else:
        base_summary += f"Modeltype: {model_type}. "

    if weather_cfg:
        base_summary += f"Weather location: {weather_cfg.get('city','?')}, {weather_cfg.get('country','?')}. "

    if event_lines:
        base_summary += "Key upcoming days:\n" + "\n".join(event_lines)

    if _OPENAI_CLIENT:
        prompt = f"""
You are a retail performance coach for a store manager.

Data:
{base_summary}

Give max 5 bullets with concrete actions:
- staffing around busy days,
- leverage peak moments (events),
- ideas to improve conversion / SPV,
- 1 bullet: how to report back to the regional manager.

Write in English, practical and to the point.
"""
        try:
            completion = _OPENAI_CLIENT.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an experienced retail operations coach."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
            )
            return completion.choices[0].message.content.strip()
        except Exception:
            return (
                "AI Store Coach could not be called. "
                "Basic guidance: staff up for peak days, test one conversion improvement, "
                "and report the outcome briefly to your regional manager."
            )

    tips = [
        "• Check the 2–3 busiest forecast days and schedule extra floor coverage.",
        "• Run one focused test to lift conversion (greeting script, better routing, faster checkout).",
        "• Use the forecast turnover as a day target guideline in your morning huddle.",
        "• Look for days where footfall is high but turnover/SPV lags — those are your fastest wins.",
        "• Share a short summary (3 bullets) with your regional manager on what you will test in the next 2 weeks.",
    ]
    if event_lines:
        tips.insert(1, "• Around special days (events/holidays), plan a small in-store activation (promo/demo/social post).")
    return "\n".join(tips)


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
            f"- **Footfall trend**: expect about {diff_foot_pct:+.1f}% {direction} visitors "
            f"vs the last 14 days (~{fmt_int(abs(diff_foot))} difference).\n"
        )
    elif fut_foot > 0:
        foot_msg = (
            f"- **Footfall trend**: forecast shows ~{fmt_int(fut_foot)} visitors for the next 14 days, "
            "but there is not enough history for a strong comparison.\n"
        )
    else:
        foot_msg = "- **Footfall trend**: insufficient data to create a reliable visitor forecast.\n"

    turnover_msg = ""
    scenario_msg = ""

    if has_recent_turn and fut_turn > 0:
        diff_turn = fut_turn - recent_turn
        diff_turn_pct = (diff_turn / recent_turn) * 100
        direction = "more" if diff_turn > 0 else "less"
        turnover_msg = (
            f"- **Turnover expectation**: forecast is about {diff_turn_pct:+.1f}% {direction} "
            f"vs the last 14 days (~{fmt_eur(abs(diff_turn))} difference).\n"
        )
    elif fut_turn > 0:
        turnover_msg = (
            f"- **Turnover expectation**: forecast turnover is ~{fmt_eur(fut_turn)}, "
            "but there is not enough history for a strong comparison.\n"
        )
    else:
        turnover_msg = "- **Turnover expectation**: insufficient data to show a reliable turnover forecast.\n"

    if fut_foot > 0 and fut_turn > 0:
        spv_forecast = fut_turn / fut_foot
        uplift_pct = 5.0
        extra_turn_spv = fut_foot * spv_forecast * uplift_pct / 100.0
        scenario_msg += (
            f"- **Scenario SPV +{uplift_pct:.0f}%**: lifting SPV by ~{uplift_pct:.0f}% over the next 14 days "
            f"adds ~{fmt_eur(extra_turn_spv)} on top of the forecast.\n"
        )

    if fut_foot > 0 and has_spv_cur and has_conv_cur:
        conv_baseline = float(conv_cur)
        spv_baseline = float(spv_cur)
        atv_est = spv_baseline * 100.0 / conv_baseline
        conv_new = conv_baseline + 1.0
        extra_trans = fut_foot * (conv_new - conv_baseline) / 100.0
        extra_turn_conv = extra_trans * atv_est

        scenario_msg += (
            f"- **Scenario conversion +1pp**: improving conversion from {conv_baseline:.1f}% to {conv_new:.1f}% "
            f"generates ~{fmt_int(extra_trans)} extra transactions and ~{fmt_eur(extra_turn_conv)} extra turnover "
            f"(assuming stable basket size).\n"
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
                f"- **Use peak moments**: busiest forecast days are {top_str}. Ensure strong staffing + active selling.\n"
                f"- **Use quiet moments**: quieter days are {low_str}. Use for training, layout improvements, preparation.\n"
            )

    action_msg = (
        "- **Focus**: combine sharp staffing on peak days with 1–2 clear actions to lift SPV and conversion "
        "(greeting, bundles, clarity at top categories). After 2 weeks, share the outcome vs forecast.\n"
    )

    header = "### AI Store Coach — next 14 days\n\n"
    intro = (
        f"For **{store_name}**, we combine recent performance with the forecast for the next 14 days. "
        "Here is what you can act on:\n\n"
    )

    return header + intro + foot_msg + turnover_msg + scenario_msg + peak_msg + action_msg


# -------------
# MAIN UI
# -------------

def main():
    st.markdown(
        f"""
        <div class="pfm-header">
          <div>
            <div class="pfm-title">PFM Retail Performance Copilot</div>
            <div class="pfm-sub">Store view · fast to read · built for action</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Retailer selection via clients.json ---
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

    # --- Locations via FastAPI ---
    try:
        locations_df = get_locations_by_company(company_id)
    except requests.exceptions.ReadTimeout:
        st.error(
            "FastAPI location endpoint timed out. This often happens on a cold start or with many locations. "
            "Click **Analyze** again or try later."
        )
        st.stop()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching locations from FastAPI: {e}")
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

    # --- Period selection (current vs previous) ---
    period_choice = st.sidebar.selectbox(
        "Period",
        [
            "This week",
            "Last week",
            "This month",
            "Last month",
            "This quarter",
            "Last quarter",
        ],
        index=2,  # default: This month
    )

    today = datetime.today().date()

    # Forecast history start date
    default_hist_start = today - timedelta(days=365)
    hist_start_input = st.sidebar.date_input(
        "Forecast history from",
        value=default_hist_start,
        help="This date defines from when historical data is used to train the forecast model.",
    )

    def get_week_range(base_date):
        wd = base_date.weekday()  # 0=Mon
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

    # Current + previous ranges
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

    else:  # "Last quarter"
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
    weather_location = st.sidebar.text_input("Weather location", value=f"{default_city},NL")
    postcode4 = st.sidebar.text_input("CBS postcode (4 digits)", value=postcode[:4] if postcode else "")

    forecast_mode = st.sidebar.radio(
        "Forecast mode",
        ["Simple (DoW)", "Pro (LightGBM beta)"],
        index=0,
    )

    run_btn = st.sidebar.button("Analyze", type="primary")

    # Pre-render ROW 5 placeholder (always visible)
    panel_start("🔮 NEXT 14 DAYS — WHAT TO EXPECT", "Run the analysis to populate forecast and peak/quiet days.")
    chip("Always visible")
    st.markdown("Forecast output will appear here after you click **Analyze**.")
    panel_end()

    if not run_btn:
        st.info("Select retailer & store, choose a period, then click **Analyze**.")
        return

    # --- Fetch data from FastAPI ---
    with st.spinner("Fetching data from Storescan / FastAPI..."):
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
        df_all_raw = normalize_vemcount_response(resp_all, kpi_keys=metric_map.keys())
        df_all_raw = df_all_raw.rename(columns=metric_map)

    if df_all_raw.empty:
        st.warning("No data found for this year for this store.")
        return

    df_all_raw["date"] = pd.to_datetime(df_all_raw["date"], errors="coerce")
    df_all_raw = df_all_raw.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Extra history for forecast via period=date
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

    # Slice current + previous
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

    # Weather history
    weather_df = pd.DataFrame()
    if weather_location and VISUALCROSSING_KEY:
        weather_df = fetch_visualcrossing_history(weather_location, start_cur, end_cur)

    weather_cfg_base = None
    if VISUALCROSSING_KEY and weather_location:
        parts = weather_location.split(",")
        city = parts[0].strip() if parts else ""
        country = parts[1].strip() if len(parts) > 1 else ""
        weather_cfg_base = {
            "mode": "city_country",
            "city": city,
            "country": country,
            "location": weather_location.strip(),
        }

    # Pathzz street traffic (weekly)
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

    # CBS context
    cbs_stats = {}
    if postcode4:
        cbs_stats = get_cbs_stats_for_postcode4(postcode4)

    # -------------------------
    # ROW 1: KPI cards + SVI
    # -------------------------
    st.markdown(f"### {selected_client['brand']} — **{shop_row['name']}**  ·  {start_cur} → {end_cur}")

    foot_cur = df_cur["footfall"].sum() if "footfall" in df_cur.columns else 0
    foot_prev = df_prev["footfall"].sum() if ("footfall" in df_prev.columns and not df_prev.empty) else 0
    foot_delta = f"{((foot_cur - foot_prev) / foot_prev * 100):+.1f}%" if foot_prev > 0 else None

    turn_cur = df_cur["turnover"].sum() if "turnover" in df_cur.columns else 0
    turn_prev = df_prev["turnover"].sum() if ("turnover" in df_prev.columns and not df_prev.empty) else 0
    turn_delta = f"{((turn_cur - turn_prev) / turn_prev * 100):+.1f}%" if turn_prev > 0 else None

    spv_cur = df_cur["sales_per_visitor"].mean() if "sales_per_visitor" in df_cur.columns else np.nan
    spv_prev = df_prev["sales_per_visitor"].mean() if ("sales_per_visitor" in df_prev.columns and not df_prev.empty) else np.nan
    spv_delta = f"{((spv_cur - spv_prev) / spv_prev * 100):+.1f}%" if (pd.notna(spv_cur) and pd.notna(spv_prev) and spv_prev > 0) else None

    conv_cur = df_cur["conversion_rate"].mean() if "conversion_rate" in df_cur.columns else np.nan
    conv_prev = df_prev["conversion_rate"].mean() if ("conversion_rate" in df_prev.columns and not df_prev.empty) else np.nan
    conv_delta = f"{((conv_cur - conv_prev) / conv_prev * 100):+.1f}%" if (pd.notna(conv_cur) and pd.notna(conv_prev) and conv_prev > 0) else None

    # Placeholder SVI (next step)
    store_svi = None          # score 0-100
    store_svi_rank = None     # rank within region (e.g., "12 / 100") - next step we compute

    r1 = st.columns([1, 1, 1, 1, 1.2], gap="small")

    with r1[0]:
        kpi_card("Footfall (current period)", fmt_int(foot_cur), delta=foot_delta, help_text="vs previous period")

    with r1[1]:
        kpi_card("Turnover (current period)", fmt_eur(turn_cur), delta=turn_delta, help_text="vs previous period")

    with r1[2]:
        spv_val = f"€ {spv_cur:.2f}".replace(".", ",") if pd.notna(spv_cur) else "-"
        kpi_card("SPV (avg)", spv_val, delta=spv_delta, help_text="Sales per visitor")

    with r1[3]:
        if avg_capture_cur is not None and pd.notna(avg_capture_cur):
            cap_delta = None
            if avg_capture_prev is not None and pd.notna(avg_capture_prev) and avg_capture_prev > 0:
                cap_delta_val = (avg_capture_cur - avg_capture_prev) / avg_capture_prev * 100
                cap_delta = f"{cap_delta_val:+.1f}%"
            kpi_card("Capture rate (avg)", fmt_pct(avg_capture_cur), delta=cap_delta, help_text="Store vs street traffic", delta_color=PFM_RED)
        else:
            conv_val = fmt_pct(conv_cur) if pd.notna(conv_cur) else "-"
            kpi_card("Conversion (avg)", conv_val, delta=conv_delta, help_text="Transactions / visitors", delta_color=PFM_RED)

    with r1[4]:
        svi_value = fmt_score_100(store_svi)
        rank_txt = f"Rank: {store_svi_rank}" if store_svi_rank else "Rank: —"
        kpi_card("Store Vitality (SVI)", svi_value, delta=None, help_text=f"{rank_txt} · benchmark vs regional peers", delta_color=PFM_RED)

    # -------------------------
    # Row 2: Weekly street vs store
    # -------------------------
    st.markdown("### Street traffic vs store traffic (weekly demo)")

    if not capture_weekly.empty:
        chart_df = capture_weekly[["week_start", "footfall", "street_footfall", "turnover", "capture_rate"]].copy()

        iso_cal = chart_df["week_start"].dt.isocalendar()
        chart_df["week_label"] = iso_cal.week.apply(lambda w: f"W{int(w):02d}")

        week_order = chart_df.sort_values("week_start")["week_label"].unique().tolist()

        fig_week = make_subplots(specs=[[{"secondary_y": True}]])

        fig_week.add_bar(
            x=chart_df["week_label"],
            y=chart_df["footfall"],
            name="Footfall (store)",
            marker_color=PFM_PURPLE,
            offsetgroup=0,
        )

        fig_week.add_bar(
            x=chart_df["week_label"],
            y=chart_df["street_footfall"],
            name="Street traffic",
            marker_color=PFM_PEACH,
            opacity=0.7,
            offsetgroup=1,
        )

        fig_week.add_bar(
            x=chart_df["week_label"],
            y=chart_df["turnover"],
            name="Turnover (€)",
            marker_color=PFM_PINK,
            opacity=0.7,
            offsetgroup=2,
        )

        fig_week.add_trace(
            go.Scatter(
                x=chart_df["week_label"],
                y=chart_df["capture_rate"],
                name="Capture rate (%)",
                mode="lines+markers",
                line=dict(color=PFM_RED, width=2),
            ),
            secondary_y=True,
        )

        fig_week.update_xaxes(title_text="Week", categoryorder="array", categoryarray=week_order)
        fig_week.update_yaxes(title_text="Footfall / street traffic / turnover (€)", secondary_y=False)
        fig_week.update_yaxes(title_text="Capture rate (%)", secondary_y=True)

        fig_week.update_layout(
            barmode="group",
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=40, r=40, t=40, b=40),
        )

        st.plotly_chart(fig_week, use_container_width=True)
    else:
        st.info("No Pathzz weekly data available for this period.")

    # -------------------------
    # Row 3: Daily chart
    # -------------------------
    st.markdown("### Daily footfall & turnover")
    if "footfall" in df_cur.columns and "turnover" in df_cur.columns:
        daily_df = df_cur[["date", "footfall", "turnover"]].copy()

        fig_daily = make_subplots(specs=[[{"secondary_y": True}]])

        fig_daily.add_bar(
            x=daily_df["date"],
            y=daily_df["footfall"],
            name="Footfall",
            marker_color=PFM_PURPLE,
        )

        fig_daily.add_trace(
            go.Scatter(
                x=daily_df["date"],
                y=daily_df["turnover"],
                name="Turnover (€)",
                mode="lines+markers",
                line=dict(color=PFM_PEACH, width=2),
            ),
            secondary_y=True,
        )

        fig_daily.update_xaxes(title_text="")
        fig_daily.update_yaxes(title_text="Footfall", secondary_y=False)
        fig_daily.update_yaxes(title_text="Turnover (€)", secondary_y=True)

        fig_daily.update_layout(
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=40, r=40, t=20, b=40),
        )

        st.plotly_chart(fig_daily, use_container_width=True)

    # -------------------------
    # Row 4: Weather vs footfall (optional)
    # -------------------------
    if not weather_df.empty:
        st.markdown("### Weather vs footfall (indicative)")

        weather_merge = pd.merge(
            df_cur[["date", "footfall"]],
            weather_df[["date", "temp", "precip"]],
            on="date",
            how="left",
        )

        fig_weather = make_subplots(specs=[[{"secondary_y": True}]])

        fig_weather.add_bar(
            x=weather_merge["date"],
            y=weather_merge["footfall"],
            name="Footfall",
            marker_color=PFM_PURPLE,
        )

        fig_weather.add_trace(
            go.Scatter(
                x=weather_merge["date"],
                y=weather_merge["temp"],
                name="Temperature (°C)",
                mode="lines+markers",
                line=dict(color=PFM_RED, width=2),
            ),
            secondary_y=True,
        )

        fig_weather.add_trace(
            go.Scatter(
                x=weather_merge["date"],
                y=weather_merge["precip"],
                name="Precipitation (mm)",
                mode="lines+markers",
                line=dict(color=PFM_BLUE, width=2, dash="dot"),
            ),
            secondary_y=True,
        )

        fig_weather.update_xaxes(title_text="")
        fig_weather.update_yaxes(title_text="Footfall", secondary_y=False)
        fig_weather.update_yaxes(title_text="Temperature / precipitation", secondary_y=True)

        fig_weather.update_layout(
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=40, r=40, t=20, b=40),
        )

        st.plotly_chart(fig_weather, use_container_width=True)

    # -------------------------
    # Forecast + ROW 5: What to expect
    # -------------------------
    st.markdown("### Forecast: footfall & turnover (next 14 days)")

    st.markdown(
        """
        <small>
        💡 <strong>Forecast explanation</strong><br>
        - <strong>Simple (DoW)</strong> uses average performance per weekday based on your history.<br>
        - <strong>Pro (LightGBM beta)</strong> adds seasonality, lags/rolling effects and (optional) weather.<br>
        - If there is insufficient history or LightGBM is unavailable, Pro falls back to Simple automatically.
        </small>
        """,
        unsafe_allow_html=True,
    )

    weather_cfg = None
    if VISUALCROSSING_KEY and weather_location:
        city_part = weather_location
        country = "Netherlands"

        parts = weather_location.split(",")
        if len(parts) >= 1:
            city_part = parts[0].strip()
        if len(parts) >= 2:
            country_code = parts[1].strip().upper()
            country_map = {
                "NL": "Netherlands",
                "BE": "Belgium",
                "DE": "Germany",
                "FR": "France",
                "UK": "United Kingdom",
                "GB": "United Kingdom",
            }
            country = country_map.get(country_code, country_code)

        weather_cfg = {
            "mode": "city_country",
            "city": city_part,
            "country": country,
            "api_key": VISUALCROSSING_KEY,
        }

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
            st.info("Not enough historical data to show a reliable forecast.")
        else:
            hist_recent = fc_res["hist_recent"]
            fc = fc_res["forecast"]

            recent_foot = fc_res["recent_footfall_sum"]
            recent_turn = fc_res["recent_turnover_sum"]
            fut_foot = fc_res["forecast_footfall_sum"]
            fut_turn = fc_res["forecast_turnover_sum"]

            # --- ROW 5: NEXT 14 DAYS — WHAT TO EXPECT (explicit + visible) ---
            panel_start("🔮 NEXT 14 DAYS — WHAT TO EXPECT", "Clear expectations + peak/quiet days + quick actions")
            chip(f"Model: {fc_res.get('model_type','?')}")
            if fc_res.get("used_simple_fallback", False):
                chip("Fallback → Simple DoW")

            fc_tmp = fc.copy()
            fc_tmp["date"] = pd.to_datetime(fc_tmp["date"], errors="coerce")
            fc_tmp = fc_tmp.dropna(subset=["date"])

            c1, c2, c3 = st.columns([1.1, 1.1, 1.4], gap="small")

            with c1:
                delta_foot = None
                if recent_foot and recent_foot > 0:
                    delta_foot = f"{(fut_foot - recent_foot) / recent_foot * 100:+.1f}%"
                kpi_card(
                    "Expected Footfall (14 days)",
                    fmt_int(fut_foot),
                    delta=delta_foot,
                    help_text="vs last 14 days",
                    delta_color=PFM_PURPLE,
                )

                st.markdown("**Busiest days**")
                peak = fc_tmp.sort_values("footfall_forecast", ascending=False).head(3)
                if peak.empty:
                    st.write("—")
                else:
                    for _, r in peak.iterrows():
                        st.write(f"• {r['date'].strftime('%a %d %b')} (~{fmt_int(r['footfall_forecast'])})")

            with c2:
                delta_turn = None
                if recent_turn and recent_turn > 0:
                    delta_turn = f"{(fut_turn - recent_turn) / recent_turn * 100:+.1f}%"
                kpi_card(
                    "Expected Turnover (14 days)",
                    fmt_eur(fut_turn),
                    delta=delta_turn,
                    help_text="vs last 14 days",
                    delta_color=PFM_RED,
                )

                st.markdown("**Quiet days**")
                quiet = fc_tmp.sort_values("footfall_forecast", ascending=True).head(2)
                if quiet.empty:
                    st.write("—")
                else:
                    for _, r in quiet.iterrows():
                        st.write(f"• {r['date'].strftime('%a %d %b')} (~{fmt_int(r['footfall_forecast'])})")

            with c3:
                st.markdown("**What you should do (simple playbook)**")
                st.markdown("🔴 **Peak days**")
                for a in ["Schedule extra floor coverage", "Reduce checkout friction", "Push bundles / add-ons"]:
                    st.write("•", a)

                st.markdown("🟠 **Quiet days**")
                for a in ["Run micro-training (15 min)", "Refresh layout + signage", "Test one conversion tactic"]:
                    st.write("•", a)

            panel_end()

            # Expected month turnover (actual + forecast remainder within horizon)
            month_start = datetime(today.year, today.month, 1).date()
            today_date = today

            if "turnover" in df_all_raw.columns:
                actual_month_turn = df_all_raw[
                    (df_all_raw["date"].dt.date >= month_start) & (df_all_raw["date"].dt.date <= today_date)
                ]["turnover"].sum()
            else:
                actual_month_turn = 0.0

            future_month_turn = 0.0
            if isinstance(fc, pd.DataFrame) and "date" in fc.columns and "turnover_forecast" in fc.columns:
                fc_month = fc.copy()
                fc_month["date"] = pd.to_datetime(fc_month["date"], errors="coerce")
                fc_month = fc_month.dropna(subset=["date"])
                if not fc_month.empty:
                    future_month_turn = fc_month[
                        (fc_month["date"].dt.date > today_date) & (fc_month["date"].dt.month == today_date.month)
                    ]["turnover_forecast"].sum()

            expected_month_turn = actual_month_turn + future_month_turn

            col_month, _ = st.columns([2, 3])
            with col_month:
                panel_start("Expected turnover — current month", "Actual MTD + forecast within the 14-day horizon")
                kpi_card("Expected turnover this month", fmt_eur(expected_month_turn), delta=None, help_text="")
                if actual_month_turn > 0:
                    remaining = expected_month_turn - actual_month_turn
                    st.caption(
                        f"Actual MTD: {fmt_eur(actual_month_turn)} · Expected extra (rest of month): {fmt_eur(remaining)}"
                    )
                panel_end()

            # Chart: last 28 days history + forecast
            fig_fc = make_subplots(specs=[[{"secondary_y": True}]])

            fig_fc.add_bar(
                x=hist_recent["date"],
                y=hist_recent["footfall"],
                name="Footfall (history)",
                marker_color=PFM_PURPLE,
            )

            fig_fc.add_bar(
                x=fc["date"],
                y=fc["footfall_forecast"],
                name="Footfall (forecast)",
                marker_color=PFM_PEACH,
            )

            if "turnover" in hist_recent.columns:
                fig_fc.add_trace(
                    go.Scatter(
                        x=hist_recent["date"],
                        y=hist_recent["turnover"],
                        name="Turnover (history)",
                        mode="lines",
                        line=dict(color=PFM_PINK, width=2),
                    ),
                    secondary_y=True,
                )

            fig_fc.add_trace(
                go.Scatter(
                    x=fc["date"],
                    y=fc["turnover_forecast"],
                    name="Turnover (forecast)",
                    mode="lines+markers",
                    line=dict(color=PFM_RED, width=2, dash="dash"),
                ),
                secondary_y=True,
            )

            fig_fc.update_xaxes(title_text="")
            fig_fc.update_yaxes(title_text="Footfall", secondary_y=False)
            fig_fc.update_yaxes(title_text="Turnover (€)", secondary_y=True)

            fig_fc.update_layout(
                height=350,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(l=40, r=40, t=20, b=40),
            )

            st.plotly_chart(fig_fc, use_container_width=True)

            # Data-driven AI store coach
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
        st.info("Forecast could not be computed (insufficient data / missing columns / weather issue).")
        st.exception(e)

    # Debug
    with st.expander("🔧 Debug"):
        st.write("Period choice:", period_choice)
        st.write("Current period:", start_cur, "→", end_cur)
        st.write("Previous period:", start_prev, "→", end_prev)
        st.write("Shop row:", shop_row)
        st.write("Daily ALL (head):", df_all_raw.head())
        st.write("Daily (cur):", df_cur.head())
        st.write("Daily (prev):", df_prev.head())
        st.write("Pathzz weekly:", pathzz_weekly.head())
        st.write(
            "Capture weekly:",
            capture_weekly.head() if isinstance(capture_weekly, pd.DataFrame) else capture_weekly,
        )
        st.write("CBS stats:", cbs_stats)
        st.write("Weather df:", weather_df.head() if isinstance(weather_df, pd.DataFrame) else weather_df)
        st.write("Forecast mode:", forecast_mode)
        st.write("Weather cfg (base):", weather_cfg_base)

        try:
            st.write("Forecast model_type:", fc_res.get("model_type"))
            st.write("Forecast used_simple_fallback:", fc_res.get("used_simple_fallback"))
            st.write("Forecast head:", fc_res["forecast"].head())
        except Exception:
            st.write("Forecast object not available in this run.")


if __name__ == "__main__":
    main()
