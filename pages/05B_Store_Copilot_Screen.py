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

@st.cache_data(ttl=600)
def get_report_with_retry(params: list[tuple[str, str]], timeout_s: int = 120, attempts: int = 2):
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(REPORT_URL, params=params, timeout=timeout_s)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            time.sleep(0.7 * attempt)
    raise last_err

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
PFM_GREEN = "#22C55E"
PFM_AMBER = "#F59E0B"

# ----------------------
# Quick UI polish (KPI tiles)
# ----------------------
st.markdown(
    f"""
    <style>
      .kpi-tile {{
        border: 1px solid {PFM_LINE};
        border-radius: 16px;
        background: white;
        padding: 14px 16px;
        box-shadow: 0 1px 0 rgba(17,24,39,0.03);
        height: 100%;
      }}
      .kpi-label {{
        color: {PFM_GRAY};
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 6px;
      }}
      .kpi-value {{
        color: {PFM_DARK};
        font-size: 1.65rem;
        font-weight: 900;
        line-height: 1.1;
      }}
      .kpi-sub {{
        color: {PFM_GRAY};
        font-size: 0.82rem;
        margin-top: 6px;
      }}
      .pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 800;
        border: 1px solid rgba(0,0,0,0.06);
        margin-top: 10px;
      }}
    </style>
    """,
    unsafe_allow_html=True
)

def _delta_pill(delta_pct: float | None):
    """
    delta_pct in % (e.g., -12.3)
    """
    if delta_pct is None or pd.isna(delta_pct):
        return ""
    d = float(delta_pct)

    # color logic (simple & readable)
    if d >= 0:
        bg = "rgba(34,197,94,0.12)"   # green-ish
        fg = "#16A34A"
        arrow = "▲"
    else:
        bg = "rgba(240,68,56,0.12)"   # red-ish
        fg = PFM_RED
        arrow = "▼"

    txt = f"{d:+.1f}%".replace(".", ",")
    return f'<div class="pill" style="background:{bg}; color:{fg};">{arrow} {txt}</div>'

def kpi_tile(label: str, value: str, delta_pct: float | None = None, sub: str = ""):
    pill = _delta_pill(delta_pct)
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="kpi-tile">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          {pill}
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------
# Format helpers
# -------------
def fmt_eur(x: float) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"€ {float(x):,.0f}".replace(",", ".")

def fmt_pct(x: float) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x):.1f}%".replace(".", ",")

def fmt_int(x: float) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x):,.0f}".replace(",", ".")

# ----------------------
# SVI UI helpers (prominent card)
# ----------------------
def svi_style(score: float | None):
    if score is None or pd.isna(score):
        return {"label": "No score", "color": PFM_LINE, "emoji": "⚪", "hint": "Not enough data for a stable benchmark"}
    s = float(score)
    if s >= 75:
        return {"label": "High performance", "color": PFM_GREEN, "emoji": "🟢", "hint": "Top tier vs regional peers"}
    if s >= 60:
        return {"label": "Good / stable", "color": PFM_PURPLE, "emoji": "🟣", "hint": "Solid performance vs peers"}
    if s >= 45:
        return {"label": "Attention required", "color": PFM_AMBER, "emoji": "🟠", "hint": "Below peers on key drivers"}
    return {"label": "Under pressure", "color": PFM_RED, "emoji": "🔴", "hint": "Material gap vs regional peers"}

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

    for c in ["footfall", "turnover", "transactions"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

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
    postcode = shop_row.get("zip") or shop_row.get("postcode", "")

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

    # ✅ Force refresh regions.csv cache when running analysis (prevents stale region_map)
    try:
        load_region_mapping.clear()
    except Exception:
        pass

    # ---------------------------
    # Store meta (locations + regions)
    # ---------------------------
    region_map = load_region_mapping()

    locations_df = locations_df.copy()
    locations_df["id"] = pd.to_numeric(locations_df["id"], errors="coerce").astype("Int64")

    merged_meta = pd.DataFrame()
    if not region_map.empty:
        merged_meta = locations_df.merge(region_map, left_on="id", right_on="shop_id", how="left")
    else:
        merged_meta = locations_df.copy()
        merged_meta["region"] = np.nan
        merged_meta["sqm_override"] = np.nan
        merged_meta["store_label"] = np.nan

    if "sqm" in merged_meta.columns:
        merged_meta["sqm_effective"] = np.where(
            merged_meta.get("sqm_override", pd.Series([np.nan] * len(merged_meta))).notna(),
            merged_meta.get("sqm_override"),
            pd.to_numeric(merged_meta["sqm"], errors="coerce"),
        )
    else:
        merged_meta["sqm_effective"] = pd.to_numeric(merged_meta.get("sqm_override", np.nan), errors="coerce")

    if "store_label" in merged_meta.columns and merged_meta["store_label"].notna().any():
        merged_meta["store_display"] = np.where(
            merged_meta["store_label"].notna(),
            merged_meta["store_label"],
            merged_meta.get("name", merged_meta["id"].astype(str)),
        )
    else:
        merged_meta["store_display"] = merged_meta.get("name", merged_meta["id"].astype(str))

    # ---------------------------
    # Fetch store data (this year, then slice locally)
    # ---------------------------
    with st.spinner("Fetching data via FastAPI..."):
        metric_map = {"count_in": "footfall", "turnover": "turnover", "sales_per_sqm": "sales_per_sqm"}
        resp_all = get_report([shop_id], list(metric_map.keys()), period="this_year", step="day", source="shops")
        df_all_raw = normalize_vemcount_response(resp_all, kpi_keys=metric_map.keys()).rename(columns=metric_map)

    if df_all_raw.empty:
        st.warning("No data found for this year for this store.")
        return

    df_all_raw["date"] = pd.to_datetime(df_all_raw["date"], errors="coerce")
    df_all_raw = df_all_raw.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    store_key_col_all = None
    for cand in ["id", "shop_id", "location_id"]:
        if cand in df_all_raw.columns:
            store_key_col_all = cand
            break
    if store_key_col_all is None:
        st.error("No store id column found in report response (id/shop_id/location_id).")
        return
    df_all_raw[store_key_col_all] = pd.to_numeric(df_all_raw[store_key_col_all], errors="coerce").astype("Int64")

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

    # ---------------------------
    # KPI cards + SVI
    # ---------------------------
    st.subheader(f"{selected_client['brand']} – {shop_row['name']}")

    foot_cur = float(df_cur["footfall"].sum()) if "footfall" in df_cur.columns else 0.0
    foot_prev = float(df_prev["footfall"].sum()) if ("footfall" in df_prev.columns and not df_prev.empty) else 0.0
    foot_delta = f"{(foot_cur - foot_prev) / foot_prev * 100:+.1f}%" if foot_prev > 0 else None

    turn_cur = float(df_cur["turnover"].sum()) if "turnover" in df_cur.columns else 0.0
    turn_prev = float(df_prev["turnover"].sum()) if ("turnover" in df_prev.columns and not df_prev.empty) else 0.0
    turn_delta = f"{(turn_cur - turn_prev) / turn_prev * 100:+.1f}%" if turn_prev > 0 else None

    spv_cur = float(df_cur["sales_per_visitor"].mean()) if "sales_per_visitor" in df_cur.columns else np.nan
    spv_prev = float(df_prev["sales_per_visitor"].mean()) if ("sales_per_visitor" in df_prev.columns and not df_prev.empty) else np.nan
    spv_delta = f"{(spv_cur - spv_prev) / spv_prev * 100:+.1f}%" if pd.notna(spv_cur) and pd.notna(spv_prev) and spv_prev > 0 else None

    conv_cur = float(df_cur["conversion_rate"].mean()) if "conversion_rate" in df_cur.columns else np.nan
    conv_prev = float(df_prev["conversion_rate"].mean()) if ("conversion_rate" in df_prev.columns and not df_prev.empty) else np.nan
    conv_delta = f"{(conv_cur - conv_prev) / conv_prev * 100:+.1f}%" if pd.notna(conv_cur) and pd.notna(conv_prev) and conv_prev > 0 else None

    # ---------------------------
    # ✅ SVI (SVI-PROOF)
    # ---------------------------
    store_svi_score = None
    store_svi_rank = None
    store_svi_peer_n = None
    store_region = None
    store_svi_status = None
    store_svi_reason = None
    store_svi_error = None

    meta = pd.DataFrame()
    try:
        loc_meta = locations_df.copy()
        loc_meta["id"] = pd.to_numeric(loc_meta["id"], errors="coerce").astype("Int64")
        if "sqm" in loc_meta.columns:
            loc_meta["sqm"] = pd.to_numeric(loc_meta["sqm"], errors="coerce")
        else:
            loc_meta["sqm"] = np.nan

        reg = region_map.copy() if region_map is not None else pd.DataFrame()
        if not reg.empty:
            reg["shop_id"] = pd.to_numeric(reg["shop_id"], errors="coerce").astype("Int64")
            reg["sqm_override"] = pd.to_numeric(reg.get("sqm_override", np.nan), errors="coerce")

        meta = loc_meta.merge(reg, left_on="id", right_on="shop_id", how="left")
        meta["sqm_effective"] = np.where(
            meta.get("sqm_override", pd.Series([np.nan] * len(meta))).notna(),
            meta["sqm_override"],
            meta["sqm"],
        )
        meta["store_display"] = np.where(
            meta.get("store_label", pd.Series([np.nan] * len(meta))).notna(),
            meta["store_label"],
            meta.get("name", meta["id"].astype(str)),
        )

        row_meta = meta.loc[meta["id"] == pd.Series([shop_id], dtype="Int64").iloc[0]]
        if not row_meta.empty:
            r = row_meta.get("region", pd.Series([None])).iloc[0]
            store_region = str(r) if (pd.notna(r) and str(r).strip() != "" and str(r).strip().lower() != "nan") else None

    except Exception as e:
        store_svi_error = f"Meta build failed: {e}"

    if store_region and isinstance(meta, pd.DataFrame) and not meta.empty:
        try:
            peer_ids = (
                meta.loc[meta["region"].astype(str) == str(store_region), "id"]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
            if shop_id not in peer_ids:
                peer_ids.append(shop_id)

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

            if df_peers is not None and not df_peers.empty:
                df_peers["date"] = pd.to_datetime(df_peers["date"], errors="coerce")
                df_peers = df_peers.dropna(subset=["date"])

                store_key_col = None
                for cand in ["id", "shop_id", "location_id"]:
                    if cand in df_peers.columns:
                        store_key_col = cand
                        break

                if store_key_col is None:
                    store_svi_error = "No store id column in peers response (id/shop_id/location_id)."
                else:
                    df_peers[store_key_col] = pd.to_numeric(df_peers[store_key_col], errors="coerce").astype("Int64")
                    df_peers = df_peers.merge(
                        meta[["id", "store_display", "sqm_effective"]],
                        left_on=store_key_col,
                        right_on="id",
                        how="left",
                    )
                    df_peers["sqm_effective"] = pd.to_numeric(df_peers["sqm_effective"], errors="coerce")

                    df_peers = compute_daily_kpis(df_peers)

                    svi_df = build_store_vitality(
                        df_period=df_peers,
                        region_shops=meta,
                        store_key_col=store_key_col,
                    )

                    if svi_df is not None and not svi_df.empty:
                        svi_df["svi_score"] = pd.to_numeric(svi_df["svi_score"], errors="coerce")
                        svi_df = svi_df.dropna(subset=["svi_score"]).sort_values("svi_score", ascending=False).reset_index(drop=True)
                        svi_df["rank"] = np.arange(1, len(svi_df) + 1)

                        target_id = pd.Series([shop_id], dtype="Int64").iloc[0]
                        row = svi_df.loc[pd.to_numeric(svi_df[store_key_col], errors="coerce").astype("Int64") == target_id]

                        if not row.empty:
                            store_svi_score = float(np.clip(row["svi_score"].iloc[0], 0, 100))
                            store_svi_rank = int(row["rank"].iloc[0])
                            store_svi_peer_n = int(len(svi_df))
                            store_svi_status = row.get("svi_status", pd.Series([None])).iloc[0]
                            store_svi_reason = row.get("reason_short", pd.Series([None])).iloc[0]
                    else:
                        store_svi_error = "SVI DF empty (no vitality computed)."
            else:
                store_svi_error = "Peers DF empty (no peer data returned)."

        except Exception as e:
            store_svi_error = f"SVI build failed: {e}"

    # ---------------------------
    # KPI row (SVI prominent)
    # ---------------------------
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1.35])
    
    # % deltas naar float (voor pill)
    foot_delta_pct = ((foot_cur - foot_prev) / foot_prev * 100.0) if foot_prev and foot_prev > 0 else None
    turn_delta_pct = ((turn_cur - turn_prev) / turn_prev * 100.0) if turn_prev and turn_prev > 0 else None
    spv_delta_pct  = ((spv_cur - spv_prev) / spv_prev * 100.0) if (pd.notna(spv_cur) and pd.notna(spv_prev) and spv_prev > 0) else None
    conv_delta_pct = ((conv_cur - conv_prev) / conv_prev * 100.0) if (pd.notna(conv_cur) and pd.notna(conv_prev) and conv_prev > 0) else None
    
    with col1:
        kpi_tile("Footfall (period)", fmt_int(foot_cur), foot_delta_pct, sub="vs vorige periode")
    with col2:
        kpi_tile("Turnover (period)", fmt_eur(turn_cur), turn_delta_pct, sub="vs vorige periode")
    with col3:
        spv_val = f"€ {float(spv_cur):.2f}".replace(".", ",") if pd.notna(spv_cur) else "-"
        kpi_tile("Avg spend / visitor", spv_val, spv_delta_pct, sub="vs vorige periode")
    
    with col4:
        # jij had hier capture óf conversion; houden we die logica, maar dan in tile vorm
        if avg_capture_cur is not None and not pd.isna(avg_capture_cur) and avg_capture_prev not in (None, 0):
            cap_delta_pct = ((avg_capture_cur - avg_capture_prev) / avg_capture_prev * 100.0) if (avg_capture_prev and avg_capture_prev > 0) else None
            kpi_tile("Avg capture rate", fmt_pct(avg_capture_cur), cap_delta_pct, sub="vs vorige periode")
        else:
            conv_val = fmt_pct(conv_cur) if pd.notna(conv_cur) else "-"
            kpi_tile("Avg conversion", conv_val, conv_delta_pct, sub="vs vorige periode")
    
    # col5: jouw SVI-card blijft (maar hij oogt nu al prominenter door col5 wat breder)

    with col5:
        svi_meta = svi_style(store_svi_score)
        svi_color = svi_meta["color"]
        svi_label = svi_meta["label"]
        svi_emoji = svi_meta["emoji"]
        svi_hint = svi_meta["hint"]

        # if service provides a status, show that instead of generic
        if store_svi_status and str(store_svi_status).strip().lower() not in ("none", "nan", ""):
            svi_label = str(store_svi_status)
            # keep color based on score (still consistent)

        score_txt = f"{store_svi_score:.0f}" if store_svi_score is not None else "—"
        rank_txt = f"{store_svi_rank} / {store_svi_peer_n}" if (store_svi_rank and store_svi_peer_n) else "—"

        st.markdown(
            f"""
            <div style="
                border:1px solid {PFM_LINE};
                border-left:10px solid {svi_color};
                border-radius:16px;
                padding:0.95rem 1.05rem;
                background:white;
                height: 124px;
                display:flex;
                flex-direction:column;
                justify-content:center;
            ">
              <div style="display:flex; align-items:center; justify-content:space-between;">
                <div style="color:{PFM_GRAY}; font-size:0.85rem; font-weight:800; letter-spacing:0.2px;">
                  Store Vitality (SVI)
                </div>
                <div style="
                    background:{svi_color}15;
                    color:{svi_color};
                    padding:0.18rem 0.55rem;
                    border-radius:999px;
                    font-size:0.80rem;
                    font-weight:900;
                ">
                  {svi_emoji} {svi_label}
                </div>
              </div>

              <div style="display:flex; align-items:baseline; gap:10px; margin-top:0.35rem;">
                <div style="color:{PFM_DARK}; font-size:2.10rem; font-weight:950; line-height:1;">
                  {score_txt}
                </div>
                <div style="color:{PFM_GRAY}; font-size:1.0rem; font-weight:900;">
                  / 100
                </div>
              </div>

              <div style="color:{PFM_GRAY}; font-size:0.85rem; margin-top:0.18rem;">
                <b>Rank:</b> {rank_txt} · benchmark vs regional peers
                {f" · <b>Region:</b> {store_region}" if store_region else ""}
              </div>

              <div style="color:{PFM_GRAY}; font-size:0.80rem; margin-top:0.10rem;">
                {svi_hint}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if store_svi_reason and str(store_svi_reason).strip().lower() not in ("none", "nan", ""):
            st.caption(store_svi_reason)

    st.markdown("### Month outlook (forecast + actual)")
    
    # --- Month boundaries (based on today) ---
    today_dt = datetime.today().date()
    month_start = datetime(today_dt.year, today_dt.month, 1).date()
    if today_dt.month == 12:
        month_end = datetime(today_dt.year + 1, 1, 1).date() - timedelta(days=1)
    else:
        month_end = datetime(today_dt.year, today_dt.month + 1, 1).date() - timedelta(days=1)
    
    mtd_end = min(today_dt, month_end)
    days_left = max(0, (month_end - mtd_end).days)
    
    # --- Actual MTD from df_all_raw ---
    df_mtd = df_all_raw.copy()
    df_mtd["date"] = pd.to_datetime(df_mtd["date"], errors="coerce")
    df_mtd = df_mtd.dropna(subset=["date"])
    df_mtd = df_mtd[(df_mtd["date"] >= pd.Timestamp(month_start)) & (df_mtd["date"] <= pd.Timestamp(mtd_end))].copy()
    
    turn_mtd_actual = float(pd.to_numeric(df_mtd.get("turnover", pd.Series([])), errors="coerce").fillna(0).sum())
    
    # --- Forecast remaining month from fc_res (if available) ---
    turn_remaining_fc = None
    foot_remaining_fc = None
    turn_total_month_fc = None
    coverage_days = None
    
    try:
        if isinstance(fc_res, dict) and fc_res.get("enough_history", False):
            fc_df = fc_res.get("forecast", pd.DataFrame()).copy()
            if isinstance(fc_df, pd.DataFrame) and not fc_df.empty:
                fc_df["date"] = pd.to_datetime(fc_df["date"], errors="coerce").dt.date
                fc_month = fc_df[(fc_df["date"] > mtd_end) & (fc_df["date"] <= month_end)].copy()
                coverage_days = int(fc_month["date"].nunique()) if not fc_month.empty else 0
    
                if not fc_month.empty:
                    turn_remaining_fc = float(pd.to_numeric(fc_month.get("turnover_forecast", 0), errors="coerce").fillna(0).sum())
                    foot_remaining_fc = float(pd.to_numeric(fc_month.get("footfall_forecast", 0), errors="coerce").fillna(0).sum())
                    turn_total_month_fc = turn_mtd_actual + turn_remaining_fc
    except Exception:
        pass
    
    # --- Same month last year (YoY baseline) ---
    turn_mtd_lastyear = None
    turn_month_lastyear = None
    
    try:
        last_year = today_dt.year - 1
        ly_month_start = datetime(last_year, today_dt.month, 1).date()
        if today_dt.month == 12:
            ly_month_end = datetime(last_year + 1, 1, 1).date() - timedelta(days=1)
        else:
            ly_month_end = datetime(last_year, today_dt.month + 1, 1).date() - timedelta(days=1)
    
        ly_mtd_end = min(datetime(last_year, today_dt.month, mtd_end.day).date(), ly_month_end)
    
        resp_ly_mtd = get_report(
            [shop_id],
            ["turnover"],
            period="date",
            step="day",
            source="shops",
            form_date_from=ly_month_start.strftime("%Y-%m-%d"),
            form_date_to=ly_mtd_end.strftime("%Y-%m-%d"),
        )
        df_ly_mtd = normalize_vemcount_response(resp_ly_mtd, kpi_keys=["turnover"]).copy()
        if not df_ly_mtd.empty:
            turn_mtd_lastyear = float(pd.to_numeric(df_ly_mtd.get("turnover", 0), errors="coerce").fillna(0).sum())
    
        resp_ly_full = get_report(
            [shop_id],
            ["turnover"],
            period="date",
            step="day",
            source="shops",
            form_date_from=ly_month_start.strftime("%Y-%m-%d"),
            form_date_to=ly_month_end.strftime("%Y-%m-%d"),
        )
        df_ly_full = normalize_vemcount_response(resp_ly_full, kpi_keys=["turnover"]).copy()
        if not df_ly_full.empty:
            turn_month_lastyear = float(pd.to_numeric(df_ly_full.get("turnover", 0), errors="coerce").fillna(0).sum())
    
    except Exception:
        pass
    
    # YoY deltas
    mtd_yoy_pct = ((turn_mtd_actual - turn_mtd_lastyear) / turn_mtd_lastyear * 100.0) if (turn_mtd_lastyear and turn_mtd_lastyear > 0) else None
    month_yoy_pct = ((turn_total_month_fc - turn_month_lastyear) / turn_month_lastyear * 100.0) if (turn_total_month_fc is not None and turn_month_lastyear and turn_month_lastyear > 0) else None
    
    m1, m2, m3, m4 = st.columns([1, 1, 1, 1])
    
    with m1:
        kpi_tile("Turnover MTD (actual)", fmt_eur(turn_mtd_actual), mtd_yoy_pct, sub="vs zelfde maand vorig jaar")
    
    with m2:
        if turn_remaining_fc is None:
            kpi_tile("Turnover remaining (forecast)", "—", None, sub="forecast not available")
        else:
            kpi_tile("Turnover remaining (forecast)", fmt_eur(turn_remaining_fc), None, sub=f"{coverage_days}/{days_left} resterende dagen")
    
    with m3:
        if turn_total_month_fc is None:
            kpi_tile("Turnover total month (forecast)", "—", None, sub="forecast not available")
        else:
            kpi_tile("Turnover total month (forecast)", fmt_eur(turn_total_month_fc), month_yoy_pct, sub="vs zelfde maand vorig jaar")
    
    with m4:
        if foot_remaining_fc is None:
            kpi_tile("Footfall remaining (forecast)", "—", None, sub="forecast not available")
        else:
            kpi_tile("Footfall remaining (forecast)", fmt_int(foot_remaining_fc), None, sub=f"{coverage_days}/{days_left} resterende dagen")
    
    if coverage_days is not None:
        st.caption(f"Forecast covers **{coverage_days}/{days_left}** remaining days.")

    # ---------------------------
    # Forecast compute (once) -> used for Month outlook + overlay + forecast section
    # ---------------------------
    fc_res = None
    hist_recent = pd.DataFrame()
    fc = pd.DataFrame()
    recent_foot = recent_turn = fut_foot = fut_turn = None

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

        if isinstance(fc_res, dict) and fc_res.get("enough_history", False):
            hist_recent = fc_res.get("hist_recent", pd.DataFrame())
            fc = fc_res.get("forecast", pd.DataFrame())

            recent_foot = fc_res.get("recent_footfall_sum")
            recent_turn = fc_res.get("recent_turnover_sum")
            fut_foot = fc_res.get("forecast_footfall_sum")
            fut_turn = fc_res.get("forecast_turnover_sum")

    except Exception:
        fc_res = None

    # ---------------------------
    # ✅ Month outlook (UNDER KPI row)
    # ---------------------------
    st.markdown("### Month outlook (forecast + actual)")

    month_start = datetime(today.year, today.month, 1).date()
    if today.month == 12:
        month_end = (datetime(today.year + 1, 1, 1).date() - timedelta(days=1))
    else:
        month_end = (datetime(today.year, today.month + 1, 1).date() - timedelta(days=1))

    df_mtd = df_all_raw[(df_all_raw["date"].dt.date >= month_start) & (df_all_raw["date"].dt.date <= today)].copy()
    df_mtd = compute_daily_kpis(df_mtd)

    mtd_turnover = float(df_mtd["turnover"].sum()) if ("turnover" in df_mtd.columns and not df_mtd.empty) else 0.0
    mtd_footfall = float(df_mtd["footfall"].sum()) if ("footfall" in df_mtd.columns and not df_mtd.empty) else 0.0

    forecast_month_turn = None
    forecast_month_foot = None
    forecast_remaining_turn = None
    forecast_remaining_foot = None
    forecast_coverage_note = ""

    try:
        if isinstance(fc, pd.DataFrame) and not fc.empty and "footfall_forecast" in fc.columns and "turnover_forecast" in fc.columns:
            fc_tmp = fc.copy()
            fc_tmp["date"] = pd.to_datetime(fc_tmp["date"], errors="coerce").dt.date
            fc_tmp = fc_tmp.dropna(subset=["date"])

            rem_start = today + timedelta(days=1)
            rem_days_total = (month_end - rem_start).days + 1 if rem_start <= month_end else 0

            fc_rem = fc_tmp[(fc_tmp["date"] >= rem_start) & (fc_tmp["date"] <= month_end)].copy()

            known_rem_foot = float(fc_rem["footfall_forecast"].sum()) if not fc_rem.empty else 0.0
            known_rem_turn = float(fc_rem["turnover_forecast"].sum()) if not fc_rem.empty else 0.0
            known_days = int(fc_rem["date"].nunique()) if not fc_rem.empty else 0

            extra_days = max(0, rem_days_total - known_days)
            if known_days > 0 and extra_days > 0:
                avg_foot_per_day = known_rem_foot / known_days
                avg_turn_per_day = known_rem_turn / known_days
                extra_foot = avg_foot_per_day * extra_days
                extra_turn = avg_turn_per_day * extra_days
                forecast_coverage_note = f"Forecast covers {known_days}/{rem_days_total} remaining days → extrapolated +{extra_days} days."
            else:
                extra_foot = 0.0
                extra_turn = 0.0
                if rem_days_total > 0:
                    forecast_coverage_note = f"Forecast covers {known_days}/{rem_days_total} remaining days."

            forecast_remaining_foot = known_rem_foot + extra_foot
            forecast_remaining_turn = known_rem_turn + extra_turn

            forecast_month_foot = mtd_footfall + (forecast_remaining_foot or 0.0)
            forecast_month_turn = mtd_turnover + (forecast_remaining_turn or 0.0)
    except Exception:
        pass

    m1, m2, m3, m4 = st.columns([1, 1, 1, 1])
    with m1:
        st.metric("Turnover MTD (actual)", fmt_eur(mtd_turnover))
    with m2:
        st.metric("Turnover remaining (forecast)", fmt_eur(forecast_remaining_turn) if forecast_remaining_turn is not None else "—")
    with m3:
        st.metric("Turnover total month (forecast)", fmt_eur(forecast_month_turn) if forecast_month_turn is not None else "—")
    with m4:
        st.metric("Footfall remaining (forecast)", fmt_int(forecast_remaining_foot) if forecast_remaining_foot is not None else "—")

    if forecast_coverage_note:
        st.caption(forecast_coverage_note)

    # ---------------------------
    # Street vs store weekly chart
    # ---------------------------
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

    # ---------------------------
    # ✅ Daily chart with forecast overlay
    # ---------------------------
    st.markdown("### Daily footfall & turnover (actual + forecast)")
    
    if "footfall" in df_cur.columns and "turnover" in df_cur.columns:
        daily_df = df_cur[["date", "footfall", "turnover"]].copy()
        daily_df["date"] = pd.to_datetime(daily_df["date"], errors="coerce")
        daily_df = daily_df.dropna(subset=["date"])
    
        fig_daily = make_subplots(specs=[[{"secondary_y": True}]])
        fig_daily.add_bar(x=daily_df["date"], y=daily_df["footfall"], name="Footfall (actual)", marker_color=PFM_PURPLE)
    
        fig_daily.add_trace(
            go.Scatter(
                x=daily_df["date"], y=daily_df["turnover"],
                name="Turnover (actual)", mode="lines+markers",
                line=dict(color=PFM_PEACH, width=2),
            ),
            secondary_y=True,
        )
    
        # Add forecast overlay (if available)
        try:
            if isinstance(fc_res, dict) and fc_res.get("enough_history", False):
                fc_df = fc_res.get("forecast", pd.DataFrame()).copy()
                if isinstance(fc_df, pd.DataFrame) and not fc_df.empty:
                    fc_df["date"] = pd.to_datetime(fc_df["date"], errors="coerce")
                    fc_df = fc_df.dropna(subset=["date"])
    
                    fig_daily.add_bar(
                        x=fc_df["date"], y=fc_df["footfall_forecast"],
                        name="Footfall (forecast)", marker_color=PFM_LINE, opacity=0.75
                    )
                    fig_daily.add_trace(
                        go.Scatter(
                            x=fc_df["date"], y=fc_df["turnover_forecast"],
                            name="Turnover (forecast)", mode="lines+markers",
                            line=dict(color=PFM_RED, width=2, dash="dash"),
                        ),
                        secondary_y=True,
                    )
        except Exception:
            pass
    
        fig_daily.update_yaxes(title_text="Footfall", secondary_y=False)
        fig_daily.update_yaxes(title_text="Turnover (€)", secondary_y=True)
        fig_daily.update_layout(
            height=360,
            barmode="overlay",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=40, r=40, t=20, b=40),
        )
        st.plotly_chart(fig_daily, use_container_width=True)

    # ---------------------------
    # Weather vs footfall
    # ---------------------------
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

    # ---------------------------
    # Forecast section (still shown, but uses the same computed fc_res)
    # ---------------------------
    st.markdown("### Forecast: footfall & turnover (next 14 days)")

    if not (isinstance(fc_res, dict) and fc_res.get("enough_history", False)):
        st.info("Not enough historical data for a reliable forecast.")
    else:
        c_model = st.columns([3, 1])[1]
        with c_model:
            st.caption(f"Model: **{fc_res.get('model_type', '-') }**")
            if fc_res.get("used_simple_fallback", False):
                st.caption("Fallback → Simple DoW")

        c1, c2 = st.columns(2)
        with c1:
            delta_foot = f"{(float(fut_foot) - float(recent_foot)) / float(recent_foot) * 100:+.1f}%" if (recent_foot and float(recent_foot) > 0) else None
            st.metric("Expected visitors (14d)", fmt_int(fut_foot), delta=delta_foot)

        with c2:
            delta_turn = f"{(float(fut_turn) - float(recent_turn)) / float(recent_turn) * 100:+.1f}%" if (recent_turn and float(recent_turn) > 0) else None
            st.metric("Expected turnover (14d)", fmt_eur(fut_turn), delta=delta_turn)

        fig_fc = make_subplots(specs=[[{"secondary_y": True}]])
        if isinstance(hist_recent, pd.DataFrame) and not hist_recent.empty:
            fig_fc.add_bar(x=hist_recent["date"], y=hist_recent.get("footfall"), name="Footfall (hist)", marker_color=PFM_PURPLE)
            if "turnover" in hist_recent.columns:
                fig_fc.add_trace(
                    go.Scatter(x=hist_recent["date"], y=hist_recent["turnover"], name="Turnover (hist)", mode="lines",
                               line=dict(color=PFM_PINK, width=2)),
                    secondary_y=True,
                )

        if isinstance(fc, pd.DataFrame) and not fc.empty:
            fig_fc.add_bar(x=fc["date"], y=fc["footfall_forecast"], name="Footfall (fc)", marker_color=PFM_PEACH)
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
            recent_foot=recent_foot or 0,
            recent_turn=recent_turn or 0,
            fut_foot=fut_foot or 0,
            fut_turn=fut_turn or 0,
            spv_cur=spv_cur,
            conv_cur=conv_cur,
            fc=fc if isinstance(fc, pd.DataFrame) else pd.DataFrame(),
        )
        st.markdown(coach_text)

    # ---------------------------
    # Debug
    # ---------------------------
    with st.expander("🔧 Debug"):
        st.write("Period choice:", period_choice)
        st.write("Current period:", start_cur, "→", end_cur)
        st.write("Previous period:", start_prev, "→", end_prev)
        st.write("Shop row:", shop_row)
        st.write("Store region:", store_region)
        st.write("SVI score:", store_svi_score, "Rank:", store_svi_rank, "Peers:", store_svi_peer_n)
        st.write("SVI status:", store_svi_status)
        st.write("SVI reason:", store_svi_reason)
        st.write("SVI error:", store_svi_error)

        try:
            st.write("Meta cols:", meta.columns.tolist() if isinstance(meta, pd.DataFrame) else None)
            if isinstance(meta, pd.DataFrame) and not meta.empty:
                st.write("Meta row (selected store):", meta.loc[meta["id"] == pd.Series([shop_id], dtype="Int64").iloc[0]].head(1))
        except Exception:
            pass

        st.write("Merged meta cols:", merged_meta.columns.tolist() if isinstance(merged_meta, pd.DataFrame) else "n/a")
        try:
            st.write("Merged meta row (this store):", merged_meta[merged_meta["id"].astype("Int64") == pd.Series([shop_id], dtype="Int64").iloc[0]].head())
        except Exception:
            pass

        st.write("All daily (head):", df_all_raw.head())
        st.write("Current df (head):", df_cur.head())
        st.write("Prev df (head):", df_prev.head())
        st.write("Pathzz weekly (head):", pathzz_weekly.head())
        st.write("Capture weekly (head):", capture_weekly.head() if isinstance(capture_weekly, pd.DataFrame) else capture_weekly)
        st.write("CBS stats:", cbs_stats)
        st.write("Weather df (head):", weather_df.head())
        try:
            st.write("Forecast model_type:", fc_res.get("model_type") if isinstance(fc_res, dict) else None)
            st.write("Forecast used_simple_fallback:", fc_res.get("used_simple_fallback") if isinstance(fc_res, dict) else None)
            st.write("Forecast head:", fc_res["forecast"].head() if isinstance(fc_res, dict) and "forecast" in fc_res else None)
        except Exception:
            st.write("Forecast object not available in this run.")


if __name__ == "__main__":
    main()
