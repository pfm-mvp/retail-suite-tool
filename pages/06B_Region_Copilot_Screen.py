# pages/06B_Region_Copilot_OneScreen.py

import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt

from datetime import datetime, timedelta, date

from helpers_clients import load_clients
from helpers_normalize import normalize_vemcount_response

from services.cbs_service import (
    get_cci_series,
    get_retail_index,
)
from services.svi_service import build_store_vitality


# ----------------------
# Page config
# ----------------------
st.set_page_config(
    page_title="PFM Region Copilot (One Screen)",
    layout="wide"
)

# ----------------------
# PFM brand-ish colors (keep simple & consistent)
# ----------------------
PFM_PURPLE = "#762181"
PFM_RED = "#F04438"
PFM_DARK = "#111827"
PFM_GRAY = "#6B7280"
PFM_LIGHT = "#F3F4F6"
PFM_LINE = "#E5E7EB"
PFM_GREEN = "#22C55E"
PFM_AMBER = "#F59E0B"

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

# ----------------------
# Minimal CSS for “dashboard cards”
# (Fix: extra top padding so header/title never gets clipped)
# ----------------------
st.markdown(
    f"""
    <style>
      .block-container {{
        padding-top: 2.25rem;   /* was 1.0rem - fixes clipped header */
        padding-bottom: 2rem;
      }}
      header[data-testid="stHeader"] {{
        height: 0rem; /* keeps the Streamlit chrome from eating vertical space */
      }}
      .pfm-header {{
        display:flex;
        align-items:center;
        justify-content:space-between;
        padding: 0.85rem 1rem;
        border: 1px solid {PFM_LINE};
        border-radius: 14px;
        background: white;
        margin-bottom: 0.75rem;
      }}
      .pfm-title {{
        font-size: 1.25rem;
        font-weight: 800;
        color: {PFM_DARK};
      }}
      .pfm-sub {{
        color: {PFM_GRAY};
        font-size: 0.9rem;
        margin-top: 0.15rem;
      }}
      .kpi-card {{
        border: 1px solid {PFM_LINE};
        border-radius: 14px;
        background: white;
        padding: 0.85rem 1rem;
      }}
      .kpi-label {{
        color: {PFM_GRAY};
        font-size: 0.85rem;
        font-weight: 600;
      }}
      .kpi-value {{
        color: {PFM_DARK};
        font-size: 1.45rem;
        font-weight: 900;
        margin-top: 0.2rem;
      }}
      .kpi-help {{
        color: {PFM_GRAY};
        font-size: 0.8rem;
        margin-top: 0.25rem;
      }}
      .panel {{
        border: 1px solid {PFM_LINE};
        border-radius: 14px;
        background: white;
        padding: 0.75rem 1rem;
      }}
      .panel-title {{
        font-weight: 800;
        color: {PFM_DARK};
        margin-bottom: 0.5rem;
      }}
      div.stButton > button {{
        background: {PFM_RED} !important;
        color: white !important;
        border: 0px !important;
        border-radius: 12px !important;
        padding: 0.65rem 1rem !important;
        font-weight: 800 !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# small spacer to ensure nothing is clipped in embedded/iframed contexts
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


# ----------------------
# Format helpers
# ----------------------
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
# Region & API helpers
# ----------------------
@st.cache_data(ttl=600)
def load_region_mapping(path: str = "data/regions.csv") -> pd.DataFrame:
    """
    Supports extra columns safely (e.g., store_type).
    Required: shop_id;region
    Optional: sqm_override; store_label; store_type; ...
    """
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

    # store_type optional – do nothing if present (keeps future-proof)
    if "store_type" in df.columns:
        df["store_type"] = df["store_type"].astype(str)

    df = df.dropna(subset=["shop_id"])
    return df

@st.cache_data(ttl=600)
def get_locations_by_company(company_id: int) -> pd.DataFrame:
    url = f"{FASTAPI_BASE_URL.rstrip('/')}/company/{company_id}/location"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "locations" in data:
        return pd.DataFrame(data["locations"])
    return pd.DataFrame(data)

@st.cache_data(ttl=600)
def get_report(shop_ids, data_outputs, period: str, step: str = "day", source: str = "shops",
               date_from: str | None = None, date_to: str | None = None):
    """
    Preferred: period='date' with date_from/date_to.
    Fallback: period presets if API doesn't accept date range (we then filter locally).
    """
    params: list[tuple[str, str]] = []
    for sid in shop_ids:
        params.append(("data", str(sid)))
    for dout in data_outputs:
        params.append(("data_output", dout))
    params.append(("period", period))
    params.append(("step", step))
    params.append(("source", source))

    if period == "date":
        if date_from:
            params.append(("date_from", date_from))
        if date_to:
            params.append(("date_to", date_to))

    resp = requests.post(REPORT_URL, params=params, timeout=90)
    resp.raise_for_status()
    return resp.json()

@st.cache_data(ttl=600)
def fetch_region_street_traffic(region: str, start_date, end_date) -> pd.DataFrame:
    """
    Robust Pathzz reader supporting BOTH formats:
    - Old:  Region;Week;Visits
    - New:  shop_id;region;week;visits;store_type (or similar)
    We aggregate to: week_start -> SUM(street_footfall) for the whole region.
    """
    csv_path = "data/pathzz_sample_weekly.csv"
    try:
        df = pd.read_csv(csv_path, sep=";", dtype=str, engine="python")
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # detect likely columns
    col_region = None
    col_week = None
    col_visits = None

    for cand in ["region"]:
        if cand in df.columns:
            col_region = cand
            break
    for cand in ["week", "weekrange", "week_range"]:
        if cand in df.columns:
            col_week = cand
            break
    for cand in ["visits", "street_footfall", "streettraffic", "street_traffic"]:
        if cand in df.columns:
            col_visits = cand
            break

    # fallback if file is the old 3-col but with different headers
    if col_region is None or col_week is None or col_visits is None:
        # If it has exactly 3 columns, assume old order.
        if len(df.columns) >= 3:
            col_region = df.columns[0]
            col_week = df.columns[1]
            col_visits = df.columns[2]
        else:
            return pd.DataFrame()

    out = df[[col_region, col_week, col_visits]].copy()
    out.columns = ["region", "week", "street_footfall"]

    out["region"] = out["region"].astype(str).str.strip()
    region_norm = str(region).strip().lower()
    out = out[out["region"].str.lower() == region_norm].copy()
    if out.empty:
        return pd.DataFrame()

    out["street_footfall"] = out["street_footfall"].astype(str).str.strip().replace("", np.nan)
    out = out.dropna(subset=["street_footfall"])

    # EU number handling:
    # - Some exports use "." as thousand separator. We remove dots and parse to float.
    out["street_footfall"] = (
        out["street_footfall"]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    out["street_footfall"] = pd.to_numeric(out["street_footfall"], errors="coerce")
    out = out.dropna(subset=["street_footfall"])

    def _parse_week_start(s: str):
        if isinstance(s, str) and "to" in s.lower():
            left = s.split("To")[0].strip() if "To" in s else s.split("to")[0].strip()
            return pd.to_datetime(left, errors="coerce")
        return pd.to_datetime(s, errors="coerce")

    out["week_start"] = out["week"].apply(_parse_week_start)
    out = out.dropna(subset=["week_start"])

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    out = out[(out["week_start"] >= start) & (out["week_start"] <= end)].copy()
    if out.empty:
        return pd.DataFrame()

    # IMPORTANT FIX: aggregate to ONE value per week (whole region)
    out = out.groupby("week_start", as_index=False)["street_footfall"].sum()
    return out.reset_index(drop=True)


# ----------------------
# Robust helpers (prevents KeyError 'region')
# ----------------------
def ensure_region_column(df: pd.DataFrame, merged_map: pd.DataFrame, store_key_col: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    if "region" in df.columns:
        return df

    for cand in ("region_x", "region_y"):
        if cand in df.columns:
            out = df.copy()
            out["region"] = out[cand]
            return out

    if merged_map is None or merged_map.empty:
        return df

    if "id" not in merged_map.columns or "region" not in merged_map.columns:
        return df

    region_lookup = merged_map[["id", "region"]].drop_duplicates().copy()

    out = df.copy()
    if store_key_col in out.columns:
        out[store_key_col] = pd.to_numeric(out[store_key_col], errors="coerce").astype("Int64")
        region_lookup["id"] = pd.to_numeric(region_lookup["id"], errors="coerce").astype("Int64")
        out = out.merge(region_lookup, left_on=store_key_col, right_on="id", how="left")
        return out

    if "id" in out.columns:
        out["id"] = pd.to_numeric(out["id"], errors="coerce").astype("Int64")
        region_lookup["id"] = pd.to_numeric(region_lookup["id"], errors="coerce").astype("Int64")
        out = out.merge(region_lookup, on="id", how="left")
        return out

    return df


# ----------------------
# KPI helpers
# ----------------------
def compute_daily_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "turnover" in df.columns and "footfall" in df.columns:
        df["sales_per_visitor"] = np.where(df["footfall"] > 0, df["turnover"] / df["footfall"], np.nan)
    if "transactions" in df.columns and "footfall" in df.columns:
        df["conversion_rate"] = np.where(df["footfall"] > 0, df["transactions"] / df["footfall"] * 100, np.nan)
    return df

def aggregate_weekly_region(df_region_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Weekly aggregation for the region (SUM footfall/turnover per week).
    """
    if df_region_daily is None or df_region_daily.empty:
        return pd.DataFrame()

    df = df_region_daily.copy()
    df["week_start"] = df["date"].dt.to_period("W-SAT").dt.start_time

    agg = {}
    if "footfall" in df.columns:
        agg["footfall"] = "sum"
    if "turnover" in df.columns:
        agg["turnover"] = "sum"

    out = df.groupby("week_start", as_index=False).agg(agg)
    return out


# ----------------------
# Opportunity fallback (if svi_service does not deliver profit_potential)
# ----------------------
def compute_opportunity_fallback(df_region_daily: pd.DataFrame, store_key_col: str, store_name_col: str = "store_display") -> pd.DataFrame:
    """
    Build a "turnover uplift potential" per store for the selected period using SPV gap.
    target_spv = P75 of store SPV in region (period)
    potential_period = max(0, (target_spv - store_spv) * store_footfall)
    """
    if df_region_daily is None or df_region_daily.empty:
        return pd.DataFrame()

    req = {store_key_col, "footfall", "turnover"}
    if not req.issubset(set(df_region_daily.columns)):
        return pd.DataFrame()

    tmp = df_region_daily.copy()

    # aggregate per store for the period
    g = tmp.groupby(store_key_col, as_index=False).agg(
        footfall=("footfall", "sum"),
        turnover=("turnover", "sum"),
    )
    g["spv"] = np.where(g["footfall"] > 0, g["turnover"] / g["footfall"], np.nan)

    # target = P75 across stores (only valid spv)
    valid_spv = g["spv"].dropna()
    if valid_spv.empty:
        return pd.DataFrame()

    target_spv = float(valid_spv.quantile(0.75))
    g["target_spv"] = target_spv
    g["spv_gap"] = g["target_spv"] - g["spv"]
    g["profit_potential_period"] = np.where(g["spv_gap"] > 0, g["spv_gap"] * g["footfall"], 0.0)

    # attach names if present
    if store_name_col in tmp.columns:
        name_map = tmp[[store_key_col, store_name_col]].drop_duplicates()
        g = g.merge(name_map, on=store_key_col, how="left")
        g = g.rename(columns={store_name_col: "store_name"})
    else:
        g["store_name"] = g[store_key_col].astype(str)

    return g[[store_key_col, "store_name", "profit_potential_period", "footfall", "spv", "target_spv"]]


# ----------------------
# Small UI helpers
# ----------------------
def kpi_card(label: str, value: str, help_text: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def status_from_score(score: float):
    if score >= 75:
        return "High performance", PFM_GREEN
    if score >= 60:
        return "Good / stable", PFM_PURPLE
    if score >= 45:
        return "Attention required", PFM_AMBER
    return "Under pressure", PFM_RED

def gauge_chart(score_0_100: float, fill_color: str):
    score_0_100 = float(np.clip(score_0_100, 0, 100))
    gauge_df = pd.DataFrame({"segment": ["filled", "empty"], "value": [score_0_100, max(0.0, 100.0 - score_0_100)]})
    arc = (
        alt.Chart(gauge_df)
        .mark_arc(innerRadius=62, outerRadius=82)
        .encode(
            theta="value:Q",
            color=alt.Color(
                "segment:N",
                scale=alt.Scale(domain=["filled", "empty"], range=[fill_color, PFM_LINE]),
                legend=None,
            ),
        )
        .properties(width=240, height=240)
    )
    text = (
        alt.Chart(pd.DataFrame({"label": [f"{score_0_100:.0f}"]}))
        .mark_text(size=34, fontWeight="bold", color=PFM_DARK)
        .encode(text="label:N")
    )
    return arc + text


# ----------------------
# Period presets
# ----------------------
def period_options() -> list[str]:
    return [
        "Kalenderjaar 2024",
        "Kalenderjaar 2025",
        "Q1 2024",
        "Q2 2024",
        "Q3 2024",
        "Q4 2024",
        "Laatste 13 weken",
        "Laatste 26 weken",
        "YTD (dit jaar)",
    ]

def resolve_period(label: str) -> tuple[date, date]:
    today = datetime.today().date()

    if label == "Kalenderjaar 2024":
        return date(2024, 1, 1), date(2024, 12, 31)
    if label == "Kalenderjaar 2025":
        return date(2025, 1, 1), date(2025, 12, 31)

    if label == "Q1 2024":
        return date(2024, 1, 1), date(2024, 3, 31)
    if label == "Q2 2024":
        return date(2024, 4, 1), date(2024, 6, 30)
    if label == "Q3 2024":
        return date(2024, 7, 1), date(2024, 9, 30)
    if label == "Q4 2024":
        return date(2024, 10, 1), date(2024, 12, 31)

    if label == "Laatste 13 weken":
        return today - timedelta(weeks=13), today
    if label == "Laatste 26 weken":
        return today - timedelta(weeks=26), today

    if label == "YTD (dit jaar)":
        return date(today.year, 1, 1), today

    # fallback
    return today - timedelta(weeks=26), today


# ----------------------
# MAIN
# ----------------------
def main():
    header_left, header_right = st.columns([2.2, 1.8])

    with header_left:
        st.markdown(
            f"""
            <div class="pfm-header">
              <div>
                <div class="pfm-title">PFM Region Performance Copilot</div>
                <div class="pfm-sub">One-screen layout – snel lezen, weinig scrollen (macro optioneel onderaan)</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    clients = load_clients("clients.json")
    clients_df = pd.DataFrame(clients)
    if clients_df.empty:
        st.error("clients.json leeg of niet gevonden.")
        return

    clients_df["label"] = clients_df.apply(
        lambda r: f"{r['brand']} – {r['name']} (company_id {r['company_id']})",
        axis=1,
    )

    with header_right:
        c1, c2 = st.columns(2)
        with c1:
            client_label = st.selectbox("Retailer", clients_df["label"].tolist(), label_visibility="collapsed")
        with c2:
            period_choice = st.selectbox(
                "Periode",
                period_options(),
                index=0,
                label_visibility="collapsed",
            )

    selected_client = clients_df[clients_df["label"] == client_label].iloc[0].to_dict()
    company_id = int(selected_client["company_id"])

    # Load locations + regions
    try:
        locations_df = get_locations_by_company(company_id)
    except requests.exceptions.RequestException as e:
        st.error(f"Fout bij ophalen van winkels uit FastAPI: {e}")
        return

    if locations_df.empty:
        st.error("Geen winkels gevonden voor deze retailer.")
        return

    region_map = load_region_mapping()
    if region_map.empty:
        st.error("Geen geldige data/regions.csv gevonden (minimaal: shop_id;region).")
        return

    locations_df["id"] = pd.to_numeric(locations_df["id"], errors="coerce").astype("Int64")
    merged = locations_df.merge(region_map, left_on="id", right_on="shop_id", how="inner")
    if merged.empty:
        st.warning("Er zijn geen winkels met een regio-mapping voor deze retailer.")
        return

    # sqm_effective + store_display
    if "sqm" in merged.columns:
        merged["sqm_effective"] = np.where(
            merged["sqm_override"].notna(),
            merged["sqm_override"],
            pd.to_numeric(merged["sqm"], errors="coerce"),
        )
    else:
        merged["sqm_effective"] = merged["sqm_override"]

    if "store_label" in merged.columns and merged["store_label"].notna().any():
        merged["store_display"] = merged["store_label"]
    else:
        merged["store_display"] = merged["name"] if "name" in merged.columns else merged["id"].astype(str)

    # Top controls row
    available_regions = sorted(merged["region"].dropna().unique().tolist())
    top_controls = st.columns([1.4, 1.2, 1.2, 1.4, 1.0])
    with top_controls[0]:
        region_choice = st.selectbox("Regio", available_regions)
    with top_controls[1]:
        compare_all_regions = st.toggle("Vergelijk met andere regio’s", value=True)
    with top_controls[2]:
        show_macro = st.toggle("Toon macro (CBS/CCI)", value=False)
    with top_controls[3]:
        # (kept empty as spacer / future filter)
        st.markdown("<div style='height:1px'></div>", unsafe_allow_html=True)
    with top_controls[4]:
        run_btn = st.button("Analyseer", type="primary")

    if not run_btn:
        st.info("Selecteer retailer/regio/periode en klik op **Analyseer**.")
        return

    # Resolve period
    start_period, end_period = resolve_period(period_choice)
    start_ts = pd.Timestamp(start_period)
    end_ts = pd.Timestamp(end_period)

    # Shop IDs
    region_shops = merged[merged["region"] == region_choice].copy()
    region_shop_ids = region_shops["id"].dropna().astype(int).unique().tolist()
    if not region_shop_ids:
        st.warning(f"Geen winkels gevonden voor regio '{region_choice}'.")
        return

    all_shop_ids = merged["id"].dropna().astype(int).unique().tolist()
    fetch_ids = all_shop_ids if compare_all_regions else region_shop_ids

    # Fetch report
    metric_map = {"count_in": "footfall", "turnover": "turnover"}

    # Prefer period=date (so 2024 works even if we're in 2026)
    with st.spinner("Data ophalen via FastAPI..."):
        try:
            resp = get_report(
                fetch_ids,
                list(metric_map.keys()),
                period="date",
                step="day",
                source="shops",
                date_from=str(start_period),
                date_to=str(end_period),
            )
        except Exception as e:
            # Fallback to this_year and filter locally (in case API doesn't accept date_from/date_to yet)
            st.warning(f"API date-range call faalde ({e}). Probeer fallback (this_year) + lokale filtering.")
            resp = get_report(
                fetch_ids,
                list(metric_map.keys()),
                period="this_year",
                step="day",
                source="shops",
            )

        df_raw = normalize_vemcount_response(resp, kpi_keys=metric_map.keys()).rename(columns=metric_map)

    if df_raw.empty:
        st.warning("Geen data ontvangen voor de gekozen selectie.")
        return

    # Identify store id column
    df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
    df_raw = df_raw.dropna(subset=["date"])

    store_key_col = None
    for cand in ["id", "shop_id", "location_id"]:
        if cand in df_raw.columns:
            store_key_col = cand
            break
    if store_key_col is None:
        st.error("Geen store-id kolom gevonden in de response (id/shop_id/location_id).")
        return

    # Filter to selected period (even if we already called date-range, keep it safe)
    df_period_all = df_raw[(df_raw["date"] >= start_ts) & (df_raw["date"] <= end_ts)].copy()
    if df_period_all.empty:
        st.warning("Geen data in de geselecteerde periode.")
        return

    df_period_all = compute_daily_kpis(df_period_all)

    # Join store metadata (all)
    join_cols = ["id", "store_display", "region", "sqm_effective"]
    if "store_type" in merged.columns:
        join_cols.append("store_type")

    df_period_all = df_period_all.merge(
        merged[join_cols].drop_duplicates(),
        left_on=store_key_col,
        right_on="id",
        how="left",
    )

    # Region-only view
    df_region = df_period_all[df_period_all["region"] == region_choice].copy()
    if df_region.empty:
        st.warning("Geen data voor geselecteerde regio binnen de periode.")
        return

    # Weekly region aggregation + Pathzz street traffic
    region_weekly = aggregate_weekly_region(df_region)
    pathzz_weekly = fetch_region_street_traffic(region=region_choice, start_date=start_period, end_date=end_period)

    capture_weekly = pd.DataFrame()
    avg_capture = np.nan
    if not region_weekly.empty and not pathzz_weekly.empty:
        region_weekly["week_start"] = pd.to_datetime(region_weekly["week_start"])
        pathzz_weekly["week_start"] = pd.to_datetime(pathzz_weekly["week_start"])

        capture_weekly = pd.merge(region_weekly, pathzz_weekly, on="week_start", how="inner")
        if not capture_weekly.empty:
            # IMPORTANT FIX: capture based on TOTAL region footfall / TOTAL region street footfall per week
            capture_weekly["capture_rate"] = np.where(
                capture_weekly["street_footfall"] > 0,
                capture_weekly["footfall"] / capture_weekly["street_footfall"] * 100.0,
                np.nan,
            )
            avg_capture = float(capture_weekly["capture_rate"].mean())

    # KPI cards
    foot_total = float(df_region["footfall"].sum()) if "footfall" in df_region.columns else 0.0
    turn_total = float(df_region["turnover"].sum()) if "turnover" in df_region.columns else 0.0
    spv_avg = float(df_region["sales_per_visitor"].mean()) if "sales_per_visitor" in df_region.columns else np.nan

    # SVI calculations
    svi_all = build_store_vitality(
        df_period=df_period_all,
        region_shops=merged,
        store_key_col=store_key_col,
    )

    svi_all = ensure_region_column(svi_all, merged, store_key_col) if isinstance(svi_all, pd.DataFrame) else pd.DataFrame()

    # Ensure store_name exists (some svi implementations output ids)
    if isinstance(svi_all, pd.DataFrame) and not svi_all.empty:
        if "store_name" not in svi_all.columns:
            # best effort mapping
            name_map = merged[["id", "store_display"]].drop_duplicates().rename(columns={"store_display": "store_name"})
            if store_key_col in svi_all.columns:
                svi_all[store_key_col] = pd.to_numeric(svi_all[store_key_col], errors="coerce").astype("Int64")
                name_map["id"] = pd.to_numeric(name_map["id"], errors="coerce").astype("Int64")
                svi_all = svi_all.merge(name_map, left_on=store_key_col, right_on="id", how="left")
            elif "id" in svi_all.columns:
                svi_all["id"] = pd.to_numeric(svi_all["id"], errors="coerce").astype("Int64")
                name_map["id"] = pd.to_numeric(name_map["id"], errors="coerce").astype("Int64")
                svi_all = svi_all.merge(name_map, on="id", how="left")

    region_scores = pd.DataFrame()
    region_svi = np.nan
    region_status, region_color = ("-", PFM_LINE)

    if not svi_all.empty and "region" in svi_all.columns and "svi_score" in svi_all.columns:
        svi_all["svi_score"] = pd.to_numeric(svi_all["svi_score"], errors="coerce")
        region_scores = (
            svi_all.groupby("region", as_index=False)["svi_score"]
            .mean()
            .rename(columns={"svi_score": "region_svi"})
            .dropna(subset=["region"])
        )
        cur = region_scores[region_scores["region"] == region_choice]
        if not cur.empty and cur["region_svi"].notna().any():
            region_svi = float(np.clip(cur["region_svi"].iloc[0], 0, 100))
            region_status, region_color = status_from_score(region_svi)

    svi_region = pd.DataFrame()
    if not svi_all.empty and "region" in svi_all.columns:
        svi_region = svi_all[svi_all["region"] == region_choice].copy()

    if not svi_region.empty and "svi_score" in svi_region.columns:
        svi_region["svi_score"] = pd.to_numeric(svi_region["svi_score"], errors="coerce")
        svi_region = svi_region.dropna(subset=["svi_score"]).sort_values("svi_score", ascending=False).reset_index(drop=True)
        svi_region["rank_in_region"] = np.arange(1, len(svi_region) + 1)

    # Title
    st.markdown(f"## {selected_client['brand']} — Regio **{region_choice}** · {start_period} → {end_period}")

    # KPI row
    k1, k2, k3, k4, k5 = st.columns([1, 1, 1, 1, 1.2])
    with k1:
        kpi_card("Footfall", fmt_int(foot_total), "Regio · periode")
    with k2:
        kpi_card("Omzet", fmt_eur(turn_total), "Regio · periode")
    with k3:
        kpi_card("SPV", (f"€ {spv_avg:.2f}".replace(".", ",") if not pd.isna(spv_avg) else "-"), "Gemiddelde")
    with k4:
        kpi_card("Capture", (fmt_pct(avg_capture) if not pd.isna(avg_capture) else "-"), "Gemiddeld (Pathzz)")
    with k5:
        st.markdown('<div class="panel"><div class="panel-title">Regio Vitality</div>', unsafe_allow_html=True)
        if not pd.isna(region_svi):
            st.altair_chart(gauge_chart(region_svi, region_color), use_container_width=True)
            st.markdown(f"**{region_svi:.0f}** · {region_status}")
        else:
            st.info("Nog geen regio-score.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 2: weekly trend + region compare
    r2_left, r2_right = st.columns([1.6, 1.0])

    with r2_left:
        st.markdown('<div class="panel"><div class="panel-title">Weekly trend — Store vs Street + Capture</div>', unsafe_allow_html=True)

        if capture_weekly.empty:
            st.info("Geen matchende Pathzz-weekdata gevonden voor deze regio/periode.")
        else:
            chart_df = capture_weekly[["week_start", "footfall", "street_footfall", "turnover", "capture_rate"]].copy()
            chart_df = chart_df.sort_values("week_start").reset_index(drop=True)

            iso = chart_df["week_start"].dt.isocalendar()
            chart_df["week_label"] = iso.week.apply(lambda w: f"W{int(w):02d}")
            week_order = chart_df["week_label"].tolist()

            long = chart_df.melt(
                id_vars=["week_label"],
                value_vars=["footfall", "street_footfall", "turnover"],
                var_name="metric",
                value_name="value",
            )

            bar = (
                alt.Chart(long)
                .mark_bar(opacity=0.85, cornerRadiusEnd=4)
                .encode(
                    x=alt.X("week_label:N", sort=week_order, title=None),
                    xOffset=alt.XOffset("metric:N"),
                    y=alt.Y("value:Q", title=""),
                    color=alt.Color(
                        "metric:N",
                        scale=alt.Scale(
                            domain=["footfall", "street_footfall", "turnover"],
                            range=[PFM_PURPLE, PFM_LINE, PFM_RED],
                        ),
                        legend=alt.Legend(title=""),
                    ),
                    tooltip=[
                        alt.Tooltip("week_label:N", title="Week"),
                        alt.Tooltip("metric:N", title="Type"),
                        alt.Tooltip("value:Q", title="Waarde", format=",.0f"),
                    ],
                )
            )

            # IMPORTANT FIX: line is based on aggregated capture_weekly (one row per week)
            line = (
                alt.Chart(chart_df)
                .mark_line(point=True, strokeWidth=2, color=PFM_DARK)
                .encode(
                    x=alt.X("week_label:N", sort=week_order, title=None),
                    y=alt.Y("capture_rate:Q", title="Capture %"),
                    tooltip=[
                        alt.Tooltip("week_label:N", title="Week"),
                        alt.Tooltip("capture_rate:Q", title="Capture", format=".1f"),
                    ],
                )
            )

            st.altair_chart(
                alt.layer(bar, line).resolve_scale(y="independent").properties(height=280),
                use_container_width=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with r2_right:
        st.markdown('<div class="panel"><div class="panel-title">Regio vergelijking — RVI (SVI gemiddeld)</div>', unsafe_allow_html=True)

        if not compare_all_regions:
            st.info("Zet ‘Vergelijk met andere regio’s’ aan om alle regio’s te tonen.")
        elif region_scores.empty or region_scores["region"].nunique() <= 1:
            st.info("Nog onvoldoende regio’s of data om te vergelijken.")
        else:
            chart_regions = region_scores.copy()
            chart_regions["is_selected"] = chart_regions["region"] == region_choice

            region_chart = (
                alt.Chart(chart_regions.sort_values("region_svi", ascending=False))
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("region_svi:Q", title="RVI (0–100)", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("region:N", sort="-x", title=None),
                    color=alt.Color(
                        "is_selected:N",
                        scale=alt.Scale(domain=[True, False], range=[PFM_PURPLE, PFM_LINE]),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("region:N", title="Regio"),
                        alt.Tooltip("region_svi:Q", title="RVI", format=".0f"),
                    ],
                )
                .properties(height=260)
            )

            st.altair_chart(region_chart, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Row 3: store ranking + opportunities
    r3_left, r3_right = st.columns([1.6, 1.0])

    with r3_left:
        st.markdown('<div class="panel"><div class="panel-title">Store Vitality ranking — geselecteerde regio</div>', unsafe_allow_html=True)

        if svi_region.empty:
            st.info("Geen stores in deze regio met SVI (of regio-koppeling ontbreekt).")
        else:
            # best effort for store_name column
            if "store_name" not in svi_region.columns:
                if "store_display" in svi_region.columns:
                    svi_region["store_name"] = svi_region["store_display"]
                elif store_key_col in svi_region.columns:
                    svi_region["store_name"] = svi_region[store_key_col].astype(str)
                else:
                    svi_region["store_name"] = "Unknown"

            svi_region["svi_score"] = pd.to_numeric(svi_region["svi_score"], errors="coerce")
            svi_region = svi_region.dropna(subset=["svi_score"])

            if svi_region.empty:
                st.info("Geen valide SVI-scores gevonden.")
            else:
                # svi_status might not exist in some implementations
                if "svi_status" not in svi_region.columns:
                    svi_region["svi_status"] = svi_region["svi_score"].apply(
                        lambda s: status_from_score(float(s))[0] if pd.notna(s) else "Unknown"
                    )

                if "reason_short" not in svi_region.columns:
                    svi_region["reason_short"] = ""

                chart_rank = (
                    alt.Chart(svi_region.sort_values("svi_score", ascending=False).head(12))
                    .mark_bar(cornerRadiusEnd=4)
                    .encode(
                        x=alt.X("svi_score:Q", title="SVI (0–100)", scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y("store_name:N", sort="-x", title=None),
                        color=alt.Color(
                            "svi_status:N",
                            scale=alt.Scale(
                                domain=["High performance", "Good / stable", "Attention required", "Under pressure"],
                                range=[PFM_GREEN, PFM_PURPLE, PFM_AMBER, PFM_RED],
                            ),
                            legend=alt.Legend(title=""),
                        ),
                        tooltip=[
                            alt.Tooltip("store_name:N", title="Winkel"),
                            alt.Tooltip("svi_score:Q", title="SVI", format=".0f"),
                            alt.Tooltip("svi_status:N", title="Status"),
                            alt.Tooltip("reason_short:N", title="Waarom"),
                        ],
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart_rank, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with r3_right:
        st.markdown('<div class="panel"><div class="panel-title">Biggest opportunities</div>', unsafe_allow_html=True)

        # Determine opportunity source: prefer svi output, else fallback compute from df_region
        opp_source = None
        if not svi_region.empty and "profit_potential_period" in svi_region.columns:
            tmp = svi_region.copy()
            tmp["profit_potential_period"] = pd.to_numeric(tmp["profit_potential_period"], errors="coerce").fillna(0.0)
            if tmp["profit_potential_period"].sum() > 0:
                opp_source = tmp[["store_name", "profit_potential_period"]].copy()

        if opp_source is None:
            # fallback: compute from SPV gap
            fb = compute_opportunity_fallback(df_region, store_key_col=store_key_col, store_name_col="store_display")
            if not fb.empty:
                opp_source = fb.rename(columns={"profit_potential_period": "profit_potential_period"})[["store_name", "profit_potential_period"]]

        if opp_source is None or opp_source.empty:
            st.info("Nog geen opportunity data (profit potential kon niet worden berekend).")
        else:
            period_days = (end_ts - start_ts).days + 1
            year_factor = 365.0 / period_days if period_days > 0 else 1.0

            opp_source["profit_potential_period"] = pd.to_numeric(opp_source["profit_potential_period"], errors="coerce").fillna(0.0)
            opp_source["profit_potential_year"] = opp_source["profit_potential_period"] * year_factor

            opp = (
                opp_source[["store_name", "profit_potential_year"]]
                .dropna()
                .sort_values("profit_potential_year", ascending=False)
                .head(6)
            )

            # if all zeros, show message
            if opp["profit_potential_year"].sum() <= 0:
                st.info("Opportunity berekend, maar alles komt op €0 uit (check turnover/footfall/SPV input).")
            else:
                opp_chart = (
                    alt.Chart(opp)
                    .mark_bar(cornerRadiusEnd=4, color=PFM_RED)
                    .encode(
                        x=alt.X("profit_potential_year:Q", title="€ / jaar", axis=alt.Axis(format=",.0f")),
                        y=alt.Y("store_name:N", sort="-x", title=None),
                        tooltip=[
                            alt.Tooltip("store_name:N", title="Winkel"),
                            alt.Tooltip("profit_potential_year:Q", title="€ / jaar", format=",.0f"),
                        ],
                    )
                    .properties(height=300)
                )
                st.altair_chart(opp_chart, use_container_width=True)

                total_top5 = float(opp["profit_potential_year"].head(5).sum())
                st.markdown(f"**Top 5 samen:** {fmt_eur(total_top5)} / jaar")

        st.markdown("</div>", unsafe_allow_html=True)

    # Macro section (toggle)
    if show_macro:
        st.markdown("### Macro-context (optioneel)")
        st.caption("Onderstaand mag scrollen — maar staat bewust *onder* je 1-screen dashboard.")

        macro_col1, macro_col2 = st.columns(2)

        region_month = df_region.copy()
        region_month["month"] = region_month["date"].dt.to_period("M").dt.to_timestamp()
        region_month = (
            region_month.groupby("month", as_index=False)[["turnover", "footfall"]]
            .sum()
            .rename(columns={"turnover": "region_turnover", "footfall": "region_footfall"})
        )

        def index_from_first_nonzero(s: pd.Series) -> pd.Series:
            s = pd.to_numeric(s, errors="coerce").astype(float)
            nonzero = s.replace(0, np.nan).dropna()
            if nonzero.empty:
                return pd.Series(np.nan, index=s.index)
            base_idx = nonzero.index[0]
            base_val = nonzero.iloc[0]
            idx = s / base_val * 100.0
            idx.loc[s.index < base_idx] = np.nan
            return idx

        if not region_month.empty:
            region_month["region_turnover_index"] = index_from_first_nonzero(region_month["region_turnover"])
            region_month["region_footfall_index"] = index_from_first_nonzero(region_month["region_footfall"])

        with macro_col1:
            st.markdown('<div class="panel"><div class="panel-title">CBS detailhandelindex vs Regio</div>', unsafe_allow_html=True)
            try:
                retail_series = get_retail_index(months_back=24)
            except Exception:
                retail_series = []

            cbs_retail_month = pd.DataFrame()
            if retail_series:
                cbs_retail_df = pd.DataFrame(retail_series)
                cbs_retail_df["date"] = pd.to_datetime(
                    cbs_retail_df["period"].str[:4] + "-" + cbs_retail_df["period"].str[-2:] + "-15",
                    errors="coerce",
                )
                cbs_retail_df = cbs_retail_df.dropna(subset=["date"])
                cbs_retail_month = cbs_retail_df.groupby("date", as_index=False)["retail_value"].mean()
                if not cbs_retail_month.empty and cbs_retail_month["retail_value"].notna().any():
                    base = cbs_retail_month["retail_value"].dropna().iloc[0]
                    cbs_retail_month["cbs_retail_index"] = np.where(base != 0, cbs_retail_month["retail_value"] / base * 100.0, np.nan)

            lines = []
            if not region_month.empty:
                a = region_month.rename(columns={"month": "date"})[["date", "region_footfall_index"]].copy()
                a["series"] = "Regio footfall-index"
                a = a.rename(columns={"region_footfall_index": "value"})
                lines.append(a)

                b = region_month.rename(columns={"month": "date"})[["date", "region_turnover_index"]].copy()
                b["series"] = "Regio omzet-index"
                b = b.rename(columns={"region_turnover_index": "value"})
                lines.append(b)

            if not cbs_retail_month.empty and "cbs_retail_index" in cbs_retail_month.columns:
                c = cbs_retail_month[["date", "cbs_retail_index"]].copy()
                c["series"] = "CBS detailhandelindex"
                c = c.rename(columns={"cbs_retail_index": "value"})
                lines.append(c)

            if lines:
                macro = (
                    alt.Chart(pd.concat(lines, ignore_index=True))
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("date:T", title="Maand"),
                        y=alt.Y("value:Q", title="Index (100 = eerste maand)"),
                        color=alt.Color("series:N", title=""),
                        tooltip=[
                            alt.Tooltip("date:T", title="Maand"),
                            alt.Tooltip("series:N", title="Reeks"),
                            alt.Tooltip("value:Q", title="Index", format=".1f"),
                        ],
                    )
                    .properties(height=260)
                )
                st.altair_chart(macro, use_container_width=True)
            else:
                st.info("Geen macro-lijnen beschikbaar.")

            st.markdown("</div>", unsafe_allow_html=True)

        with macro_col2:
            st.markdown('<div class="panel"><div class="panel-title">Consumentenvertrouwen (CCI) vs Regio</div>', unsafe_allow_html=True)
            try:
                cci_series = get_cci_series(months_back=24)
            except Exception:
                cci_series = []

            cci_df = pd.DataFrame()
            if cci_series:
                cci_df = pd.DataFrame(cci_series)
                cci_df["date"] = pd.to_datetime(
                    cci_df["period"].str[:4] + "-" + cci_df["period"].str[-2:] + "-15",
                    errors="coerce",
                )
                cci_df = cci_df.dropna(subset=["date"])
                if not cci_df.empty and cci_df["cci"].notna().any():
                    base = cci_df["cci"].dropna().iloc[0]
                    cci_df["cci_index"] = np.where(base != 0, cci_df["cci"] / base * 100.0, np.nan)

            lines = []
            if not cci_df.empty and "cci_index" in cci_df.columns:
                c = cci_df[["date", "cci_index"]].copy()
                c["series"] = "Consumentenvertrouwen-index"
                c = c.rename(columns={"cci_index": "value"})
                lines.append(c)

            if not region_month.empty:
                a = region_month.rename(columns={"month": "date"})[["date", "region_footfall_index"]].copy()
                a["series"] = "Regio footfall-index"
                a = a.rename(columns={"region_footfall_index": "value"})
                lines.append(a)

                b = region_month.rename(columns={"month": "date"})[["date", "region_turnover_index"]].copy()
                b["series"] = "Regio omzet-index"
                b = b.rename(columns={"region_turnover_index": "value"})
                lines.append(b)

            if lines:
                cc = (
                    alt.Chart(pd.concat(lines, ignore_index=True))
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("date:T", title="Maand"),
                        y=alt.Y("value:Q", title="Index (100 = eerste maand)"),
                        color=alt.Color("series:N", title=""),
                        tooltip=[
                            alt.Tooltip("date:T", title="Maand"),
                            alt.Tooltip("series:N", title="Reeks"),
                            alt.Tooltip("value:Q", title="Index", format=".1f"),
                        ],
                    )
                    .properties(height=260)
                )
                st.altair_chart(cc, use_container_width=True)
            else:
                st.info("Geen CCI-data beschikbaar.")

            st.markdown("</div>", unsafe_allow_html=True)

    # Optional debug expander
    with st.expander("🔧 Debug"):
        st.write("Retailer:", selected_client)
        st.write("Regio:", region_choice)
        st.write("Periode:", start_period, "→", end_period, "| preset:", period_choice)
        st.write("Compare all regions:", compare_all_regions)
        st.write("Store key col:", store_key_col)
        st.write("All shops:", len(all_shop_ids), "Region shops:", len(region_shop_ids))
        st.write("REPORT_URL:", REPORT_URL)
        st.write("df_period_all head:", df_period_all.head())
        st.write("df_region head:", df_region.head())
        st.write("capture_weekly head:", capture_weekly.head())
        st.write("svi_all cols:", svi_all.columns.tolist() if isinstance(svi_all, pd.DataFrame) else "n/a")
        if isinstance(svi_all, pd.DataFrame) and not svi_all.empty:
            st.write("svi_all head:", svi_all.head())


if __name__ == "__main__":
    main()
