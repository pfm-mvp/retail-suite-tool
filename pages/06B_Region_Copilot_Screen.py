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
# Colors
# ----------------------
PFM_PURPLE = "#762181"
PFM_RED = "#F04438"
PFM_DARK = "#111827"
PFM_GRAY = "#6B7280"
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
# CSS
# ----------------------
st.markdown(
    f"""
    <style>
      .block-container {{
        padding-top: 2.25rem;
        padding-bottom: 2rem;
      }}

      .pfm-header {{
        display:flex;
        align-items:center;
        justify-content:space-between;
        padding: 0.85rem 1rem;
        border: 1px solid {PFM_LINE};
        border-radius: 14px;
        background: white;
        margin-bottom: 0.85rem;
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
        height: 118px;
      }}
      .kpi-label {{
        color: {PFM_GRAY};
        font-size: 0.85rem;
        font-weight: 700;
      }}
      .kpi-value {{
        color: {PFM_DARK};
        font-size: 1.55rem;
        font-weight: 900;
        margin-top: 0.2rem;
      }}
      .kpi-help {{
        color: {PFM_GRAY};
        font-size: 0.85rem;
        margin-top: 0.35rem;
      }}

      .panel {{
        border: 1px solid {PFM_LINE};
        border-radius: 14px;
        background: white;
        padding: 0.85rem 1rem;
      }}
      .panel-title {{
        font-weight: 900;
        color: {PFM_DARK};
        margin-bottom: 0.5rem;
        font-size: 1.02rem;
      }}

      div.stButton > button {{
        background: {PFM_RED} !important;
        color: white !important;
        border: 0px !important;
        border-radius: 14px !important;
        padding: 0.75rem 1.1rem !important;
        font-weight: 900 !important;
      }}

      [data-testid="stHorizontalBlock"] {{
        gap: 0.9rem;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

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

def fmt_eur_dec(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"€ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# ----------------------
# Data loaders
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

    if "store_type" in df.columns:
        df["store_type"] = df["store_type"].astype(str)
    else:
        df["store_type"] = np.nan

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

    resp = requests.post(REPORT_URL, params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()

@st.cache_data(ttl=600)
def fetch_region_street_traffic_weekly(shop_ids_in_region: list[int], region_label: str, start_date, end_date) -> pd.DataFrame:
    """
    Supports both:
    A) region;week;visits
    B) shop_id;region;week;visits;...
    We SUM street traffic across the region shops per week.
    """
    csv_path = "data/pathzz_sample_weekly.csv"

    try:
        df = pd.read_csv(csv_path, sep=";", header=0, dtype=str, engine="python")
    except Exception:
        return pd.DataFrame()

    df.columns = [c.strip().lower() for c in df.columns]

    col_region = "region" if "region" in df.columns else None
    col_week = "week" if "week" in df.columns else None

    col_visits = None
    for c in df.columns:
        if c in ("visits", "street_footfall", "streettraffic", "street_traffic"):
            col_visits = c
            break

    col_shop = None
    for c in df.columns:
        if c in ("shop_id", "shopid", "store_id", "location_id", "id"):
            col_shop = c
            break

    if col_week is None or col_visits is None:
        return pd.DataFrame()

    # parse visits (EU)
    df[col_visits] = df[col_visits].astype(str).str.strip().replace("", np.nan)
    df = df.dropna(subset=[col_visits])
    df[col_visits] = (
        df[col_visits]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    def _parse_week_start(s: str):
        if isinstance(s, str) and "To" in s:
            return pd.to_datetime(s.split("To")[0].strip(), errors="coerce")
        return pd.NaT

    df["week_start"] = df[col_week].astype(str).str.strip().apply(_parse_week_start)
    df = df.dropna(subset=["week_start"])

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    df = df[(df["week_start"] >= start) & (df["week_start"] <= end)]
    if df.empty:
        return pd.DataFrame()

    # prefer shop_id filtering
    if col_shop is not None and shop_ids_in_region:
        df[col_shop] = pd.to_numeric(df[col_shop], errors="coerce").astype("Int64")
        df = df[df[col_shop].isin(set([int(x) for x in shop_ids_in_region]))]
    else:
        if col_region is None:
            return pd.DataFrame()
        region_norm = str(region_label).strip().lower()
        df[col_region] = df[col_region].astype(str).str.strip()
        df = df[df[col_region].str.lower() == region_norm]

    if df.empty:
        return pd.DataFrame()

    out = (
        df.groupby("week_start", as_index=False)[col_visits]
        .sum()
        .rename(columns={col_visits: "street_footfall"})
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    return out

# ----------------------
# KPI helpers
# ----------------------
def compute_daily_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("footfall", "turnover", "transactions", "sales_per_visitor", "conversion_rate"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "sales_per_visitor" not in df.columns and "turnover" in df.columns and "footfall" in df.columns:
        df["sales_per_visitor"] = np.where(df["footfall"] > 0, df["turnover"] / df["footfall"], np.nan)

    if "conversion_rate" not in df.columns and "transactions" in df.columns and "footfall" in df.columns:
        df["conversion_rate"] = np.where(df["footfall"] > 0, df["transactions"] / df["footfall"] * 100.0, np.nan)

    return df

def aggregate_weekly_region(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    x = df.copy()
    x["week_start"] = x["date"].dt.to_period("W-SAT").dt.start_time

    agg = {"footfall": "sum", "turnover": "sum"}
    if "transactions" in x.columns:
        agg["transactions"] = "sum"

    out = x.groupby("week_start", as_index=False).agg(agg)
    out["sales_per_visitor"] = np.where(out["footfall"] > 0, out["turnover"] / out["footfall"], np.nan)
    if "transactions" in out.columns:
        out["conversion_rate"] = np.where(out["footfall"] > 0, out["transactions"] / out["footfall"] * 100.0, np.nan)
    return out.sort_values("week_start").reset_index(drop=True)

# ----------------------
# UI helpers
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
        .mark_arc(innerRadius=62, outerRadius=84)
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
        .mark_text(size=38, fontWeight="bold", color=PFM_DARK)
        .encode(text="label:N")
    )
    return arc + text

# ----------------------
# Period logic
# ----------------------
def period_options():
    return [
        "Kalenderjaar 2024",
        "Kalenderjaar 2025",
        "Q1 2024",
        "Q2 2024",
        "Q3 2024",
        "Q4 2024",
        "Q1 2025",
        "Q2 2025",
        "Q3 2025",
        "Q4 2025",
        "Laatste 26 weken",
    ]

def resolve_period_dates(choice: str):
    today = datetime.today().date()

    if choice == "Laatste 26 weken":
        end_period = today
        start_period = today - timedelta(weeks=26)
        macro_year = end_period.year
        return start_period, end_period, macro_year

    if choice.startswith("Kalenderjaar"):
        y = int(choice.split()[-1])
        return date(y, 1, 1), date(y, 12, 31), y

    if choice.startswith("Q"):
        q, y = choice.split()
        y = int(y)
        qn = int(q.replace("Q", ""))
        if qn == 1:
            return date(y, 1, 1), date(y, 3, 31), y
        if qn == 2:
            return date(y, 4, 1), date(y, 6, 30), y
        if qn == 3:
            return date(y, 7, 1), date(y, 9, 30), y
        if qn == 4:
            return date(y, 10, 1), date(y, 12, 31), y

    return date(today.year, 1, 1), today, today.year

# ----------------------
# Opportunities (robust + explainable)
# ----------------------
def compute_opportunities_store_level(df_period: pd.DataFrame, store_key_col: str) -> pd.DataFrame:
    """
    Opportunity model (always works if turnover+footfall exist):
    - Aggregate per store over selected period
    - SPV = turnover/footfall
    - Benchmark SPV = median SPV of top quartile stores (by SPV)
    - Potential (period) = max(0, (benchmark_spv - store_spv) * store_footfall)
    """
    if df_period is None or df_period.empty:
        return pd.DataFrame()

    x = df_period.copy()
    for c in ("footfall", "turnover", "transactions", "sales_per_visitor", "conversion_rate"):
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    # require basics
    if "footfall" not in x.columns or "turnover" not in x.columns:
        return pd.DataFrame()

    agg = {"footfall": "sum", "turnover": "sum"}
    if "transactions" in x.columns:
        agg["transactions"] = "sum"

    g = x.groupby(store_key_col, as_index=False).agg(agg)

    g["spv"] = np.where(g["footfall"] > 0, g["turnover"] / g["footfall"], np.nan)

    if "transactions" in g.columns:
        g["conversion"] = np.where(g["footfall"] > 0, g["transactions"] / g["footfall"] * 100.0, np.nan)
        g["atv"] = np.where(g["transactions"] > 0, g["turnover"] / g["transactions"], np.nan)
    else:
        g["conversion"] = np.nan
        g["atv"] = np.nan

    valid = g.dropna(subset=["spv", "footfall"]).copy()
    valid = valid[valid["footfall"] >= 1]  # <-- relaxed so it won't go empty
    if valid.empty:
        return pd.DataFrame()

    q75 = valid["spv"].quantile(0.75)
    top = valid[valid["spv"] >= q75].copy()
    target_spv = float(top["spv"].median()) if not top.empty else float(valid["spv"].median())

    g["target_spv"] = target_spv
    g["spv_gap"] = g["target_spv"] - g["spv"]
    g["revenue_potential_period"] = np.where(g["spv_gap"] > 0, g["spv_gap"] * g["footfall"], 0.0)

    # driver label
    def _driver(row):
        if row["revenue_potential_period"] <= 0:
            return "On track"
        if not pd.isna(row.get("conversion", np.nan)) and not pd.isna(row.get("atv", np.nan)):
            # simple heuristic: which is further behind (relative)
            # benchmark conv/atv from top stores if possible
            return "SPV gap (conversion/ATV mix)"
        return "SPV uplift (benchmark gap)"

    g["opportunity_driver"] = g.apply(_driver, axis=1)
    return g

# ----------------------
# MAIN
# ----------------------
def main():
    header_left, header_right = st.columns([2.3, 1.7])

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
    clients_df["label"] = clients_df.apply(
        lambda r: f"{r['brand']} – {r['name']} (company_id {r['company_id']})",
        axis=1,
    )

    with header_right:
        c1, c2 = st.columns([1.2, 1.0])
        with c1:
            client_label = st.selectbox("Retailer", clients_df["label"].tolist(), label_visibility="collapsed")
        with c2:
            period_choice = st.selectbox("Periode", period_options(), index=0, label_visibility="collapsed")

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

    # Controls row
    available_regions = sorted(merged["region"].dropna().unique().tolist())
    top_controls = st.columns([1.0, 1.15, 1.05, 1.35])
    with top_controls[0]:
        region_choice = st.selectbox("Regio", available_regions)
    with top_controls[1]:
        compare_all_regions = st.toggle("Vergelijk met andere regio’s", value=True)
    with top_controls[2]:
        show_macro = st.toggle("Toon macro (CBS/CCI)", value=False)
    with top_controls[3]:
        run_btn = st.button("Analyseer", type="primary")

    if not run_btn:
        st.info("Selecteer retailer/regio/periode en klik op **Analyseer**.")
        return

    # Period selection
    start_period, end_period, macro_year = resolve_period_dates(period_choice)
    start_ts = pd.Timestamp(start_period)
    end_ts = pd.Timestamp(end_period)

    # shop ids
    region_shops = merged[merged["region"] == region_choice].copy()
    region_shop_ids = region_shops["id"].dropna().astype(int).unique().tolist()
    all_shop_ids = merged["id"].dropna().astype(int).unique().tolist()

    if not region_shop_ids:
        st.warning(f"Geen winkels gevonden voor regio '{region_choice}'.")
        return

    fetch_ids = all_shop_ids if compare_all_regions else region_shop_ids

    # Request keys
    request_keys = ["count_in", "turnover", "transactions", "sales_per_visitor", "conversion_rate"]
    metric_map = {
        "count_in": "footfall",
        "turnover": "turnover",
        "transactions": "transactions",
        "sales_per_visitor": "sales_per_visitor",
        "conversion_rate": "conversion_rate",
    }

    with st.spinner("Data ophalen via FastAPI..."):
        try:
            resp = get_report(
                fetch_ids,
                request_keys,
                period="date",
                step="day",
                source="shops",
                date_from=str(start_period),
                date_to=str(end_period),
            )
        except Exception:
            resp = get_report(
                fetch_ids,
                request_keys,
                period="all_time",
                step="day",
                source="shops",
            )

        df_raw = normalize_vemcount_response(resp, kpi_keys=request_keys).rename(columns=metric_map)

    if df_raw.empty:
        st.warning("Geen data ontvangen voor de gekozen selectie.")
        return

    # identify store id col
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

    # filter to selected period
    df_period_all = df_raw[(df_raw["date"] >= start_ts) & (df_raw["date"] <= end_ts)].copy()
    if df_period_all.empty:
        st.warning("Geen data in de geselecteerde periode.")
        return

    df_period_all = compute_daily_kpis(df_period_all)

    # SAFE MERGE store meta (avoid id_x/id_y issues)
    meta_cols = ["id", "store_display", "region", "sqm_effective"]
    if "store_type" in merged.columns:
        meta_cols.append("store_type")

    meta = merged[meta_cols].drop_duplicates().copy()

    if store_key_col == "id":
        df_period_all["id"] = pd.to_numeric(df_period_all["id"], errors="coerce").astype("Int64")
        df_period_all = df_period_all.merge(meta, on="id", how="left")
    else:
        df_period_all[store_key_col] = pd.to_numeric(df_period_all[store_key_col], errors="coerce").astype("Int64")
        df_period_all = df_period_all.merge(meta, left_on=store_key_col, right_on="id", how="left")

    df_region = df_period_all[df_period_all["region"] == region_choice].copy()
    if df_region.empty:
        st.warning("Geen data voor geselecteerde regio binnen de periode.")
        return

    # Weekly trend (region aggregated) + Pathzz (summed)
    region_weekly = aggregate_weekly_region(df_region)
    pathzz_weekly = fetch_region_street_traffic_weekly(region_shop_ids, region_choice, start_period, end_period)

    capture_weekly = pd.DataFrame()
    avg_capture = np.nan

    if not region_weekly.empty and not pathzz_weekly.empty:
        capture_weekly = pd.merge(region_weekly, pathzz_weekly, on="week_start", how="inner")

        # CRITICAL FIX: ensure 1 record per week_start (collapse duplicates from pathzz)
        capture_weekly = (
            capture_weekly
            .groupby("week_start", as_index=False)
            .agg({
                "footfall": "sum",
                "turnover": "sum",
                "street_footfall": "sum",
                "transactions": "sum" if "transactions" in capture_weekly.columns else "sum",
            })
        )

        capture_weekly["capture_rate"] = np.where(
            capture_weekly["street_footfall"] > 0,
            capture_weekly["footfall"] / capture_weekly["street_footfall"] * 100.0,
            np.nan,
        )
        avg_capture = float(capture_weekly["capture_rate"].mean()) if capture_weekly["capture_rate"].notna().any() else np.nan

    # KPI totals
    foot_total = float(df_region["footfall"].sum()) if "footfall" in df_region.columns else 0.0
    turn_total = float(df_region["turnover"].sum()) if "turnover" in df_region.columns else 0.0
    spv_avg = float(df_region["sales_per_visitor"].mean()) if "sales_per_visitor" in df_region.columns else np.nan

    # SVI
    svi_all = build_store_vitality(df_period=df_period_all, region_shops=merged, store_key_col=store_key_col)
    if not isinstance(svi_all, pd.DataFrame):
        svi_all = pd.DataFrame()

    # region scores
    region_scores = pd.DataFrame()
    region_svi = np.nan
    region_status, region_color = ("-", PFM_LINE)

    if not svi_all.empty and "region" in svi_all.columns and "svi_score" in svi_all.columns:
        region_scores = (
            svi_all.groupby("region", as_index=False)["svi_score"]
            .mean()
            .rename(columns={"svi_score": "region_svi"})
            .dropna(subset=["region"])
        )
        cur = region_scores[region_scores["region"] == region_choice]
        if not cur.empty:
            region_svi = float(np.clip(cur["region_svi"].iloc[0], 0, 100))
            region_status, region_color = status_from_score(region_svi)

    svi_region = pd.DataFrame()
    if not svi_all.empty and "region" in svi_all.columns:
        svi_region = svi_all[svi_all["region"] == region_choice].copy()

    # Opportunities (always)
    opp_base = compute_opportunities_store_level(df_period_all, store_key_col=store_key_col)
    if not opp_base.empty:
        # attach store names / region safely
        if store_key_col == "id":
            name_meta = df_period_all[["id", "store_display", "region"]].drop_duplicates()
            opp_base = opp_base.merge(name_meta, on="id", how="left")
            opp_base = opp_base.rename(columns={"store_display": "store_name"})
        else:
            name_meta = df_period_all[[store_key_col, "store_display", "region"]].drop_duplicates()
            opp_base = opp_base.merge(name_meta, on=store_key_col, how="left")
            opp_base = opp_base.rename(columns={"store_display": "store_name"})

    # ---------- Header ----------
    st.markdown(f"## {selected_client['brand']} — Regio **{region_choice}** · {start_period} → {end_period}")

    # ---------- KPI row ----------
    k1, k2, k3, k4 = st.columns([1, 1, 1, 1])
    with k1:
        kpi_card("Footfall", fmt_int(foot_total), "Regio · periode")
    with k2:
        kpi_card("Omzet", fmt_eur(turn_total), "Regio · periode")
    with k3:
        kpi_card("SPV", (fmt_eur_dec(spv_avg) if not pd.isna(spv_avg) else "-"), "Gemiddelde")
    with k4:
        kpi_card("Capture", (fmt_pct(avg_capture) if not pd.isna(avg_capture) else "-"), "Gemiddeld (Pathzz)")

    st.markdown("")

    # ---------- Weekly trend FULL WIDTH ----------
    st.markdown('<div class="panel"><div class="panel-title">Weekly trend — Store vs Street + Capture</div>', unsafe_allow_html=True)
    if capture_weekly.empty:
        st.info("Geen matchende Pathzz-weekdata gevonden voor deze regio/periode.")
    else:
        chart_df = capture_weekly[["week_start", "footfall", "street_footfall", "turnover", "capture_rate"]].copy()
        chart_df["week_start"] = pd.to_datetime(chart_df["week_start"])

        long = chart_df.melt(
            id_vars=["week_start"],
            value_vars=["footfall", "street_footfall", "turnover"],
            var_name="metric",
            value_name="value",
        )

        bar = (
            alt.Chart(long)
            .mark_bar(opacity=0.85, cornerRadiusEnd=4)
            .encode(
                x=alt.X("week_start:T", title=None, axis=alt.Axis(labelAngle=-35)),
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
                    alt.Tooltip("week_start:T", title="Week start"),
                    alt.Tooltip("metric:N", title="Type"),
                    alt.Tooltip("value:Q", title="Waarde", format=",.0f"),
                ],
            )
        )

        line = (
            alt.Chart(chart_df)
            .mark_line(point=True, strokeWidth=2, color=PFM_DARK)
            .encode(
                x=alt.X("week_start:T", title=None),
                y=alt.Y("capture_rate:Q", title="Capture %"),
                tooltip=[
                    alt.Tooltip("week_start:T", title="Week start"),
                    alt.Tooltip("capture_rate:Q", title="Capture", format=".1f"),
                ],
            )
        )

        st.altair_chart(
            alt.layer(bar, line).resolve_scale(y="independent").properties(height=310),
            use_container_width=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    # ---------- Row: Region compare + SVI donut HORIZONTAL ----------
    c_left, c_right = st.columns([1.45, 0.85])

    with c_left:
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

    with c_right:
        st.markdown('<div class="panel"><div class="panel-title">Regio Vitality</div>', unsafe_allow_html=True)
        if not pd.isna(region_svi):
            st.altair_chart(gauge_chart(region_svi, region_color), use_container_width=True)
            st.markdown(f"**{region_svi:.0f}** · {region_status}")
        else:
            st.info("Nog geen regio-score.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    # ---------- Row: Store ranking + Opportunities ----------
    r3_left, r3_right = st.columns([1.45, 1.05])

    with r3_left:
        st.markdown('<div class="panel"><div class="panel-title">Store Vitality ranking — geselecteerde regio</div>', unsafe_allow_html=True)

        if svi_region.empty:
            st.info("Geen stores in deze regio met SVI (of regio-koppeling ontbreekt).")
        else:
            keep = svi_region.copy()
            keep["svi_score"] = pd.to_numeric(keep["svi_score"], errors="coerce")
            keep = keep.dropna(subset=["svi_score"])

            if keep.empty:
                st.info("Geen valide SVI-scores gevonden.")
            else:
                chart_rank = (
                    alt.Chart(keep.sort_values("svi_score", ascending=False).head(12))
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
                    .properties(height=310)
                )
                st.altair_chart(chart_rank, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with r3_right:
        st.markdown('<div class="panel"><div class="panel-title">Biggest opportunities — uitgelegd</div>', unsafe_allow_html=True)

        if opp_base.empty:
            st.info("Nog geen opportunity data (check: turnover+footfall aanwezig? Zie Debug).")
        else:
            opp_r = opp_base[opp_base["region"] == region_choice].copy()
            opp_r["revenue_potential_period"] = pd.to_numeric(opp_r["revenue_potential_period"], errors="coerce").fillna(0.0)

            if opp_r.empty:
                st.info("Geen opportunities gevonden binnen deze regio.")
            else:
                period_days = (end_ts - start_ts).days + 1
                year_factor = 365.0 / period_days if period_days > 0 else 1.0
                opp_r["revenue_potential_year"] = opp_r["revenue_potential_period"] * year_factor

                top = opp_r.sort_values("revenue_potential_year", ascending=False).head(6).copy()

                opp_chart = (
                    alt.Chart(top)
                    .mark_bar(cornerRadiusEnd=4, color=PFM_RED)
                    .encode(
                        x=alt.X("revenue_potential_year:Q", title="€ omzetpotentie / jaar", axis=alt.Axis(format=",.0f")),
                        y=alt.Y("store_name:N", sort="-x", title=None),
                        tooltip=[
                            alt.Tooltip("store_name:N", title="Winkel"),
                            alt.Tooltip("revenue_potential_year:Q", title="€ / jaar", format=",.0f"),
                            alt.Tooltip("opportunity_driver:N", title="Driver"),
                            alt.Tooltip("spv:Q", title="SPV", format=".2f"),
                            alt.Tooltip("target_spv:Q", title="Target SPV", format=".2f"),
                        ],
                    )
                    .properties(height=310)
                )
                st.altair_chart(opp_chart, use_container_width=True)

                total_top5 = float(top["revenue_potential_year"].head(5).sum())
                st.markdown(f"**Top 5 samen:** {fmt_eur(total_top5)} / jaar")

                target_spv = float(top["target_spv"].dropna().iloc[0]) if top["target_spv"].notna().any() else np.nan
                st.caption(
                    "Berekening: **SPV = omzet/footfall** per store over de gekozen periode. "
                    "Benchmark = **median SPV van de top 25% stores**. "
                    "Potentie = max(0, (Benchmark SPV − Store SPV) × Footfall), omgerekend naar jaar."
                )
                if not pd.isna(target_spv):
                    st.caption(f"Benchmark SPV: **€ {target_spv:.2f}**".replace(".", ","))

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Macro section: ALWAYS full year + highlight selected period ----------
    if show_macro:
        st.markdown("## Macro-context (optioneel)")
        st.caption("We tonen altijd het hele geselecteerde jaar. De gekozen periode (Q/jaar) wordt gemarkeerd.")

        macro_col1, macro_col2 = st.columns(2)

        year_start = date(macro_year, 1, 1)
        year_end = date(macro_year, 12, 31)

        # IMPORTANT: extra fetch for full year to ensure chart is complete
        with st.spinner("Macro: jaar-data ophalen..."):
            try:
                resp_year = get_report(
                    fetch_ids,
                    request_keys,
                    period="date",
                    step="day",
                    source="shops",
                    date_from=str(year_start),
                    date_to=str(year_end),
                )
                df_year_raw = normalize_vemcount_response(resp_year, kpi_keys=request_keys).rename(columns=metric_map)
            except Exception:
                df_year_raw = pd.DataFrame()

        if not df_year_raw.empty:
            df_year_raw["date"] = pd.to_datetime(df_year_raw["date"], errors="coerce")
            df_year_raw = df_year_raw.dropna(subset=["date"])
            df_year_raw = compute_daily_kpis(df_year_raw)

            # attach region mapping
            if store_key_col == "id":
                df_year_raw["id"] = pd.to_numeric(df_year_raw["id"], errors="coerce").astype("Int64")
                df_year = df_year_raw.merge(meta[["id", "region"]].drop_duplicates(), on="id", how="left")
            else:
                df_year_raw[store_key_col] = pd.to_numeric(df_year_raw[store_key_col], errors="coerce").astype("Int64")
                df_year = df_year_raw.merge(meta[["id", "region"]].drop_duplicates(), left_on=store_key_col, right_on="id", how="left")

            df_year = df_year[df_year["region"] == region_choice].copy()
        else:
            df_year = pd.DataFrame()

        band_df = pd.DataFrame({
            "start": [pd.Timestamp(start_period)],
            "end": [pd.Timestamp(end_period)]
        })

        # monthly region index full year
        region_month = pd.DataFrame()
        if not df_year.empty:
            df_year["month"] = df_year["date"].dt.to_period("M").dt.to_timestamp()
            region_month = (
                df_year.groupby("month", as_index=False)[["turnover", "footfall"]]
                .sum()
                .rename(columns={"turnover": "region_turnover", "footfall": "region_footfall"})
            )

        def index_from_first_nonzero(s: pd.Series) -> pd.Series:
            s = pd.to_numeric(s, errors="coerce").astype(float)
            nonzero = s.replace(0, np.nan).dropna()
            if nonzero.empty:
                return pd.Series(np.nan, index=s.index)
            base = nonzero.iloc[0]
            return s / base * 100.0 if base != 0 else pd.Series(np.nan, index=s.index)

        if not region_month.empty:
            region_month["region_turnover_index"] = index_from_first_nonzero(region_month["region_turnover"])
            region_month["region_footfall_index"] = index_from_first_nonzero(region_month["region_footfall"])

        with macro_col1:
            st.markdown('<div class="panel"><div class="panel-title">CBS detailhandelindex vs Regio</div>', unsafe_allow_html=True)
            try:
                retail_series = get_retail_index(months_back=36)
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
                cbs_retail_df = cbs_retail_df[(cbs_retail_df["date"].dt.year == macro_year)]
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
                base_chart = alt.Chart(pd.concat(lines, ignore_index=True))
                band = (
                    alt.Chart(band_df)
                    .mark_rect(opacity=0.10, color=PFM_PURPLE)
                    .encode(x="start:T", x2="end:T")
                )
                macro = (
                    band +
                    base_chart.mark_line(point=True).encode(
                        x=alt.X("date:T", title="Maand"),
                        y=alt.Y("value:Q", title="Index (100 = eerste maand)"),
                        color=alt.Color("series:N", title=""),
                        tooltip=[
                            alt.Tooltip("date:T", title="Maand"),
                            alt.Tooltip("series:N", title="Reeks"),
                            alt.Tooltip("value:Q", title="Index", format=".1f"),
                        ],
                    )
                ).properties(height=260)
                st.altair_chart(macro, use_container_width=True)
            else:
                st.info("Geen macro-lijnen beschikbaar.")

            st.markdown("</div>", unsafe_allow_html=True)

        with macro_col2:
            st.markdown('<div class="panel"><div class="panel-title">Consumentenvertrouwen (CCI) vs Regio</div>', unsafe_allow_html=True)
            try:
                cci_series = get_cci_series(months_back=36)
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
                cci_df = cci_df[(cci_df["date"].dt.year == macro_year)]
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
                base_chart = alt.Chart(pd.concat(lines, ignore_index=True))
                band = (
                    alt.Chart(band_df)
                    .mark_rect(opacity=0.10, color=PFM_PURPLE)
                    .encode(x="start:T", x2="end:T")
                )
                cc = (
                    band +
                    base_chart.mark_line(point=True).encode(
                        x=alt.X("date:T", title="Maand"),
                        y=alt.Y("value:Q", title="Index (100 = eerste maand)"),
                        color=alt.Color("series:N", title=""),
                        tooltip=[
                            alt.Tooltip("date:T", title="Maand"),
                            alt.Tooltip("series:N", title="Reeks"),
                            alt.Tooltip("value:Q", title="Index", format=".1f"),
                        ],
                    )
                ).properties(height=260)
                st.altair_chart(cc, use_container_width=True)
            else:
                st.info("Geen CCI-data beschikbaar.")

            st.markdown("</div>", unsafe_allow_html=True)

    # Debug
    with st.expander("🔧 Debug"):
        st.write("Retailer:", selected_client)
        st.write("Regio:", region_choice)
        st.write("Periode:", start_period, "→", end_period)
        st.write("Macro year:", macro_year)
        st.write("Compare all regions:", compare_all_regions)
        st.write("Store key col:", store_key_col)
        st.write("All shops:", len(all_shop_ids), "Region shops:", len(region_shop_ids))
        st.write("df_period_all cols:", df_period_all.columns.tolist())
        st.write("df_region head:", df_region.head())
        st.write("capture_weekly head:", capture_weekly.head())
        st.write("opp_base head:", opp_base.head())
        st.write("opp_base nonzero potential:", (opp_base["revenue_potential_period"] > 0).sum() if not opp_base.empty else 0)


if __name__ == "__main__":
    main()
