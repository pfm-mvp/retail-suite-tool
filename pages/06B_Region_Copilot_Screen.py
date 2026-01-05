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
# ----------------------
st.markdown(
    f"""
    <style>
      .block-container {{
        padding-top: 2.25rem;      /* FIX: title chopped off */
        padding-bottom: 2rem;
      }}
      .pfm-header {{
        display:flex;
        align-items:center;
        justify-content:space-between;
        padding: 0.75rem 1rem;
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

def fmt_eur_2(x: float) -> str:
    if pd.isna(x):
        return "-"
    s = f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"€ {s}"

# ----------------------
# Period helpers
# ----------------------
def _quarter_range(year: int, q: int) -> tuple[date, date]:
    if q == 1:
        return date(year, 1, 1), date(year, 3, 31)
    if q == 2:
        return date(year, 4, 1), date(year, 6, 30)
    if q == 3:
        return date(year, 7, 1), date(year, 9, 30)
    return date(year, 10, 1), date(year, 12, 31)

def period_catalog(today: date) -> dict[str, dict]:
    # Macro year is used for CBS charts: we always show full year of the selected year
    return {
        "Kalenderjaar 2024": {"start": date(2024, 1, 1), "end": date(2024, 12, 31), "macro_year": 2024},
        "Kalenderjaar 2025": {"start": date(2025, 1, 1), "end": date(2025, 12, 31), "macro_year": 2025},

        "Q1 2024": {"start": _quarter_range(2024, 1)[0], "end": _quarter_range(2024, 1)[1], "macro_year": 2024},
        "Q2 2024": {"start": _quarter_range(2024, 2)[0], "end": _quarter_range(2024, 2)[1], "macro_year": 2024},
        "Q3 2024": {"start": _quarter_range(2024, 3)[0], "end": _quarter_range(2024, 3)[1], "macro_year": 2024},
        "Q4 2024": {"start": _quarter_range(2024, 4)[0], "end": _quarter_range(2024, 4)[1], "macro_year": 2024},

        "Q1 2025": {"start": _quarter_range(2025, 1)[0], "end": _quarter_range(2025, 1)[1], "macro_year": 2025},
        "Q2 2025": {"start": _quarter_range(2025, 2)[0], "end": _quarter_range(2025, 2)[1], "macro_year": 2025},
        "Q3 2025": {"start": _quarter_range(2025, 3)[0], "end": _quarter_range(2025, 3)[1], "macro_year": 2025},
        "Q4 2025": {"start": _quarter_range(2025, 4)[0], "end": _quarter_range(2025, 4)[1], "macro_year": 2025},

        "Laatste 26 weken": {"start": today - timedelta(weeks=26), "end": today, "macro_year": today.year},
    }

# ----------------------
# Region & API helpers
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

    # Optional columns - do NOT break existing scripts
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
def get_report(
    shop_ids,
    data_outputs,
    period: str,
    step: str = "day",
    source: str = "shops",
    date_from: date | None = None,
    date_to: date | None = None,
):
    """
    IMPORTANT:
    - For historical periods (2024/2025), use period="date" + date_from/date_to
    - Query param style must be: data=..&data=.. (no [])
    """
    params: list[tuple[str, str]] = []
    for sid in shop_ids:
        params.append(("data", str(sid)))
    for dout in data_outputs:
        params.append(("data_output", str(dout)))
    params.append(("period", period))
    params.append(("step", step))
    params.append(("source", source))

    if period == "date":
        if date_from is None or date_to is None:
            raise ValueError("period='date' requires date_from and date_to")
        params.append(("date_from", str(date_from)))
        params.append(("date_to", str(date_to)))

    resp = requests.post(REPORT_URL, params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()

@st.cache_data(ttl=600)
def fetch_region_street_traffic(region: str, start_date, end_date) -> pd.DataFrame:
    """
    Expects data/pathzz_sample_weekly.csv with at least:
    Region;Week;Visits
    (extra columns are allowed; we only read first 3 columns)
    """
    csv_path = "data/pathzz_sample_weekly.csv"
    try:
        df = pd.read_csv(
            csv_path,
            sep=";",
            header=0,
            usecols=[0, 1, 2],
            dtype=str,
            engine="python",
        )
    except Exception:
        return pd.DataFrame()

    df.columns = ["region", "week", "street_footfall"]
    df["region"] = df["region"].astype(str).str.strip()
    region_norm = str(region).strip().lower()
    df = df[df["region"].str.lower() == region_norm].copy()
    if df.empty:
        return pd.DataFrame()

    df["street_footfall"] = df["street_footfall"].astype(str).str.strip().replace("", np.nan)
    df = df.dropna(subset=["street_footfall"])

    # EU format: 45.654
    df["street_footfall"] = (
        df["street_footfall"]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    def _parse_week_start(s: str):
        if isinstance(s, str) and "To" in s:
            return pd.to_datetime(s.split("To")[0].strip(), errors="coerce")
        return pd.NaT

    df["week_start"] = df["week"].apply(_parse_week_start)
    df = df.dropna(subset=["week_start"])

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    df = df[(df["week_start"] >= start) & (df["week_start"] <= end)]

    return df[["week_start", "street_footfall"]].reset_index(drop=True)

# ----------------------
# Robust helpers
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
    gauge_df = pd.DataFrame(
        {"segment": ["filled", "empty"], "value": [score_0_100, max(0.0, 100.0 - score_0_100)]}
    )
    arc = (
        alt.Chart(gauge_df)
        .mark_arc(innerRadius=54, outerRadius=70)
        .encode(
            theta="value:Q",
            color=alt.Color(
                "segment:N",
                scale=alt.Scale(domain=["filled", "empty"], range=[fill_color, PFM_LINE]),
                legend=None,
            ),
        )
        .properties(width=180, height=180)
    )
    text = (
        alt.Chart(pd.DataFrame({"label": [f"{score_0_100:.0f}"]}))
        .mark_text(size=28, fontWeight="bold", color=PFM_DARK)
        .encode(text="label:N")
        .properties(width=180, height=180)
    )
    return arc + text

# ----------------------
# KPI computations + de-duplication
# ----------------------
def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def collapse_to_daily_store(df: pd.DataFrame, store_key_col: str) -> pd.DataFrame:
    """
    FIX #1:
    normalize_vemcount_response can yield multiple rows per (shop_id, date) when metrics are returned in a "stacked" way.
    We collapse to exactly 1 row per store/day.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])

    # candidate numeric columns we might have
    numeric_cols = [
        "footfall", "turnover", "transactions",
        "sales_per_visitor", "conversion_rate",
        "avg_basket_size", "sales_per_sqm", "sales_per_transaction",
    ]
    out = _coerce_numeric(out, [c for c in numeric_cols if c in out.columns])

    group_cols = [store_key_col, "date"]

    agg = {}
    for c in numeric_cols:
        if c not in out.columns:
            continue
        # sums for additive metrics
        if c in ("footfall", "turnover", "transactions"):
            agg[c] = "sum"
        else:
            # ratios/derived: take mean of available (we'll recompute core ones below)
            agg[c] = "mean"

    out = out.groupby(group_cols, as_index=False).agg(agg)

    # recompute robust derived metrics if base present
    if "turnover" in out.columns and "footfall" in out.columns:
        out["sales_per_visitor"] = np.where(out["footfall"] > 0, out["turnover"] / out["footfall"], np.nan)
    if "transactions" in out.columns and "footfall" in out.columns:
        out["conversion_rate"] = np.where(out["footfall"] > 0, out["transactions"] / out["footfall"] * 100.0, np.nan)
    if "turnover" in out.columns and "transactions" in out.columns:
        out["avg_basket_size"] = np.where(out["transactions"] > 0, out["turnover"] / out["transactions"], np.nan)
    if "avg_basket_size" in out.columns and "sales_per_visitor" in out.columns and "avg_basket_size"].notna().any():
        # sales_per_transaction already defined above (same as avg_basket_size), keep if provided
        pass

    return out

def aggregate_weekly_region(df_region_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Region weekly totals (one line per week).
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
    if "transactions" in df.columns:
        agg["transactions"] = "sum"

    out = df.groupby("week_start", as_index=False).agg(agg)

    # robust weekly SPV / conversion
    if "turnover" in out.columns and "footfall" in out.columns:
        out["sales_per_visitor"] = np.where(out["footfall"] > 0, out["turnover"] / out["footfall"], np.nan)
    if "transactions" in out.columns and "footfall" in out.columns:
        out["conversion_rate"] = np.where(out["footfall"] > 0, out["transactions"] / out["footfall"] * 100.0, np.nan)

    return out

# ----------------------
# Opportunity logic (explainable)
# ----------------------
def build_opportunities(
    df_store_period: pd.DataFrame,
    svi_region: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """
    We prefer profit_potential_period from build_store_vitality.
    If that's missing/zero, we fall back to:
      uplift = max(0, benchmark_spv - store_spv) * store_footfall_period
    Benchmark SPV = 75th percentile of region stores (or overall if region too small).
    """
    period_days = (end_ts - start_ts).days + 1
    year_factor = 365.0 / period_days if period_days > 0 else 1.0

    # Store-level totals in the selected period
    # df_store_period must be daily+store collapsed and already joined with store_display/region
    if df_store_period is None or df_store_period.empty:
        return pd.DataFrame()

    cols_needed = ["id", "store_display", "footfall", "turnover", "sales_per_visitor", "region"]
    for c in cols_needed:
        if c not in df_store_period.columns:
            return pd.DataFrame()

    store_agg = (
        df_store_period
        .groupby(["id", "store_display", "region"], as_index=False)
        .agg(
            footfall=("footfall", "sum"),
            turnover=("turnover", "sum"),
            spv=("sales_per_visitor", "mean"),
        )
    )

    store_agg["spv"] = pd.to_numeric(store_agg["spv"], errors="coerce")
    store_agg["footfall"] = pd.to_numeric(store_agg["footfall"], errors="coerce")
    store_agg["turnover"] = pd.to_numeric(store_agg["turnover"], errors="coerce")

    # Merge in SVI fields when available
    if svi_region is not None and not svi_region.empty:
        use_cols = ["id", "svi_score", "svi_status", "reason_short", "profit_potential_period", "sales_per_visitor"]
        use_cols = [c for c in use_cols if c in svi_region.columns]
        svi_min = svi_region[use_cols].drop_duplicates("id")
        store_agg = store_agg.merge(svi_min, on="id", how="left")

    # Primary opportunity source
    store_agg["profit_potential_period"] = pd.to_numeric(store_agg.get("profit_potential_period", np.nan), errors="coerce")
    store_agg["profit_potential_year"] = store_agg["profit_potential_period"] * year_factor

    # If profit_potential_year is all missing/0 → fallback
    all_zeroish = store_agg["profit_potential_year"].fillna(0).abs().sum() < 1e-6

    if all_zeroish:
        # Benchmark SPV = 75th percentile within region
        spv_vals = store_agg["spv"].dropna()
        if spv_vals.empty:
            return pd.DataFrame()

        benchmark_spv = float(spv_vals.quantile(0.75))
        store_agg["benchmark_spv"] = benchmark_spv

        store_agg["uplift_spv"] = np.maximum(0.0, store_agg["benchmark_spv"] - store_agg["spv"])
        store_agg["profit_potential_period"] = store_agg["uplift_spv"] * store_agg["footfall"]
        store_agg["profit_potential_year"] = store_agg["profit_potential_period"] * year_factor

        # driver label
        store_agg["reason_short"] = store_agg.get("reason_short", np.nan)
        store_agg["opportunity_driver"] = np.where(
            store_agg["uplift_spv"] > 0,
            "SPV uplift to top quartile",
            "—"
        )
    else:
        store_agg["opportunity_driver"] = np.where(
            store_agg["profit_potential_year"].fillna(0) > 0,
            "SVI profit potential",
            "—"
        )

    # Clean
    store_agg = store_agg.replace([np.inf, -np.inf], np.nan)

    return store_agg

# ----------------------
# MAIN
# ----------------------
def main():
    # small spacer to avoid header cropping in embedded views
    st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)

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
    clients_df["label"] = clients_df.apply(
        lambda r: f"{r['brand']} – {r['name']} (company_id {r['company_id']})",
        axis=1,
    )

    today = datetime.today().date()
    periods = period_catalog(today)
    period_labels = list(periods.keys())

    with header_right:
        c1, c2 = st.columns(2)
        with c1:
            client_label = st.selectbox("Retailer", clients_df["label"].tolist(), label_visibility="collapsed")
        with c2:
            period_choice = st.selectbox(
                "Periode",
                period_labels,
                index=period_labels.index("Q3 2024") if "Q3 2024" in period_labels else 0,
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
    # ----------------------
# DEBUG: sqm availability
# ----------------------
with st.expander("🧱 Debug: SQM (location surface)"):
    st.write("Columns in locations_df:", list(locations_df.columns))
    st.write("Columns in merged:", list(merged.columns))

    # Detect possible sqm columns
    sqm_candidates = ["sqm", "sq_meter", "sq_meters", "square_meters", "area", "surface", "m2", "size"]
    found_in_locations = [c for c in sqm_candidates if c in locations_df.columns]
    found_in_merged = [c for c in sqm_candidates if c in merged.columns]

    st.write("SQM candidates found in locations_df:", found_in_locations if found_in_locations else "None")
    st.write("SQM candidates found in merged:", found_in_merged if found_in_merged else "None")

    # Pick the column that will be used (same logic as script)
    sqm_col = None
    for cand in ["sqm", "sq_meter", "sq_meters", "square_meters"]:
        if cand in merged.columns:
            sqm_col = cand
            break

    st.write("sqm_col selected by script:", sqm_col if sqm_col else "None")

    # Show counts and sample rows
    show_cols = ["id", "name", "store_display", "region", "shop_id", "sqm_override"]
    show_cols = [c for c in show_cols if c in merged.columns]

    if sqm_col is not None:
        merged["_sqm_raw"] = pd.to_numeric(merged[sqm_col], errors="coerce")
        st.write("Non-null sqm raw (from locations):", int(merged["_sqm_raw"].notna().sum()), "/", int(len(merged)))
        show_cols = show_cols + [sqm_col, "_sqm_raw"]

    if "sqm_effective" in merged.columns:
        merged["_sqm_effective_num"] = pd.to_numeric(merged["sqm_effective"], errors="coerce")
        st.write("Non-null sqm_effective:", int(merged["_sqm_effective_num"].notna().sum()), "/", int(len(merged)))
        show_cols = show_cols + ["sqm_effective", "_sqm_effective_num"]

    # Unique store count + missing list
    if "id" in merged.columns:
        missing = merged[merged.get("sqm_effective", np.nan).isna()][["id"]].drop_duplicates()
        st.write("Stores missing sqm_effective (first 30 ids):", missing["id"].astype(str).head(30).tolist())

    st.dataframe(
        merged[show_cols].drop_duplicates(subset=["id"]).head(20),
        use_container_width=True
    )
    if merged.empty:
        st.warning("Er zijn geen winkels met een regio-mapping voor deze retailer.")
        return

    # sqm_effective + store_display
    # NOTE: if sqm is not provided by locations endpoint, sqm_effective will remain NaN unless overridden
    sqm_col = None
    for cand in ["sqm", "sq_meter", "sq_meters", "square_meters"]:
        if cand in merged.columns:
            sqm_col = cand
            break

    if sqm_col is not None:
        merged["sqm_effective"] = np.where(
            merged["sqm_override"].notna(),
            merged["sqm_override"],
            pd.to_numeric(merged[sqm_col], errors="coerce"),
        )
    else:
        merged["sqm_effective"] = merged["sqm_override"]

    if "store_label" in merged.columns and merged["store_label"].notna().any():
        merged["store_display"] = merged["store_label"]
    else:
        merged["store_display"] = merged["name"] if "name" in merged.columns else merged["id"].astype(str)

    # Region selector + analyse button (single row controls)
    available_regions = sorted(merged["region"].dropna().unique().tolist())
    top_controls = st.columns([1.3, 1.3, 1.2, 1.4])
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

    # Date window
    start_period = periods[period_choice]["start"]
    end_period = periods[period_choice]["end"]
    macro_year = periods[period_choice]["macro_year"]

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

    # Metrics to fetch (requested by you)
    # If the API doesn't return some, we still compute key KPIs from base metrics.
    metric_map = {
        "count_in": "footfall",
        "turnover": "turnover",
        "transactions": "transactions",
        "sales_per_visitor": "sales_per_visitor",
        "conversion_rate": "conversion_rate",
        "avg_basket_size": "avg_basket_size",
        "sales_per_sqm": "sales_per_sqm",
        "sales_per_transaction": "sales_per_transaction",
    }

    # Fetch report using period="date" so 2024/2025 work reliably
    with st.spinner("Data ophalen via FastAPI..."):
        resp = get_report(
            fetch_ids,
            list(metric_map.keys()),
            period="date",
            step="day",
            source="shops",
            date_from=start_period,
            date_to=end_period,
        )

        # normalize + rename
        df_norm = normalize_vemcount_response(resp, kpi_keys=metric_map.keys()).rename(columns=metric_map)

    if df_norm.empty:
        st.warning("Geen data ontvangen voor de gekozen selectie.")
        return

    # Identify store id column in normalized df
    store_key_col = None
    for cand in ["shop_id", "id", "location_id"]:
        if cand in df_norm.columns:
            store_key_col = cand
            break
    if store_key_col is None:
        st.error("Geen store-id kolom gevonden in de response (shop_id/id/location_id).")
        return

    # FIX #1: collapse to one row per store/day (stops duplicate capture points)
    df_daily_store = collapse_to_daily_store(df_norm, store_key_col=store_key_col)

    if df_daily_store.empty:
        st.warning("Geen data na opschonen (daily/store collapse).")
        return

    # Join store metadata
    join_cols = ["id", "store_display", "region", "sqm_effective"]
    if "store_type" in merged.columns:
        join_cols.append("store_type")

    df_daily_store = df_daily_store.merge(
        merged[join_cols].drop_duplicates(),
        left_on=store_key_col,
        right_on="id",
        how="left",
    )

    # Region-only view
    df_region_daily = df_daily_store[df_daily_store["region"] == region_choice].copy()
    if df_region_daily.empty:
        st.warning("Geen data voor geselecteerde regio binnen de periode.")
        return

    # KPI totals (period)
    foot_total = float(df_region_daily["footfall"].sum()) if "footfall" in df_region_daily.columns else 0.0
    turn_total = float(df_region_daily["turnover"].sum()) if "turnover" in df_region_daily.columns else 0.0

    spv_avg = np.nan
    if "turnover" in df_region_daily.columns and "footfall" in df_region_daily.columns:
        spv_avg = (df_region_daily["turnover"].sum() / df_region_daily["footfall"].sum()) if df_region_daily["footfall"].sum() > 0 else np.nan

    # Weekly trend + capture
    region_weekly = aggregate_weekly_region(df_region_daily)
    pathzz_weekly = fetch_region_street_traffic(region=region_choice, start_date=start_period, end_date=end_period)

    capture_weekly = pd.DataFrame()
    avg_capture = np.nan
    if not region_weekly.empty and not pathzz_weekly.empty:
        region_weekly["week_start"] = pd.to_datetime(region_weekly["week_start"])
        pathzz_weekly["week_start"] = pd.to_datetime(pathzz_weekly["week_start"])
        capture_weekly = pd.merge(region_weekly, pathzz_weekly, on="week_start", how="inner")

        # FIX #2: capture rate is REGION TOTAL per week, not per store
        if not capture_weekly.empty:
            capture_weekly["capture_rate"] = np.where(
                capture_weekly["street_footfall"] > 0,
                capture_weekly["footfall"] / capture_weekly["street_footfall"] * 100.0,
                np.nan,
            )
            avg_capture = float(capture_weekly["capture_rate"].mean())

    # SVI calculations (based on df_daily_store incl all regions in selected period)
    svi_all = build_store_vitality(
        df_period=df_daily_store,
        region_shops=merged,
        store_key_col=store_key_col,
    )
    svi_all = ensure_region_column(svi_all, merged, store_key_col) if isinstance(svi_all, pd.DataFrame) else pd.DataFrame()

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

    # Opportunities (works even when profit_potential_period is empty/0)
    opp_base = build_opportunities(
        df_store_period=df_region_daily,
        svi_region=svi_region,
        start_ts=start_ts,
        end_ts=end_ts,
    )

    # Header
    st.markdown(
        f"## {selected_client['brand']} — Regio **{region_choice}** · {start_period} → {end_period}"
    )

    # KPI row (4 cards only; SVI donut moved away to avoid white space)
    k1, k2, k3, k4 = st.columns([1, 1, 1, 1])
    with k1:
        kpi_card("Footfall", fmt_int(foot_total), "Regio · periode")
    with k2:
        kpi_card("Omzet", fmt_eur(turn_total), "Regio · periode")
    with k3:
        kpi_card("SPV", (fmt_eur_2(spv_avg) if not pd.isna(spv_avg) else "-"), "Omzet / bezoeker (gewogen)")
    with k4:
        kpi_card("Capture", (fmt_pct(avg_capture) if not pd.isna(avg_capture) else "-"), "Regio totaal (Pathzz)")

    # Row 2: Weekly trend + region compare + SVI donut (one line)
    r2_a, r2_b, r2_c = st.columns([1.7, 1.05, 0.75])

    with r2_a:
        st.markdown('<div class="panel"><div class="panel-title">Weekly trend — Store vs Street + Capture</div>', unsafe_allow_html=True)

        if capture_weekly.empty:
            st.info("Geen matchende Pathzz-weekdata gevonden voor deze regio/periode.")
        else:
            chart_df = capture_weekly[["week_start", "footfall", "street_footfall", "turnover", "capture_rate"]].copy()
            chart_df = chart_df.sort_values("week_start")

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
                alt.layer(bar, line).resolve_scale(y="independent").properties(height=260),
                use_container_width=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with r2_b:
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

    with r2_c:
        st.markdown('<div class="panel"><div class="panel-title">Regio Vitality</div>', unsafe_allow_html=True)
        if not pd.isna(region_svi):
            st.altair_chart(gauge_chart(region_svi, region_color), use_container_width=True)
            st.markdown(f"**{region_svi:.0f}** · {region_status}")
        else:
            st.info("Nog geen regio-score.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 3: Store ranking + opportunities
    r3_left, r3_right = st.columns([1.45, 1.15])

    with r3_left:
        st.markdown('<div class="panel"><div class="panel-title">Store Vitality ranking — geselecteerde regio</div>', unsafe_allow_html=True)

        if svi_region.empty:
            st.info("Geen stores in deze regio met SVI (of regio-koppeling ontbreekt).")
        else:
            svi_region = svi_region.copy()
            if "svi_score" in svi_region.columns:
                svi_region["svi_score"] = pd.to_numeric(svi_region["svi_score"], errors="coerce")
                svi_region = svi_region.dropna(subset=["svi_score"])

            if svi_region.empty or "svi_score" not in svi_region.columns:
                st.info("Geen valide SVI-scores gevonden.")
            else:
                # pick best label column
                y_col = "store_name" if "store_name" in svi_region.columns else ("store_display" if "store_display" in svi_region.columns else "id")

                chart_rank = (
                    alt.Chart(svi_region.sort_values("svi_score", ascending=False).head(12))
                    .mark_bar(cornerRadiusEnd=4)
                    .encode(
                        x=alt.X("svi_score:Q", title="SVI (0–100)", scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y(f"{y_col}:N", sort="-x", title=None),
                        color=alt.Color(
                            "svi_status:N",
                            scale=alt.Scale(
                                domain=["High performance", "Good / stable", "Attention required", "Under pressure"],
                                range=[PFM_GREEN, PFM_PURPLE, PFM_AMBER, PFM_RED],
                            ),
                            legend=alt.Legend(title=""),
                        ),
                        tooltip=[
                            alt.Tooltip(f"{y_col}:N", title="Winkel"),
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

        if opp_base is None or opp_base.empty:
            st.info("Geen opportunity data beschikbaar.")
        else:
            opp = opp_base.copy()
            opp["profit_potential_year"] = pd.to_numeric(opp["profit_potential_year"], errors="coerce")
            opp = opp.dropna(subset=["profit_potential_year"])
            opp = opp[opp["profit_potential_year"] > 0].copy()

            if opp.empty:
                st.info("Geen positieve opportunity gevonden (mogelijk te weinig data of benchmark = store).")
            else:
                # Top N
                topn = opp.sort_values("profit_potential_year", ascending=False).head(6).copy()

                opp_chart = (
                    alt.Chart(topn)
                    .mark_bar(cornerRadiusEnd=4, color=PFM_RED)
                    .encode(
                        x=alt.X("profit_potential_year:Q", title="€ / jaar", axis=alt.Axis(format=",.0f")),
                        y=alt.Y("store_display:N", sort="-x", title=None),
                        tooltip=[
                            alt.Tooltip("store_display:N", title="Winkel"),
                            alt.Tooltip("profit_potential_year:Q", title="Potentie € / jaar", format=",.0f"),
                            alt.Tooltip("opportunity_driver:N", title="Driver"),
                            alt.Tooltip("reason_short:N", title="Toelichting"),
                        ],
                    )
                    .properties(height=300)
                )
                st.altair_chart(opp_chart, use_container_width=True)

                total_top5 = float(topn["profit_potential_year"].head(5).sum())
                st.markdown(f"**Top 5 samen:** {fmt_eur(total_top5)} / jaar")

                # Explainability box
                st.caption("Hoe dit wordt berekend:")
                if (topn.get("opportunity_driver") == "SVI profit potential").any():
                    st.caption("• Primair: `profit_potential_period` uit Store Vitality (SVI) → geannualiseerd naar € / jaar.")
                else:
                    st.caption("• Fallback: uplift naar benchmark **SPV (top quartile)** × **footfall** in de periode → geannualiseerd.")

        st.markdown("</div>", unsafe_allow_html=True)

    # Macro section (toggle)
    if show_macro:
        st.markdown("## Macro-context (optioneel)")
        st.caption("Macro laat altijd het héle jaar zien van de geselecteerde periode; je selectie (Q/maanden) highlighten we.")

        # Always use full macro-year for region indices
        macro_start = date(macro_year, 1, 1)
        macro_end = date(macro_year, 12, 31)

        # Build regional monthly indices (FULL YEAR)
        df_region_year = df_daily_store[df_daily_store["region"] == region_choice].copy()
        df_region_year = df_region_year[(df_region_year["date"] >= pd.Timestamp(macro_start)) & (df_region_year["date"] <= pd.Timestamp(macro_end))].copy()

        region_month = pd.DataFrame()
        if not df_region_year.empty:
            region_month = df_region_year.copy()
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
            base_val = nonzero.iloc[0]
            return s / base_val * 100.0

        if not region_month.empty:
            region_month["region_turnover_index"] = index_from_first_nonzero(region_month["region_turnover"])
            region_month["region_footfall_index"] = index_from_first_nonzero(region_month["region_footfall"])

            # Highlight selected period window in macro charts
            region_month["is_selected_window"] = (
                (region_month["month"] >= pd.Timestamp(start_period).to_period("M").to_timestamp())
                & (region_month["month"] <= pd.Timestamp(end_period).to_period("M").to_timestamp())
            )

        macro_col1, macro_col2 = st.columns(2)

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
                a = region_month.rename(columns={"month": "date"})[["date", "region_footfall_index", "is_selected_window"]].copy()
                a["series"] = "Regio footfall-index"
                a = a.rename(columns={"region_footfall_index": "value"})
                lines.append(a)

                b = region_month.rename(columns={"month": "date"})[["date", "region_turnover_index", "is_selected_window"]].copy()
                b["series"] = "Regio omzet-index"
                b = b.rename(columns={"region_turnover_index": "value"})
                lines.append(b)

            if not cbs_retail_month.empty and "cbs_retail_index" in cbs_retail_month.columns:
                c = cbs_retail_month[["date", "cbs_retail_index"]].copy()
                c["series"] = "CBS detailhandelindex"
                c = c.rename(columns={"cbs_retail_index": "value"})
                c["is_selected_window"] = False
                lines.append(c)

            if lines:
                macro_df = pd.concat(lines, ignore_index=True)

                highlight = None
                if "is_selected_window" in macro_df.columns:
                    highlight = (
                        alt.Chart(macro_df[macro_df["is_selected_window"] == True].dropna(subset=["date"]))
                        .mark_area(opacity=0.12, color=PFM_PURPLE)
                        .encode(x="date:T")
                    )

                base_chart = (
                    alt.Chart(macro_df.dropna(subset=["date"]))
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

                if highlight is not None:
                    st.altair_chart(highlight + base_chart, use_container_width=True)
                else:
                    st.altair_chart(base_chart, use_container_width=True)
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
                c["is_selected_window"] = False
                lines.append(c)

            if not region_month.empty:
                a = region_month.rename(columns={"month": "date"})[["date", "region_footfall_index", "is_selected_window"]].copy()
                a["series"] = "Regio footfall-index"
                a = a.rename(columns={"region_footfall_index": "value"})
                lines.append(a)

                b = region_month.rename(columns={"month": "date"})[["date", "region_turnover_index", "is_selected_window"]].copy()
                b["series"] = "Regio omzet-index"
                b = b.rename(columns={"region_turnover_index": "value"})
                lines.append(b)

            if lines:
                macro_df = pd.concat(lines, ignore_index=True)

                highlight = (
                    alt.Chart(macro_df[macro_df["is_selected_window"] == True].dropna(subset=["date"]))
                    .mark_area(opacity=0.12, color=PFM_PURPLE)
                    .encode(x="date:T")
                )

                base_chart = (
                    alt.Chart(macro_df.dropna(subset=["date"]))
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

                st.altair_chart(highlight + base_chart, use_container_width=True)
            else:
                st.info("Geen CCI-data beschikbaar.")

            st.markdown("</div>", unsafe_allow_html=True)

    # Debug expander
    with st.expander("🔧 Debug"):
        st.write("Retailer:", selected_client)
        st.write("Regio:", region_choice)
        st.write("Periode:", start_period, "→", end_period, f"({period_choice})")
        st.write("Macro year:", macro_year)
        st.write("Compare all regions:", compare_all_regions)
        st.write("Store key col:", store_key_col)
        st.write("All shops:", len(all_shop_ids), "Region shops:", len(region_shop_ids))
        st.write("df_norm head:", df_norm.head())
        st.write("df_daily_store head:", df_daily_store.head())
        st.write("df_region_daily head:", df_region_daily.head())
        st.write("svi_all cols:", svi_all.columns.tolist() if isinstance(svi_all, pd.DataFrame) else "n/a")
        if isinstance(opp_base, pd.DataFrame):
            st.write("opp_base head:", opp_base.head())


if __name__ == "__main__":
    main()
