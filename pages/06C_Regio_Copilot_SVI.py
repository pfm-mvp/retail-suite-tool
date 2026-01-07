# pages/06C_Region_Copilot_V2.py

import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt

from datetime import datetime, timedelta

from helpers_clients import load_clients
from helpers_normalize import normalize_vemcount_response
from helpers_periods import period_catalog
from helpers_vemcount_api import VemcountApiConfig, fetch_report, build_report_params

from services.cbs_service import (
    get_cci_series,
    get_retail_index,
)

# ----------------------
# Page config
# ----------------------
st.set_page_config(
    page_title="PFM Region Copilot v2 (Regio)",
    layout="wide"
)

# ----------------------
# PFM brand-ish colors
# ----------------------
PFM_PURPLE = "#762181"
PFM_RED = "#F04438"
PFM_DARK = "#111827"
PFM_GRAY = "#6B7280"
PFM_LIGHT = "#F3F4F6"
PFM_LINE = "#E5E7EB"
PFM_GREEN = "#22C55E"
PFM_AMBER = "#F59E0B"

# Better visible “other regions” color (still on-brand)
OTHER_REGION_PURPLE = "#C4B5FD"  # stronger than #D8B4FE
BLACK = "#111111"

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
# Minimal CSS
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
      .pill {{
        display:inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 800;
        border: 1px solid {PFM_LINE};
        background: {PFM_LIGHT};
        color: {PFM_DARK};
      }}
      .muted {{
        color: {PFM_GRAY};
        font-size: 0.86rem;
      }}
      .hint {{
        color: {PFM_GRAY};
        font-size: 0.82rem;
      }}
      .callout {{
        border: 1px solid {PFM_LINE};
        border-radius: 14px;
        background: #fff7ed;
        padding: 0.75rem 1rem;
      }}
      .callout-title {{
        font-weight: 900;
        color: {PFM_DARK};
        margin-bottom: 0.15rem;
      }}
      .callout-sub {{
        color: {PFM_GRAY};
        font-size: 0.86rem;
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

def safe_div(a, b):
    try:
        if pd.isna(a) or pd.isna(b) or float(b) == 0.0:
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan

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
def fetch_region_street_traffic(region: str, start_date, end_date) -> pd.DataFrame:
    """
    Reads weekly street footfall per region (Pathzz sample).
    Must contain columns: region ; week ; street_footfall
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
# Data helpers
# ----------------------
def _coerce_numeric(df: pd.DataFrame, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def collapse_to_daily_store(df: pd.DataFrame, store_key_col: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])

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
        if c in ("footfall", "turnover", "transactions"):
            agg[c] = "sum"
        else:
            agg[c] = "mean"

    out = out.groupby(group_cols, as_index=False).agg(agg)

    # recompute derived metrics robustly
    if "turnover" in out.columns and "footfall" in out.columns:
        out["sales_per_visitor"] = np.where(out["footfall"] > 0, out["turnover"] / out["footfall"], np.nan)
    if "transactions" in out.columns and "footfall" in out.columns:
        out["conversion_rate"] = np.where(out["footfall"] > 0, out["transactions"] / out["footfall"] * 100.0, np.nan)
    if "turnover" in out.columns and "transactions" in out.columns:
        out["sales_per_transaction"] = np.where(out["transactions"] > 0, out["turnover"] / out["transactions"], np.nan)

    return out

def aggregate_weekly_region(df_region_daily: pd.DataFrame) -> pd.DataFrame:
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

    if "turnover" in out.columns and "footfall" in out.columns:
        out["sales_per_visitor"] = np.where(out["footfall"] > 0, out["turnover"] / out["footfall"], np.nan)
    if "transactions" in out.columns and "footfall" in out.columns:
        out["conversion_rate"] = np.where(out["footfall"] > 0, out["transactions"] / out["footfall"] * 100.0, np.nan)

    return out

def enrich_merged_with_sqm_from_df_norm(merged: pd.DataFrame, df_norm: pd.DataFrame, store_key_col: str) -> pd.DataFrame:
    if merged is None or merged.empty or df_norm is None or df_norm.empty:
        return merged

    sqm_col_norm = None
    for cand in ["sq_meter", "sqm", "sq_meters", "square_meters"]:
        if cand in df_norm.columns:
            sqm_col_norm = cand
            break
    if sqm_col_norm is None:
        return merged

    tmp = df_norm[[store_key_col, sqm_col_norm]].copy()
    tmp[store_key_col] = pd.to_numeric(tmp[store_key_col], errors="coerce")
    tmp[sqm_col_norm] = pd.to_numeric(tmp[sqm_col_norm], errors="coerce")
    tmp = tmp.dropna(subset=[store_key_col, sqm_col_norm])
    if tmp.empty:
        return merged

    sqm_by_shop = (
        tmp.groupby(store_key_col, as_index=False)[sqm_col_norm]
        .first()
        .rename(columns={sqm_col_norm: "sqm_api"})
    )

    out = merged.copy()
    out["id"] = pd.to_numeric(out["id"], errors="coerce")
    out = out.merge(sqm_by_shop, left_on="id", right_on=store_key_col, how="left")
    if store_key_col in out.columns:
        out = out.drop(columns=[store_key_col])
    return out

def mark_closed_days_as_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    If a day has footfall=0 AND turnover=0 AND transactions=0 -> treat as closed day.
    We set turnover/footfall/transactions to NaN so line charts don't dive to 0.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in ["footfall", "turnover", "transactions"]:
        if c not in out.columns:
            return out

    base = (
        pd.to_numeric(out["footfall"], errors="coerce").fillna(0).eq(0)
        & pd.to_numeric(out["turnover"], errors="coerce").fillna(0).eq(0)
        & pd.to_numeric(out["transactions"], errors="coerce").fillna(0).eq(0)
    )
    cols_to_nan = ["footfall", "turnover", "transactions", "conversion_rate", "sales_per_visitor", "sales_per_transaction", "sales_per_sqm"]
    for c in cols_to_nan:
        if c in out.columns:
            out.loc[base, c] = np.nan
    return out

# ----------------------
# Lever scan score mapping
# ----------------------
def ratio_to_score_0_100(ratio_pct: float, floor: float, cap: float) -> float:
    """
    ratio_pct: 100 = benchmark
    floor/cap: e.g. 80..120 makes it more sensitive than 70..130
    """
    if pd.isna(ratio_pct):
        return np.nan
    r = float(np.clip(ratio_pct, floor, cap))
    return (r - floor) / (cap - floor) * 100.0

def nice_component_name(key: str) -> str:
    mapping = {
        "sales_per_sqm": "Sales / m² (€)",
        "capture_rate": "Capture rate (%)",
        "conversion_rate": "Conversie (%)",
        "sales_per_visitor": "SPV (€ / visitor)",
        "sales_per_transaction": "ATV (€)",
        "footfall": "Footfall",
        "turnover": "Omzet",
    }
    return mapping.get(key, key)

# ----------------------
# SVI: composite + explainable
# ----------------------
SVI_DRIVERS = [
    ("sales_per_visitor", "SPV (€/visitor)"),
    ("sales_per_sqm", "Sales / m² (€)"),
    ("capture_rate", "Capture rate (%)"),
    ("conversion_rate", "Conversion (%)"),         # optional if transactions present
    ("sales_per_transaction", "ATV (€)"),          # optional if transactions present
]

def compute_driver_values_from_period(footfall, turnover, transactions, sqm_sum, capture_pct):
    # base drivers
    spv = safe_div(turnover, footfall)
    spsqm = safe_div(turnover, sqm_sum)
    cr = safe_div(transactions, footfall) * 100.0 if (pd.notna(transactions) and pd.notna(footfall) and float(footfall) != 0.0) else np.nan
    atv = safe_div(turnover, transactions)
    cap = capture_pct  # already in %
    return {
        "sales_per_visitor": spv,
        "sales_per_sqm": spsqm,
        "capture_rate": cap,
        "conversion_rate": cr,
        "sales_per_transaction": atv,
    }

def compute_svi_explainable(vals_a: dict, vals_b: dict, floor: float, cap: float, weights=None):
    """
    Returns:
      svi_score_0_100
      avg_ratio_pct
      breakdown_df with per driver: label, value_a, value_b, ratio_pct, score_0_100, weight
    """
    if weights is None:
        # equal weights across available drivers
        weights = {k: 1.0 for k, _ in SVI_DRIVERS}

    rows = []
    for key, label in SVI_DRIVERS:
        va = vals_a.get(key, np.nan)
        vb = vals_b.get(key, np.nan)

        ratio = np.nan
        if pd.notna(va) and pd.notna(vb) and float(vb) != 0.0:
            ratio = (float(va) / float(vb)) * 100.0

        score = ratio_to_score_0_100(ratio, floor=float(floor), cap=float(cap))
        w = float(weights.get(key, 1.0))

        # only include if ratio is valid
        include = pd.notna(ratio) and pd.notna(score)
        rows.append({
            "driver_key": key,
            "driver": label,
            "value": va,
            "benchmark": vb,
            "ratio_pct": ratio,
            "score": score,
            "weight": w,
            "include": include,
        })

    bd = pd.DataFrame(rows)

    usable = bd[bd["include"]].copy()
    if usable.empty:
        return np.nan, np.nan, bd.drop(columns=["include"])

    # Weighted average in ratio-space (more interpretable), then map to score
    usable["w"] = usable["weight"].astype(float)
    wsum = float(usable["w"].sum()) if float(usable["w"].sum()) > 0 else float(len(usable))
    avg_ratio = float((usable["ratio_pct"] * usable["w"]).sum() / wsum)

    svi = ratio_to_score_0_100(avg_ratio, floor=float(floor), cap=float(cap))

    return float(svi), float(avg_ratio), bd.drop(columns=["include"])

def style_heatmap_ratio(val):
    """
    val = ratio % vs benchmark (100 = equal)
    green when >110, amber around 90-110, red when <90
    """
    try:
        if pd.isna(val):
            return ""
        v = float(val)
        # soft thresholds
        if v >= 110:
            return "background-color: #ecfdf5; color:#065f46; font-weight:800;"
        if v >= 95:
            return "background-color: #fffbeb; color:#92400e; font-weight:800;"
        return "background-color: #fff1f2; color:#9f1239; font-weight:800;"
    except Exception:
        return ""

# ----------------------
# MAIN
# ----------------------
def main():
    # ---------- Session state to prevent "Analyseer" wipe on drilldown ----------
    if "rcp_last_key" not in st.session_state:
        st.session_state.rcp_last_key = None
    if "rcp_payload" not in st.session_state:
        st.session_state.rcp_payload = None
    if "rcp_ran" not in st.session_state:
        st.session_state.rcp_ran = False

    st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)

    header_left, header_right = st.columns([2.2, 1.8])

    with header_left:
        st.markdown(
            f"""
            <div class="pfm-header">
              <div>
                <div class="pfm-title">PFM Region Performance Copilot <span class="pill">v2</span></div>
                <div class="pfm-sub">Regio-level: explainable SVI + heatmap scanning + value upside + drilldown + macro</div>
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

    start_period = periods[period_choice].start
    end_period = periods[period_choice].end

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

    # store display name
    if "store_label" in merged.columns and merged["store_label"].notna().any():
        merged["store_display"] = merged["store_label"]
    else:
        merged["store_display"] = merged["name"] if "name" in merged.columns else merged["id"].astype(str)

    available_regions = sorted(merged["region"].dropna().unique().tolist())

    top_controls = st.columns([1.2, 1.2, 1.2, 1.2, 1.2])
    with top_controls[0]:
        region_choice = st.selectbox("Regio", available_regions)
    with top_controls[1]:
        show_macro = st.toggle("Toon macro (CBS/CCI)", value=True)
    with top_controls[2]:
        show_quadrant = st.toggle("Toon quadrant", value=True)
    with top_controls[3]:
        lever_floor = st.selectbox("SVI gevoeligheid (vloer)", [70, 75, 80, 85], index=2)
    with top_controls[4]:
        run_btn = st.button("Analyseer", type="primary")

    lever_cap = 200 - lever_floor  # e.g. 80 -> 120 ; 85 -> 115

    run_key = (company_id, region_choice, str(start_period), str(end_period), int(lever_floor), int(lever_cap))
    should_fetch = run_btn or (st.session_state.rcp_last_key != run_key) or (not st.session_state.rcp_ran)

    if not should_fetch and st.session_state.rcp_payload is None:
        st.info("Selecteer retailer/regio/periode en klik op **Analyseer**.")
        return

    if should_fetch:
        region_shops = merged[merged["region"] == region_choice].copy()
        region_shop_ids = region_shops["id"].dropna().astype(int).unique().tolist()
        if not region_shop_ids:
            st.warning(f"Geen winkels gevonden voor regio '{region_choice}'.")
            return

        all_shop_ids = merged["id"].dropna().astype(int).unique().tolist()

        # NOTE: transactions included (needed for CR & ATV)
        metric_map = {
            "count_in": "footfall",
            "turnover": "turnover",
            "transactions": "transactions",
            "conversion_rate": "conversion_rate",
            "sales_per_visitor": "sales_per_visitor",
            "avg_basket_size": "avg_basket_size",
            "sales_per_sqm": "sales_per_sqm",
            "sales_per_transaction": "sales_per_transaction",
        }

        cfg = VemcountApiConfig(report_url=REPORT_URL)

        params_preview = build_report_params(
            shop_ids=all_shop_ids,
            data_outputs=list(metric_map.keys()),
            period="date",
            step="day",
            source="shops",
            date_from=start_period,
            date_to=end_period,
        )

        with st.spinner("Data ophalen via FastAPI..."):
            try:
                resp = fetch_report(
                    cfg=cfg,
                    shop_ids=all_shop_ids,
                    data_outputs=list(metric_map.keys()),
                    period="date",
                    step="day",
                    source="shops",
                    date_from=start_period,
                    date_to=end_period,
                    timeout=120,
                )
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ HTTPError bij /get-report: {e}")
                try:
                    st.code(e.response.text)
                except Exception:
                    pass
                with st.expander("🔧 Debug request (params)"):
                    st.write("REPORT_URL:", REPORT_URL)
                    st.write("Params:", params_preview)
                return
            except requests.exceptions.RequestException as e:
                st.error(f"❌ RequestException bij /get-report: {e}")
                with st.expander("🔧 Debug request (params)"):
                    st.write("REPORT_URL:", REPORT_URL)
                    st.write("Params:", params_preview)
                return

        df_norm = normalize_vemcount_response(resp, kpi_keys=metric_map.keys()).rename(columns=metric_map)

        if df_norm is None or df_norm.empty:
            st.warning("Geen data ontvangen voor de gekozen selectie.")
            return

        store_key_col = None
        for cand in ["shop_id", "id", "location_id"]:
            if cand in df_norm.columns:
                store_key_col = cand
                break
        if store_key_col is None:
            st.error("Geen store-id kolom gevonden in de response (shop_id/id/location_id).")
            return

        merged2 = enrich_merged_with_sqm_from_df_norm(merged, df_norm, store_key_col=store_key_col)

        # sqm_effective: override > locations sqm > sqm_api
        sqm_col = None
        for cand in ["sqm", "sq_meter", "sq_meters", "square_meters"]:
            if cand in merged2.columns:
                sqm_col = cand
                break

        base_sqm = pd.to_numeric(merged2[sqm_col], errors="coerce") if sqm_col is not None else np.nan
        sqm_api = pd.to_numeric(merged2["sqm_api"], errors="coerce") if "sqm_api" in merged2.columns else np.nan

        merged2["sqm_effective"] = np.where(
            merged2["sqm_override"].notna(),
            pd.to_numeric(merged2["sqm_override"], errors="coerce"),
            np.where(pd.notna(base_sqm), base_sqm, sqm_api)
        )

        df_daily_store = collapse_to_daily_store(df_norm, store_key_col=store_key_col)
        if df_daily_store is None or df_daily_store.empty:
            st.warning("Geen data na opschonen (daily/store collapse).")
            return

        join_cols = ["id", "store_display", "region", "sqm_effective"]
        if "store_type" in merged2.columns:
            join_cols.append("store_type")

        df_daily_store = df_daily_store.merge(
            merged2[join_cols].drop_duplicates(),
            left_on=store_key_col,
            right_on="id",
            how="left",
        )

        # Compute sales_per_sqm robustly (if API delivers empty)
        if "sales_per_sqm" not in df_daily_store.columns:
            df_daily_store["sales_per_sqm"] = np.nan

        sqm_eff = pd.to_numeric(df_daily_store.get("sqm_effective", np.nan), errors="coerce")
        turn = pd.to_numeric(df_daily_store.get("turnover", np.nan), errors="coerce")
        calc_spm2 = np.where((pd.notna(sqm_eff) & (sqm_eff > 0)), (turn / sqm_eff), np.nan)

        df_daily_store["sales_per_sqm"] = pd.to_numeric(df_daily_store["sales_per_sqm"], errors="coerce")
        df_daily_store["sales_per_sqm"] = df_daily_store["sales_per_sqm"].combine_first(pd.Series(calc_spm2, index=df_daily_store.index))

        # Smooth closed days to NaN for nicer trends
        df_daily_store = mark_closed_days_as_nan(df_daily_store)

        df_region_daily = df_daily_store[df_daily_store["region"] == region_choice].copy()

        # -------- cache payload --------
        st.session_state.rcp_last_key = run_key
        st.session_state.rcp_payload = {
            "df_daily_store": df_daily_store,
            "df_region_daily": df_region_daily,
            "merged": merged2,
            "store_key_col": store_key_col,
            "start_period": start_period,
            "end_period": end_period,
            "selected_client": selected_client,
            "region_choice": region_choice,
        }
        st.session_state.rcp_ran = True

    # ---------- read cached ----------
    payload = st.session_state.rcp_payload
    df_daily_store = payload["df_daily_store"]
    df_region_daily = payload["df_region_daily"]
    merged = payload["merged"]
    start_period = payload["start_period"]
    end_period = payload["end_period"]
    selected_client = payload["selected_client"]
    region_choice = payload["region_choice"]

    if df_region_daily is None or df_region_daily.empty:
        st.warning("Geen data voor geselecteerde regio binnen de periode.")
        return

    # ----------------------
    # Header + Region totals
    # ----------------------
    st.markdown(f"## {selected_client['brand']} — Regio **{region_choice}** · {start_period} → {end_period}")

    foot_total = float(pd.to_numeric(df_region_daily["footfall"], errors="coerce").dropna().sum()) if "footfall" in df_region_daily.columns else 0.0
    turn_total = float(pd.to_numeric(df_region_daily["turnover"], errors="coerce").dropna().sum()) if "turnover" in df_region_daily.columns else 0.0
    trans_total = float(pd.to_numeric(df_region_daily["transactions"], errors="coerce").dropna().sum()) if "transactions" in df_region_daily.columns else 0.0

    conv = (trans_total / foot_total * 100.0) if foot_total > 0 else np.nan
    spv = (turn_total / foot_total) if foot_total > 0 else np.nan
    atv = (turn_total / trans_total) if trans_total > 0 else np.nan

    # ----------------------
    # Capture weekly chart data (region) + avg capture
    # ----------------------
    region_weekly = aggregate_weekly_region(df_region_daily)
    pathzz_weekly = fetch_region_street_traffic(region=region_choice, start_date=start_period, end_date=end_period)

    capture_weekly = pd.DataFrame()
    avg_capture = np.nan

    if not region_weekly.empty and not pathzz_weekly.empty:
        region_weekly = region_weekly.copy()
        region_weekly["week_start"] = pd.to_datetime(region_weekly["week_start"], errors="coerce")
        region_weekly = region_weekly.dropna(subset=["week_start"])
        region_weekly = region_weekly.groupby("week_start", as_index=False).agg(
            footfall=("footfall", "sum"),
            turnover=("turnover", "sum") if "turnover" in region_weekly.columns else ("footfall", "sum"),
            transactions=("transactions", "sum") if "transactions" in region_weekly.columns else ("footfall", "sum"),
        )

        pathzz_weekly = pathzz_weekly.copy()
        pathzz_weekly["week_start"] = pd.to_datetime(pathzz_weekly["week_start"], errors="coerce")
        pathzz_weekly = pathzz_weekly.dropna(subset=["week_start"])
        pathzz_weekly = pathzz_weekly.groupby("week_start", as_index=False).agg(street_footfall=("street_footfall", "mean"))

        capture_weekly = pd.merge(region_weekly, pathzz_weekly, on="week_start", how="inner")

    if not capture_weekly.empty:
        capture_weekly["capture_rate"] = np.where(
            capture_weekly["street_footfall"] > 0,
            capture_weekly["footfall"] / capture_weekly["street_footfall"] * 100.0,
            np.nan,
        )
        avg_capture = float(pd.to_numeric(capture_weekly["capture_rate"], errors="coerce").dropna().mean())

    # ----------------------
    # Company baseline (period totals)
    # ----------------------
    def agg_period(df_: pd.DataFrame) -> dict:
        foot = float(pd.to_numeric(df_.get("footfall", 0), errors="coerce").dropna().sum())
        turn = float(pd.to_numeric(df_.get("turnover", 0), errors="coerce").dropna().sum())
        trans = float(pd.to_numeric(df_.get("transactions", 0), errors="coerce").dropna().sum())

        # sqm sum: unique stores (avoid daily duplicates)
        sqm = pd.to_numeric(df_.get("sqm_effective", np.nan), errors="coerce")
        sqm_sum = float(sqm.dropna().drop_duplicates().sum()) if sqm.notna().any() else np.nan

        return {
            "footfall": foot,
            "turnover": turn,
            "transactions": trans,
            "sqm_sum": sqm_sum,
        }

    reg_tot = agg_period(df_region_daily)
    comp_tot = agg_period(df_daily_store)

    # SVI driver values (region vs company)
    reg_vals = compute_driver_values_from_period(
        footfall=reg_tot["footfall"],
        turnover=reg_tot["turnover"],
        transactions=reg_tot["transactions"],
        sqm_sum=reg_tot["sqm_sum"],
        capture_pct=avg_capture,  # already %
    )

    comp_vals = compute_driver_values_from_period(
        footfall=comp_tot["footfall"],
        turnover=comp_tot["turnover"],
        transactions=comp_tot["transactions"],
        sqm_sum=comp_tot["sqm_sum"],
        capture_pct=np.nan,  # company capture not available without company street traffic
    )

    # For capture benchmark we use region capture vs itself (neutral) unless you later add company street traffic.
    # That means capture driver won’t distort region-vs-company SVI; but it remains visible in breakdown & heatmap vs region benchmark.
    # (This is deliberate: better honest than fake precision.)
    if pd.isna(comp_vals.get("capture_rate", np.nan)):
        comp_vals["capture_rate"] = reg_vals.get("capture_rate", np.nan)

    # ----------------------
    # KPI cards row
    # ----------------------
    k1, k2, k3, k4, k5 = st.columns([1, 1, 1, 1, 1])
    with k1:
        kpi_card("Footfall", fmt_int(foot_total), "Regio · periode")
    with k2:
        kpi_card("Omzet", fmt_eur(turn_total), "Regio · periode")
    with k3:
        kpi_card("Conversion", fmt_pct(conv), "Transacties / bezoekers")
    with k4:
        kpi_card("ATV", fmt_eur_2(atv), "Omzet / transactie")
    with k5:
        kpi_card("Capture", fmt_pct(avg_capture), "Regio totaal (Pathzz)")

    st.markdown(
        "<div class='muted'>Benchmark: Company = alle shops in company (zelfde periode). Regio = shops binnen gekozen regio. Capture is Pathzz-regio-weekniveau.</div>",
        unsafe_allow_html=True,
    )

    # ======================
    # 1) Explainable SVI (composite + breakdown)
    # ======================
    region_svi, region_avg_ratio, region_bd = compute_svi_explainable(
        vals_a=reg_vals,
        vals_b=comp_vals,
        floor=float(lever_floor),
        cap=float(lever_cap),
        weights=None
    )
    status_txt, status_color = status_from_score(region_svi if pd.notna(region_svi) else 0)

    c_svi_1, c_svi_2 = st.columns([1.1, 2.9])
    with c_svi_1:
        st.altair_chart(gauge_chart(region_svi if pd.notna(region_svi) else 0, status_color), use_container_width=False)
    with c_svi_2:
        st.markdown(
            f"""
            <div class="panel">
              <div class="panel-title">Regio Vitality Index (SVI) — composite & explainable</div>
              <div style="font-size:2rem;font-weight:900;color:{PFM_DARK};line-height:1.1">
                {"" if pd.isna(region_svi) else f"{region_svi:.0f}"} <span class="pill">/ 100</span>
              </div>
              <div class="muted" style="margin-top:0.35rem">
                Status: <span style="font-weight:900;color:{status_color}">{status_txt}</span><br/>
                Gemiddelde ratio vs company (drivers) ≈ <b>{"" if pd.isna(region_avg_ratio) else f"{region_avg_ratio:.0f}%"} </b>
                <span class="hint"> (ratio wordt geclipt {lever_floor}–{lever_cap}% → 0–100)</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Breakdown: compact bar chart + table
    st.markdown('<div class="panel"><div class="panel-title">SVI breakdown — drivers</div>', unsafe_allow_html=True)

    bd = region_bd.copy()
    bd["ratio_pct"] = pd.to_numeric(bd["ratio_pct"], errors="coerce")
    bd["score"] = pd.to_numeric(bd["score"], errors="coerce")
    bd = bd.dropna(subset=["ratio_pct", "score"])

    if bd.empty:
        st.info("Nog onvoldoende data om drivers te berekenen (check footfall/omzet/sqm/transactions).")
    else:
        # bar: driver score
        bar = (
            alt.Chart(bd)
            .mark_bar(cornerRadiusEnd=4, color=PFM_PURPLE)
            .encode(
                x=alt.X("score:Q", title="Driver score (0–100)", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("driver:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip("driver:N", title="Driver"),
                    alt.Tooltip("ratio_pct:Q", title="Ratio vs benchmark (%)", format=".1f"),
                    alt.Tooltip("score:Q", title="Score", format=".0f"),
                ],
            )
            .properties(height=220)
        )
        st.altair_chart(bar, use_container_width=True)

        # table
        bd_tbl = bd.copy()
        bd_tbl["value"] = bd_tbl["value"].apply(lambda x: fmt_eur_2(x) if pd.notna(x) and x < 100000 else (fmt_pct(x) if "rate" in str(bd_tbl.loc[bd_tbl["value"] == x, "driver_key"].values[:1]) else fmt_eur_2(x)))
        # safer formatting explicitly:
        def _fmt_driver_row(r):
            key = r["driver_key"]
            if key in ("conversion_rate", "capture_rate"):
                return fmt_pct(r["value"])
            return fmt_eur_2(r["value"])
        def _fmt_bench_row(r):
            key = r["driver_key"]
            if key in ("conversion_rate", "capture_rate"):
                return fmt_pct(r["benchmark"])
            return fmt_eur_2(r["benchmark"])

        bd_show = bd.copy()
        bd_show["Regio"] = bd_show.apply(_fmt_driver_row, axis=1)
        bd_show["Company"] = bd_show.apply(_fmt_bench_row, axis=1)
        bd_show["Ratio vs company"] = bd_show["ratio_pct"].apply(lambda x: "-" if pd.isna(x) else f"{x:.0f}%")
        bd_show["Score"] = bd_show["score"].apply(lambda x: "-" if pd.isna(x) else f"{x:.0f}")
        bd_show = bd_show[["driver", "Regio", "Company", "Ratio vs company", "Score"]].rename(columns={"driver": "Driver"})
        st.dataframe(bd_show, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ======================
    # Weekly trend — Store vs Street + Capture (week labels)
    # ======================
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

    # ======================
    # 2) Heatmap table (scanning-machine)
    # ======================
    st.markdown("## Heatmap — stores vs benchmark (scanning-machine)")

    reg_store_daily = df_daily_store[df_daily_store["region"] == region_choice].copy()

    agg = reg_store_daily.groupby(["id", "store_display"], as_index=False).agg(
        turnover=("turnover", "sum"),
        footfall=("footfall", "sum"),
        transactions=("transactions", "sum"),
    )

    agg["conversion_rate"] = np.where(agg["footfall"] > 0, agg["transactions"] / agg["footfall"] * 100.0, np.nan)
    agg["sales_per_visitor"] = np.where(agg["footfall"] > 0, agg["turnover"] / agg["footfall"], np.nan)
    agg["sales_per_transaction"] = np.where(agg["transactions"] > 0, agg["turnover"] / agg["transactions"], np.nan)

    sqm_map = merged.loc[merged["region"] == region_choice, ["id", "sqm_effective"]].drop_duplicates()
    sqm_map["sqm_effective"] = pd.to_numeric(sqm_map["sqm_effective"], errors="coerce")
    agg = agg.merge(sqm_map, on="id", how="left")
    agg["sales_per_sqm"] = np.where((agg["sqm_effective"] > 0) & pd.notna(agg["sqm_effective"]), agg["turnover"] / agg["sqm_effective"], np.nan)

    # Store capture proxy: store weekly footfall / region street weekly (then average)
    store_capture_proxy = {}
    if not capture_weekly.empty:
        cw = capture_weekly[["week_start", "street_footfall"]].copy()
        cw["week_start"] = pd.to_datetime(cw["week_start"], errors="coerce")
        cw = cw.dropna(subset=["week_start"])

        dd = reg_store_daily.copy()
        dd["date"] = pd.to_datetime(dd["date"], errors="coerce")
        dd = dd.dropna(subset=["date"])
        dd["week_start"] = dd["date"].dt.to_period("W-SAT").dt.start_time

        sw = dd.groupby(["id", "week_start"], as_index=False).agg(footfall=("footfall", "sum"))
        sw = sw.merge(cw, on="week_start", how="inner")
        sw["cap_proxy"] = np.where(sw["street_footfall"] > 0, sw["footfall"] / sw["street_footfall"] * 100.0, np.nan)

        for sid, g in sw.groupby("id"):
            store_capture_proxy[int(sid)] = float(pd.to_numeric(g["cap_proxy"], errors="coerce").dropna().mean())

    agg["capture_rate"] = agg["id"].apply(lambda x: store_capture_proxy.get(int(x), np.nan) if pd.notna(x) else np.nan)

    # Benchmark for heatmap = REGIO baseline (more operationally useful)
    reg_bench = compute_driver_values_from_period(
        footfall=reg_tot["footfall"],
        turnover=reg_tot["turnover"],
        transactions=reg_tot["transactions"],
        sqm_sum=reg_tot["sqm_sum"],
        capture_pct=avg_capture,  # region capture
    )

    # Store SVI (vs region benchmark) + indices per driver
    def store_driver_vals(row):
        return compute_driver_values_from_period(
            footfall=row["footfall"],
            turnover=row["turnover"],
            transactions=row["transactions"],
            sqm_sum=row["sqm_effective"],
            capture_pct=row["capture_rate"],
        )

    # driver ratios & store svi
    weights = {
        "sales_per_visitor": 1.0,
        "sales_per_sqm": 1.0,
        "capture_rate": 1.0,
        "conversion_rate": 1.0,
        "sales_per_transaction": 1.0,
    }

    svi_list = []
    ratios_map = {k: [] for k, _ in SVI_DRIVERS}
    for _, r in agg.iterrows():
        vals = store_driver_vals(r)
        svi, avg_ratio, bd = compute_svi_explainable(vals, reg_bench, float(lever_floor), float(lever_cap), weights=weights)
        svi_list.append(svi)

        # collect ratios per driver for heatmap columns
        bd = bd.copy()
        for dk, _ in SVI_DRIVERS:
            rr = bd.loc[bd["driver_key"] == dk, "ratio_pct"]
            ratios_map[dk].append(float(rr.iloc[0]) if (not rr.empty and pd.notna(rr.iloc[0])) else np.nan)

    agg["SVI"] = svi_list
    for dk, _ in SVI_DRIVERS:
        agg[f"{dk}_idx"] = ratios_map[dk]

    # Heatmap table view
    heat = agg.copy()
    # Keep it “scanable”
    heat["SVI"] = pd.to_numeric(heat["SVI"], errors="coerce")

    # Optional upside column (computed below, but also handy to sort)
    heat["upside_period_eur"] = np.nan
    heat["upside_driver"] = ""

    # ======================
    # 3) Value Upside (scenario) — period + annualized + driver label
    # ======================
    days_in_period = max(1, (pd.to_datetime(end_period) - pd.to_datetime(start_period)).days + 1)

    # driver gaps vs region benchmark
    def calc_upside_for_store(row):
        """
        Returns (upside_period_eur, driver_label)
        Conservative: lift weakest driver to benchmark, estimate incremental turnover.
        """
        foot = row["footfall"]
        turn = row["turnover"]
        sqm = row["sqm_effective"]
        trans = row["transactions"]

        # store driver values
        spv_s = safe_div(turn, foot)
        spsqm_s = safe_div(turn, sqm)
        cap_s = row.get("capture_rate", np.nan)

        # benchmark values (region)
        spv_b = reg_bench.get("sales_per_visitor", np.nan)
        spsqm_b = reg_bench.get("sales_per_sqm", np.nan)
        cap_b = reg_bench.get("capture_rate", np.nan)

        # compute candidate upsides (turnover)
        candidates = []

        # 1) SPV upside (footfall held constant)
        if pd.notna(foot) and foot > 0 and pd.notna(spv_s) and pd.notna(spv_b) and spv_s < spv_b:
            candidates.append(("Low SPV", float(foot) * float(spv_b - spv_s)))

        # 2) Sales/m² upside (sqm held constant)
        if pd.notna(sqm) and sqm > 0 and pd.notna(spsqm_s) and pd.notna(spsqm_b) and spsqm_s < spsqm_b:
            candidates.append(("Low Sales / m²", float(sqm) * float(spsqm_b - spsqm_s)))

        # 3) Capture upside (street traffic fixed, lift capture to benchmark → more footfall → more turnover via store SPV)
        # proxy: if cap_s < cap_b then extra footfall = foot * (cap_b/cap_s - 1)
        if pd.notna(cap_s) and pd.notna(cap_b) and cap_s > 0 and cap_s < cap_b and pd.notna(spv_s):
            extra_foot = float(foot) * (float(cap_b) / float(cap_s) - 1.0)
            candidates.append(("Low Capture", max(0.0, extra_foot) * float(spv_s)))

        if not candidates:
            return np.nan, ""

        # pick biggest
        best = sorted(candidates, key=lambda x: x[1], reverse=True)[0]
        upside = float(best[1]) if best[1] > 0 else np.nan
        return upside, best[0]

    ups = heat.apply(calc_upside_for_store, axis=1, result_type="expand")
    if isinstance(ups, pd.DataFrame) and ups.shape[1] >= 2:
        heat["upside_period_eur"] = pd.to_numeric(ups.iloc[:, 0], errors="coerce")
        heat["upside_driver"] = ups.iloc[:, 1].astype(str)

    heat["upside_annual_eur"] = heat["upside_period_eur"] * (365.0 / float(days_in_period))

    # Format ratios for heatmap display
    heat_show = heat[[
        "store_display",
        "SVI",
        "turnover",
        "footfall",
        "sales_per_visitor_idx",
        "sales_per_sqm_idx",
        "capture_rate_idx",
        "conversion_rate_idx",
        "sales_per_transaction_idx",
        "upside_period_eur",
        "upside_annual_eur",
        "upside_driver",
    ]].copy()

    heat_show = heat_show.rename(columns={
        "store_display": "Winkel",
        "turnover": "Omzet",
        "footfall": "Footfall",
        "sales_per_visitor_idx": "SPV idx",
        "sales_per_sqm_idx": "Sales/m² idx",
        "capture_rate_idx": "Capture idx",
        "conversion_rate_idx": "CR idx",
        "sales_per_transaction_idx": "ATV idx",
        "upside_period_eur": "Upside (periode)",
        "upside_annual_eur": "Upside (annualized)",
        "upside_driver": "Main driver",
    })

    # Format base numeric columns (keep idx numeric for styling)
    heat_show["SVI"] = pd.to_numeric(heat_show["SVI"], errors="coerce")
    heat_show["Omzet"] = pd.to_numeric(heat_show["Omzet"], errors="coerce")
    heat_show["Footfall"] = pd.to_numeric(heat_show["Footfall"], errors="coerce")
    heat_show["Upside (periode)"] = pd.to_numeric(heat_show["Upside (periode)"], errors="coerce")
    heat_show["Upside (annualized)"] = pd.to_numeric(heat_show["Upside (annualized)"], errors="coerce")

    # Display: first a plain sortable table (fast), then optional styled “heatmap” view
    cA, cB = st.columns([2, 1])
    with cA:
        st.caption("Sort tip: klik op **SVI** (laag → hoog) of **Upside (annualized)** (hoog → laag) voor snelle focus.")
    with cB:
        show_heat_styling = st.toggle("Toon heatmap-kleuren", value=True)

    if not show_heat_styling:
        disp = heat_show.copy()
        disp["SVI"] = disp["SVI"].apply(lambda x: "-" if pd.isna(x) else f"{x:.0f}")
        disp["Omzet"] = disp["Omzet"].apply(fmt_eur)
        disp["Footfall"] = disp["Footfall"].apply(fmt_int)
        for c in ["SPV idx", "Sales/m² idx", "Capture idx", "CR idx", "ATV idx"]:
            disp[c] = disp[c].apply(lambda x: "-" if pd.isna(x) else f"{x:.0f}%")
        disp["Upside (periode)"] = disp["Upside (periode)"].apply(lambda x: "-" if pd.isna(x) else fmt_eur(x))
        disp["Upside (annualized)"] = disp["Upside (annualized)"].apply(lambda x: "-" if pd.isna(x) else fmt_eur(x))
        st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        # styled heatmap columns
        styled = heat_show.copy()
        # friendly formatting in non-idx columns via Styler format
        styler = styled.style

        # apply coloring to idx columns + SVI (optional)
        for col in ["SPV idx", "Sales/m² idx", "Capture idx", "CR idx", "ATV idx"]:
            if col in styled.columns:
                styler = styler.applymap(style_heatmap_ratio, subset=[col])

        # slight highlight for low SVI rows (readability)
        def _svi_row_style(v):
            try:
                if pd.isna(v):
                    return ""
                v = float(v)
                if v < 45:
                    return "background-color:#fff1f2; font-weight:900;"
                if v < 60:
                    return "background-color:#fffbeb; font-weight:900;"
                return ""
            except Exception:
                return ""
        if "SVI" in styled.columns:
            styler = styler.applymap(_svi_row_style, subset=["SVI"])

        styler = styler.format({
            "SVI": lambda x: "-" if pd.isna(x) else f"{x:.0f}",
            "Omzet": lambda x: "-" if pd.isna(x) else fmt_eur(x),
            "Footfall": lambda x: "-" if pd.isna(x) else fmt_int(x),
            "SPV idx": lambda x: "-" if pd.isna(x) else f"{x:.0f}%",
            "Sales/m² idx": lambda x: "-" if pd.isna(x) else f"{x:.0f}%",
            "Capture idx": lambda x: "-" if pd.isna(x) else f"{x:.0f}%",
            "CR idx": lambda x: "-" if pd.isna(x) else f"{x:.0f}%",
            "ATV idx": lambda x: "-" if pd.isna(x) else f"{x:.0f}%",
            "Upside (periode)": lambda x: "-" if pd.isna(x) else fmt_eur(x),
            "Upside (annualized)": lambda x: "-" if pd.isna(x) else fmt_eur(x),
        })

        st.dataframe(styler, use_container_width=True, hide_index=True)

    # ======================
    # Value Upside summary (bulletproof copy)
    # ======================
    st.markdown("## Value Upside (scenario) — biggest opportunities")

    opp = heat_show[["Winkel", "SVI", "Upside (periode)", "Upside (annualized)", "Main driver"]].copy()
    opp = opp.rename(columns={"Upside (periode)": "Upside (periode)", "Upside (annualized)": "Upside (annualized)"})

    opp["SVI_num"] = pd.to_numeric(heat["SVI"], errors="coerce")
    opp["up_period"] = pd.to_numeric(heat["upside_period_eur"], errors="coerce")
    opp["up_annual"] = pd.to_numeric(heat["upside_annual_eur"], errors="coerce")

    opp = opp.dropna(subset=["up_period"]).sort_values("up_period", ascending=False).head(5).copy()

    total_period = float(pd.to_numeric(opp["up_period"], errors="coerce").dropna().sum()) if not opp.empty else np.nan
    total_annual = float(pd.to_numeric(opp["up_annual"], errors="coerce").dropna().sum()) if not opp.empty else np.nan

    st.markdown(
        f"""
        <div class="callout">
          <div class="callout-title">Top 5 upside (in periode): {fmt_eur(total_period) if pd.notna(total_period) else "-"}</div>
          <div class="callout-sub">
            Annualized upside: <b>{fmt_eur(total_annual) if pd.notna(total_annual) else "-"}</b> / jaar
            <span class="hint">(rekenkundige extrapolatie; seizoenseffecten & uitvoerbaarheid bepalen realisme)</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    if opp.empty:
        st.info("Nog geen duidelijke upside te berekenen (check drivers: SPV / Sales/m² / Capture).")
    else:
        show_opp = pd.DataFrame({
            "Winkel": opp["Winkel"].values,
            "Main driver": opp["Main driver"].values,
            "Upside (periode)": opp["up_period"].apply(fmt_eur).values,
            "Upside (annualized)": opp["up_annual"].apply(fmt_eur).values,
        })
        st.dataframe(show_opp, use_container_width=True, hide_index=True)

    # ======================
    # Quadrant (v2) — region overview BEFORE drilldown
    # ======================
    if show_quadrant:
        st.markdown("## Regio quadrant (v2)")

        regs = sorted(merged["region"].dropna().unique().tolist())
        rows = []

        # company totals for quadrant
        comp_vals_all = compute_driver_values_from_period(
            footfall=comp_tot["footfall"],
            turnover=comp_tot["turnover"],
            transactions=comp_tot["transactions"],
            sqm_sum=comp_tot["sqm_sum"],
            capture_pct=np.nan,
        )

        for r in regs:
            drr = df_daily_store[df_daily_store["region"] == r].copy()
            if drr.empty:
                continue

            rt = agg_period(drr)

            # region capture avg if available (Pathzz only for selected region -> set neutral for others)
            reg_capture = avg_capture if r == region_choice else np.nan

            reg_vals_r = compute_driver_values_from_period(
                footfall=rt["footfall"],
                turnover=rt["turnover"],
                transactions=rt["transactions"],
                sqm_sum=rt["sqm_sum"],
                capture_pct=reg_capture,
            )

            # Y: SPV index vs company (100 = equal)
            reg_spv = reg_vals_r.get("sales_per_visitor", np.nan)
            cmp_spv = comp_vals_all.get("sales_per_visitor", np.nan)
            y = (reg_spv / cmp_spv * 100.0) if (pd.notna(reg_spv) and pd.notna(cmp_spv) and cmp_spv != 0) else 100.0

            # X: SVI proxy (drivers vs company)
            # for capture: neutral (use benchmark=self) unless company street data is available
            if pd.isna(reg_vals_r.get("capture_rate", np.nan)):
                reg_vals_r["capture_rate"] = comp_vals_all.get("capture_rate", np.nan)

            svi_proxy, _, _ = compute_svi_explainable(
                vals_a=reg_vals_r,
                vals_b=comp_vals_all,
                floor=float(lever_floor),
                cap=float(lever_cap),
                weights=None
            )

            rows.append({
                "region": r,
                "x_svi_proxy": svi_proxy,
                "y_spv_index": y,
                "is_selected": (r == region_choice),
            })

        quad = pd.DataFrame(rows)
        quad["x_svi_proxy"] = pd.to_numeric(quad["x_svi_proxy"], errors="coerce")
        quad["y_spv_index"] = pd.to_numeric(quad["y_spv_index"], errors="coerce")

        if quad.empty or quad["x_svi_proxy"].dropna().empty:
            st.info("Nog onvoldoende data voor quadrant.")
        else:
            st.caption("X-as: SVI-proxy (0–100) vs company (drivers). Y-as: SPV-index vs company (100 = gelijk).")

            chart = (
                alt.Chart(quad.dropna(subset=["x_svi_proxy", "y_spv_index"]))
                .mark_circle(size=220, opacity=0.95)
                .encode(
                    x=alt.X("x_svi_proxy:Q", title="SVI proxy (0–100)", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("y_spv_index:Q", title="SPV index vs company", axis=alt.Axis(format=".0f")),
                    color=alt.Color(
                        "is_selected:N",
                        scale=alt.Scale(domain=[True, False], range=[PFM_PURPLE, OTHER_REGION_PURPLE]),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("region:N", title="Regio"),
                        alt.Tooltip("x_svi_proxy:Q", title="SVI proxy", format=".0f"),
                        alt.Tooltip("y_spv_index:Q", title="SPV index", format=".0f"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)

    # ======================
    # Store drilldown (after top-level scanning)
    # ======================
    st.markdown("## Store drilldown")

    region_stores = merged[merged["region"] == region_choice].copy()
    region_stores = region_stores.dropna(subset=["id"]).copy()
    region_stores["id_int"] = region_stores["id"].astype(int)
    region_stores["dd_label"] = region_stores["store_display"].fillna(region_stores["id"].astype(str)) + " · " + region_stores["id"].astype(str)

    if "rcp_store_choice" not in st.session_state:
        st.session_state.rcp_store_choice = int(region_stores["id_int"].iloc[0])

    store_choice_label = st.selectbox(
        "Winkel",
        region_stores["dd_label"].tolist(),
        index=int(np.where(region_stores["id_int"].values == st.session_state.rcp_store_choice)[0][0]) if (st.session_state.rcp_store_choice in region_stores["id_int"].values) else 0,
    )
    chosen_id = int(store_choice_label.split("·")[-1].strip())
    st.session_state.rcp_store_choice = chosen_id

    df_store = df_daily_store[pd.to_numeric(df_daily_store["id"], errors="coerce").astype("Int64") == chosen_id].copy()
    store_name = region_stores.loc[region_stores["id_int"] == chosen_id, "store_display"].iloc[0] if (region_stores["id_int"] == chosen_id).any() else str(chosen_id)

    st.markdown(f"### **{store_name}** · storeID {chosen_id}")

    # Store period totals
    foot_s = float(pd.to_numeric(df_store["footfall"], errors="coerce").dropna().sum()) if "footfall" in df_store.columns else 0.0
    turn_s = float(pd.to_numeric(df_store["turnover"], errors="coerce").dropna().sum()) if "turnover" in df_store.columns else 0.0
    trans_s = float(pd.to_numeric(df_store["transactions"], errors="coerce").dropna().sum()) if "transactions" in df_store.columns else 0.0

    conv_s = (trans_s / foot_s * 100.0) if foot_s > 0 else np.nan
    atv_s = (turn_s / trans_s) if trans_s > 0 else np.nan

    sqm_eff_store = pd.to_numeric(region_stores.loc[region_stores["id_int"] == chosen_id, "sqm_effective"], errors="coerce")
    sqm_eff_store = float(sqm_eff_store.iloc[0]) if (sqm_eff_store is not None and not sqm_eff_store.empty and pd.notna(sqm_eff_store.iloc[0])) else np.nan
    spm2_s = (turn_s / sqm_eff_store) if (pd.notna(sqm_eff_store) and sqm_eff_store > 0) else np.nan
    spv_s = (turn_s / foot_s) if foot_s > 0 else np.nan

    # Store capture proxy avg
    cap_proxy_store = store_capture_proxy.get(int(chosen_id), np.nan)

    # Store explainable SVI vs REGION benchmark
    store_vals = compute_driver_values_from_period(
        footfall=foot_s,
        turnover=turn_s,
        transactions=trans_s,
        sqm_sum=sqm_eff_store,
        capture_pct=cap_proxy_store,
    )
    store_svi, store_avg_ratio, store_bd = compute_svi_explainable(
        vals_a=store_vals,
        vals_b=reg_bench,
        floor=float(lever_floor),
        cap=float(lever_cap),
        weights=None
    )
    store_status, store_status_color = status_from_score(store_svi if pd.notna(store_svi) else 0)

    sk1, sk2, sk3, sk4, sk5 = st.columns([1, 1, 1, 1, 1])
    with sk1:
        kpi_card("Footfall", fmt_int(foot_s), "Store · periode")
    with sk2:
        kpi_card("Omzet", fmt_eur(turn_s), "Store · periode")
    with sk3:
        kpi_card("Conversion", fmt_pct(conv_s), "Store · periode")
    with sk4:
        kpi_card("Sales / m²", fmt_eur(spm2_s), "Store · periode")
    with sk5:
        kpi_card("Store SVI", "-" if pd.isna(store_svi) else f"{store_svi:.0f} / 100", "vs regio benchmark")

    st.markdown(
        f"<div class='muted'>Status: <span style='font-weight:900;color:{store_status_color}'>{store_status}</span> · "
        f"Gemiddelde ratio vs regio ≈ <b>{'' if pd.isna(store_avg_ratio) else f'{store_avg_ratio:.0f}%'}</b></div>",
        unsafe_allow_html=True
    )

    # Store breakdown
    st.markdown('<div class="panel"><div class="panel-title">Store SVI breakdown (vs regio)</div>', unsafe_allow_html=True)
    bd2 = store_bd.copy()
    bd2["ratio_pct"] = pd.to_numeric(bd2["ratio_pct"], errors="coerce")
    bd2 = bd2.dropna(subset=["ratio_pct"])
    if bd2.empty:
        st.info("Geen breakdown beschikbaar voor deze store (missende drivers).")
    else:
        bd2_show = bd2.copy()
        bd2_show["Ratio vs regio"] = bd2_show["ratio_pct"].apply(lambda x: "-" if pd.isna(x) else f"{x:.0f}%")
        bd2_show = bd2_show[["driver", "Ratio vs regio"]].rename(columns={"driver": "Driver"})
        st.dataframe(bd2_show, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------
    # Trends — Store vs Regio vs Company (daily)
    # ----------------------
    st.markdown('<div class="panel"><div class="panel-title">Trends — Store vs Regio vs Company (daily)</div>', unsafe_allow_html=True)

    metric_options = {
        "Conversie (%)": "conversion_rate",
        "SPV (€ / visitor)": "sales_per_visitor",
        "ATV (€)": "sales_per_transaction",
        "Omzet (€)": "turnover",
        "Footfall": "footfall",
        "Sales / m² (€)": "sales_per_sqm",
    }
    metric_label = st.selectbox("Metric", list(metric_options.keys()), index=0, label_visibility="collapsed")
    metric_col = metric_options[metric_label]

    d = df_daily_store.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])

    store_series = df_store[["date", metric_col]].copy() if (not df_store.empty and metric_col in df_store.columns) else pd.DataFrame(columns=["date", metric_col])
    store_series = store_series.rename(columns={metric_col: "value"})
    store_series["series"] = "Store"

    reg = d[d["region"] == region_choice].copy()

    def baseline_daily(df_in: pd.DataFrame, which: str) -> pd.DataFrame:
        tmp = df_in.copy()
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
        tmp = tmp.dropna(subset=["date"])

        if which == "conversion_rate":
            g = tmp.groupby("date", as_index=False).agg(footfall=("footfall", "sum"), transactions=("transactions", "sum"))
            g["value"] = np.where(g["footfall"] > 0, g["transactions"] / g["footfall"] * 100.0, np.nan)
            return g[["date", "value"]]
        if which == "sales_per_visitor":
            g = tmp.groupby("date", as_index=False).agg(footfall=("footfall", "sum"), turnover=("turnover", "sum"))
            g["value"] = np.where(g["footfall"] > 0, g["turnover"] / g["footfall"], np.nan)
            return g[["date", "value"]]
        if which == "sales_per_transaction":
            g = tmp.groupby("date", as_index=False).agg(transactions=("transactions", "sum"), turnover=("turnover", "sum"))
            g["value"] = np.where(g["transactions"] > 0, g["turnover"] / g["transactions"], np.nan)
            return g[["date", "value"]]
        if which == "sales_per_sqm":
            tmp["sqm_effective"] = pd.to_numeric(tmp.get("sqm_effective", np.nan), errors="coerce")
            tmp["turnover"] = pd.to_numeric(tmp.get("turnover", np.nan), errors="coerce")
            g = tmp.groupby(["date", "id"], as_index=False).agg(turnover=("turnover", "sum"), sqm=("sqm_effective", "first"))
            g_active = g[g["turnover"].fillna(0) > 0].copy()
            g2 = g_active.groupby("date", as_index=False).agg(turnover=("turnover", "sum"), sqm=("sqm", "sum"))
            g2["value"] = np.where(g2["sqm"] > 0, g2["turnover"] / g2["sqm"], np.nan)
            return g2[["date", "value"]]

        g = tmp.groupby("date", as_index=False).agg(value=(which, "sum"))
        return g[["date", "value"]]

    region_base = baseline_daily(reg, metric_col)
    region_base["series"] = "Regio"
    company_base = baseline_daily(d, metric_col)
    company_base["series"] = "Company"

    plot_df = pd.concat([store_series, region_base, company_base], ignore_index=True)
    plot_df["value"] = pd.to_numeric(plot_df["value"], errors="coerce")

    chart = (
        alt.Chart(plot_df.dropna(subset=["date"]))
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("date:T", title="Datum"),
            y=alt.Y("value:Q", title=metric_label),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=["Store", "Regio", "Company"], range=[PFM_PURPLE, PFM_LINE, PFM_RED]),
                legend=alt.Legend(title=""),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Datum"),
                alt.Tooltip("series:N", title="Reeks"),
                alt.Tooltip("value:Q", title=metric_label, format=",.2f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("<div class='muted'>Baselines: Regio/Company zijn volume-gewogen per dag. Closed days worden leeg (NaN) getoond i.p.v. 0.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ======================
    # Macro context (CBS/CCI)
    # ======================
    if show_macro:
        st.markdown("## Macro-context (CBS/CCI)")

        macro_start = start_period - timedelta(days=365)
        macro_end = end_period

        st.caption(f"Macro toont: {macro_start} → {macro_end} (1 jaar terug vanaf start van je periode t/m einddatum).")

        region_shop_ids = merged[merged["region"] == region_choice]["id"].dropna().astype(int).unique().tolist()
        cfg = VemcountApiConfig(report_url=REPORT_URL)

        macro_metric_map = {"count_in": "footfall", "turnover": "turnover", "transactions": "transactions"}

        resp_macro = None
        with st.spinner("Macro data ophalen (regio footfall/omzet) ..."):
            try:
                resp_macro = fetch_report(
                    cfg=cfg,
                    shop_ids=region_shop_ids,
                    data_outputs=list(macro_metric_map.keys()),
                    period="date",
                    step="day",
                    source="shops",
                    date_from=macro_start,
                    date_to=macro_end,
                    timeout=120,
                )
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Macro fetch faalde: {e}")

        region_month = pd.DataFrame()
        if resp_macro:
            df_norm_macro = normalize_vemcount_response(resp_macro, kpi_keys=macro_metric_map.keys()).rename(columns=macro_metric_map)
            if df_norm_macro is not None and not df_norm_macro.empty:
                macro_store_key = None
                for cand in ["shop_id", "id", "location_id"]:
                    if cand in df_norm_macro.columns:
                        macro_store_key = cand
                        break

                if macro_store_key:
                    df_macro_daily_store = collapse_to_daily_store(df_norm_macro, store_key_col=macro_store_key)
                    if df_macro_daily_store is not None and not df_macro_daily_store.empty:
                        df_macro_daily_store["date"] = pd.to_datetime(df_macro_daily_store["date"], errors="coerce")
                        df_macro_daily_store = df_macro_daily_store.dropna(subset=["date"])
                        df_macro_daily_store = df_macro_daily_store[
                            (df_macro_daily_store["date"] >= pd.Timestamp(macro_start)) &
                            (df_macro_daily_store["date"] <= pd.Timestamp(macro_end))
                        ].copy()

                        region_month = df_macro_daily_store.copy()
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
            base_val = float(nonzero.iloc[0])
            return (s / base_val) * 100.0

        if not region_month.empty:
            region_month["region_turnover_index"] = index_from_first_nonzero(region_month["region_turnover"])
            region_month["region_footfall_index"] = index_from_first_nonzero(region_month["region_footfall"])

        # CBS retail index
        months_back = int(((macro_end.year - macro_start.year) * 12 + (macro_end.month - macro_start.month)) + 6)
        months_back = max(60, min(240, months_back))

        cbs_retail_month = pd.DataFrame()
        try:
            retail_series = get_retail_index(months_back=months_back) or []
        except Exception:
            retail_series = []

        if retail_series:
            raw = pd.DataFrame(retail_series)
            value_col = next((c for c in ["retail_value", "value", "index", "retail_index"] if c in raw.columns), None)
            period_col = next((c for c in ["period", "periode", "month", "maand"] if c in raw.columns), None)
            if value_col and period_col:
                raw["date"] = pd.to_datetime(
                    raw[period_col].astype(str).str[:4] + "-" + raw[period_col].astype(str).str[-2:] + "-01",
                    errors="coerce",
                )
                raw[value_col] = pd.to_numeric(raw[value_col], errors="coerce")
                raw = raw.dropna(subset=["date", value_col])
                raw = raw[
                    (raw["date"] >= pd.Timestamp(macro_start.replace(day=1))) &
                    (raw["date"] <= pd.Timestamp(macro_end.replace(day=1)))
                ].copy()
                if not raw.empty:
                    cbs_retail_month = raw.groupby("date", as_index=False)[value_col].mean().rename(columns={value_col: "retail_value"})
                    base = cbs_retail_month["retail_value"].replace(0, np.nan).dropna()
                    if not base.empty:
                        b = float(base.iloc[0])
                        cbs_retail_month["cbs_retail_index"] = (cbs_retail_month["retail_value"] / b) * 100.0

        # CCI
        cci_df = pd.DataFrame()
        try:
            cci_series = get_cci_series(months_back=months_back) or []
        except Exception:
            cci_series = []

        if cci_series:
            raw = pd.DataFrame(cci_series)
            value_col = next((c for c in ["cci", "value", "index", "cci_value"] if c in raw.columns), None)
            period_col = next((c for c in ["period", "periode", "month", "maand"] if c in raw.columns), None)
            if value_col and period_col:
                raw["date"] = pd.to_datetime(
                    raw[period_col].astype(str).str[:4] + "-" + raw[period_col].astype(str).str[-2:] + "-01",
                    errors="coerce",
                )
                raw[value_col] = pd.to_numeric(raw[value_col], errors="coerce")
                raw = raw.dropna(subset=["date", value_col])
                raw = raw[
                    (raw["date"] >= pd.Timestamp(macro_start.replace(day=1))) &
                    (raw["date"] <= pd.Timestamp(macro_end.replace(day=1)))
                ].copy()
                if not raw.empty:
                    cci_df = raw.groupby("date", as_index=False)[value_col].mean().rename(columns={value_col: "cci"})
                    base = cci_df["cci"].replace(0, np.nan).dropna()
                    if not base.empty:
                        b = float(base.iloc[0])
                        cci_df["cci_index"] = (cci_df["cci"] / b) * 100.0

        def dual_axis_macro_chart(df: pd.DataFrame, title: str, left_series: list[str], right_series: list[str]):
            st.markdown(f'<div class="panel"><div class="panel-title">{title}</div>', unsafe_allow_html=True)
            if df is None or df.empty:
                st.info("Geen macro-data beschikbaar.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            left = df[df["series"].isin(left_series)].copy()
            right = df[df["series"].isin(right_series)].copy()

            domain = left_series + right_series
            range_ = []
            for s in domain:
                if s in right_series:
                    range_.append(BLACK)
                elif "footfall" in s.lower():
                    range_.append(PFM_PURPLE)
                elif "omzet" in s.lower() or "turnover" in s.lower():
                    range_.append(PFM_RED)
                else:
                    range_.append(PFM_DARK)

            color_scale = alt.Scale(domain=domain, range=range_)

            left_chart = (
                alt.Chart(left)
                .mark_line(point=True, strokeWidth=2)
                .encode(
                    x=alt.X("date:T", title="Maand", axis=alt.Axis(format="%b %Y", labelAngle=0)),
                    y=alt.Y("value:Q", title="Regio index", axis=alt.Axis(format=".0f")),
                    color=alt.Color("series:N", scale=color_scale, title="Legenda"),
                    tooltip=[
                        alt.Tooltip("date:T", title="Maand", format="%b %Y"),
                        alt.Tooltip("series:N", title="Reeks"),
                        alt.Tooltip("value:Q", title="Index", format=".1f"),
                    ],
                )
            )

            right_chart = (
                alt.Chart(right)
                .mark_line(point=True, strokeWidth=2)
                .encode(
                    x=alt.X("date:T", title=""),
                    y=alt.Y("value:Q", title="CBS / CCI index", axis=alt.Axis(orient="right", format=".0f")),
                    color=alt.Color("series:N", scale=color_scale, title="Legenda"),
                    tooltip=[
                        alt.Tooltip("date:T", title="Maand", format="%b %Y"),
                        alt.Tooltip("series:N", title="Reeks"),
                        alt.Tooltip("value:Q", title="Index", format=".1f"),
                    ],
                )
            )

            chart = alt.layer(left_chart, right_chart).resolve_scale(y="independent").properties(height=280)
            st.altair_chart(chart, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        macro_lines = []
        if not region_month.empty:
            a = region_month.rename(columns={"month": "date"})[["date", "region_footfall_index"]].copy()
            a["series"] = "Regio footfall-index"
            a = a.rename(columns={"region_footfall_index": "value"})
            macro_lines.append(a)

            b = region_month.rename(columns={"month": "date"})[["date", "region_turnover_index"]].copy()
            b["series"] = "Regio omzet-index"
            b = b.rename(columns={"region_turnover_index": "value"})
            macro_lines.append(b)

        macro_df_cbs = pd.DataFrame()
        if macro_lines and (not cbs_retail_month.empty and "cbs_retail_index" in cbs_retail_month.columns):
            c = cbs_retail_month[["date", "cbs_retail_index"]].copy()
            c["series"] = "CBS detailhandelindex"
            c = c.rename(columns={"cbs_retail_index": "value"})
            macro_df_cbs = pd.concat(macro_lines + [c], ignore_index=True)

        macro_df_cci = pd.DataFrame()
        if macro_lines and (not cci_df.empty and "cci_index" in cci_df.columns):
            c = cci_df[["date", "cci_index"]].copy()
            c["series"] = "CCI consumentenvertrouwen"
            c = c.rename(columns={"cci_index": "value"})
            macro_df_cci = pd.concat(macro_lines + [c], ignore_index=True)

        m1, m2 = st.columns(2)
        with m1:
            dual_axis_macro_chart(
                df=macro_df_cbs,
                title="CBS detailhandelindex vs Regio (dual axis)",
                left_series=["Regio footfall-index", "Regio omzet-index"],
                right_series=["CBS detailhandelindex"],
            )
        with m2:
            dual_axis_macro_chart(
                df=macro_df_cci,
                title="CCI consumentenvertrouwen vs Regio (dual axis)",
                left_series=["Regio footfall-index", "Regio omzet-index"],
                right_series=["CCI consumentenvertrouwen"],
            )

    # ----------------------
    # Debug
    # ----------------------
    with st.expander("🔧 Debug (v2)"):
        st.write("REPORT_URL:", REPORT_URL)
        st.write("Company:", company_id)
        st.write("Region:", region_choice)
        st.write("Period:", start_period, "→", end_period)
        st.write("SVI floor/cap:", lever_floor, lever_cap)
        st.write("Reg bench:", reg_bench)
        st.write("Company vals:", comp_vals)
        st.write("df_daily_store cols:", df_daily_store.columns.tolist())
        st.write("Example store rows:", df_store.head(10))

if __name__ == "__main__":
    main()
