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
        out["avg_basket_size"] = np.where(out["transactions"] > 0, out["turnover"] / out["transactions"], np.nan)
        out["sales_per_transaction"] = out["avg_basket_size"]

    return out

# ----------------------
# Weekly (region) for capture chart (as in v1)
# ----------------------
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

# ----------------------
# sqm enrichment from report response
# ----------------------
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

# ----------------------
# SVI v2 helpers
# ----------------------
def _ratio_to_score_0_100(ratio: float, floor: float = 70.0, cap: float = 130.0) -> float:
    if pd.isna(ratio):
        return np.nan
    r = float(np.clip(ratio, floor, cap))
    return (r - floor) / (cap - floor) * 100.0

def nice_component_name(key: str) -> str:
    mapping = {
        "sales_per_sqm": "Sales / m²",
        "capture_rate": "Capture rate",
        "conversion_rate": "Conversion",
        "avg_basket_size": "Avg basket (ATV)",
        "sales_per_transaction": "Sales / transaction (proxy)",
    }
    return mapping.get(key, key)

# ----------------------
# Weighted daily baselines: region & company
# ----------------------
def build_daily_baselines(df_daily_store: pd.DataFrame, region_choice: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      region_daily: date, region_* KPIs (weighted)
      company_daily: date, company_* KPIs (weighted)
    """
    if df_daily_store is None or df_daily_store.empty:
        return pd.DataFrame(), pd.DataFrame()

    d = df_daily_store.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])

    # region subset
    reg = d[d["region"] == region_choice].copy()
    cmp = d.copy()

    def _agg(df_: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if df_.empty:
            return pd.DataFrame()

        g = df_.groupby("date", as_index=False).agg(
            footfall=("footfall", "sum"),
            turnover=("turnover", "sum"),
            transactions=("transactions", "sum"),
            sqm=("sqm_effective", "sum"),  # careful: this is NOT used for weighting, only for sales/m² baseline if needed
        )
        g[f"{prefix}_conversion_rate"] = np.where(g["footfall"] > 0, g["transactions"] / g["footfall"] * 100.0, np.nan)
        g[f"{prefix}_sales_per_visitor"] = np.where(g["footfall"] > 0, g["turnover"] / g["footfall"], np.nan)
        g[f"{prefix}_avg_basket_size"] = np.where(g["transactions"] > 0, g["turnover"] / g["transactions"], np.nan)
        # baseline sales/m² is tricky (sqm should be “sum of unique store sqm”, not daily sum). We'll keep it simple:
        g[f"{prefix}_sales_per_sqm"] = np.nan

        keep = ["date", f"{prefix}_conversion_rate", f"{prefix}_sales_per_visitor", f"{prefix}_avg_basket_size", f"{prefix}_sales_per_sqm"]
        return g[keep]

    return _agg(reg, "region"), _agg(cmp, "company")

# ----------------------
# Store drilldown helpers
# ----------------------
def lever_scan_scores(store_period: dict, region_period: dict, company_period: dict) -> pd.DataFrame:
    """
    Builds explainable lever scan table (scores + ratios).
    """
    # metrics we want to compare
    metrics = [
        ("conversion_rate", "Conversie (%)"),
        ("sales_per_visitor", "SPV (€ / visitor)"),
        ("avg_basket_size", "ATV (€)"),
        ("sales_per_sqm", "Sales / m² (€)"),
    ]

    rows = []
    for k, label in metrics:
        v_store = store_period.get(k, np.nan)
        v_reg = region_period.get(k, np.nan)
        v_cmp = company_period.get(k, np.nan)

        ratio_reg = (v_store / v_reg * 100.0) if (pd.notna(v_store) and pd.notna(v_reg) and v_reg != 0) else np.nan
        ratio_cmp = (v_store / v_cmp * 100.0) if (pd.notna(v_store) and pd.notna(v_cmp) and v_cmp != 0) else np.nan

        score_reg = _ratio_to_score_0_100(ratio_reg)
        score_cmp = _ratio_to_score_0_100(ratio_cmp)

        rows.append({
            "metric_key": k,
            "metric": label,
            "store_value": v_store,
            "ratio_vs_region": ratio_reg,
            "score_vs_region": score_reg,
            "ratio_vs_company": ratio_cmp,
            "score_vs_company": score_cmp,
        })

    return pd.DataFrame(rows)

# ----------------------
# MAIN
# ----------------------
def main():
    st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)

    # --- session state init (prevents “everything disappears” on store select)
    if "has_data" not in st.session_state:
        st.session_state["has_data"] = False
    if "data_bundle" not in st.session_state:
        st.session_state["data_bundle"] = {}

    header_left, header_right = st.columns([2.2, 1.8])

    with header_left:
        st.markdown(
            f"""
            <div class="pfm-header">
              <div>
                <div class="pfm-title">PFM Region Performance Copilot <span class="pill">v2</span></div>
                <div class="pfm-sub">Regio-level upgrade: explainable SVI + opportunities + store drilldown</div>
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
    if merged.empty:
        st.warning("Er zijn geen winkels met een regio-mapping voor deze retailer.")
        return

    # store display name
    if "store_label" in merged.columns and merged["store_label"].notna().any():
        merged["store_display"] = merged["store_label"]
    else:
        merged["store_display"] = merged["name"] if "name" in merged.columns else merged["id"].astype(str)

    # Controls
    available_regions = sorted(merged["region"].dropna().unique().tolist())
    top_controls = st.columns([1.2, 1.2, 1.2, 1.2, 1.2])
    with top_controls[0]:
        region_choice = st.selectbox("Regio", available_regions)
    with top_controls[1]:
        show_macro = st.toggle("Toon macro (CBS/CCI)", value=True)
    with top_controls[2]:
        show_quadrant = st.toggle("Toon quadrant", value=True)
    with top_controls[3]:
        show_drilldown = st.toggle("Toon store drilldown", value=True)
    with top_controls[4]:
        run_btn = st.button("Analyseer", type="primary")

    # use PeriodDef
    start_period = periods[period_choice].start
    end_period = periods[period_choice].end

    region_shops = merged[merged["region"] == region_choice].copy()
    region_shop_ids = region_shops["id"].dropna().astype(int).unique().tolist()
    if not region_shop_ids:
        st.warning(f"Geen winkels gevonden voor regio '{region_choice}'.")
        return

    all_shop_ids = merged["id"].dropna().astype(int).unique().tolist()

    # Always fetch ALL for correct company baseline
    fetch_ids = all_shop_ids

    metric_map = {
        "count_in": "footfall",
        "turnover": "turnover",
        "transactions": "transactions",  # <-- jouw vraag: dit is de key
        "conversion_rate": "conversion_rate",
        "sales_per_visitor": "sales_per_visitor",
        "avg_basket_size": "avg_basket_size",
        "sales_per_sqm": "sales_per_sqm",
        "sales_per_transaction": "sales_per_transaction",
        # (optioneel) als je sqm uit API wilt proberen:
        # "sq_meter": "sq_meter",
    }

    cfg = VemcountApiConfig(report_url=REPORT_URL)

    params_preview = build_report_params(
        shop_ids=fetch_ids,
        data_outputs=list(metric_map.keys()),
        period="date",
        step="day",
        source="shops",
        date_from=start_period,
        date_to=end_period,
    )

    # -------------------------
    # Fetch only on Analyseer
    # -------------------------
    if run_btn:
        with st.spinner("Data ophalen via FastAPI..."):
            try:
                resp = fetch_report(
                    cfg=cfg,
                    shop_ids=fetch_ids,
                    data_outputs=list(metric_map.keys()),
                    period="date",
                    step="day",
                    source="shops",
                    date_from=start_period,
                    date_to=end_period,
                    timeout=120,
                )
            except requests.exceptions.RequestException as e:
                st.error(f"❌ /get-report faalde: {e}")
                with st.expander("🔧 Debug request (params)"):
                    st.write("REPORT_URL:", REPORT_URL)
                    st.write("Params:", params_preview)
                return

        df_norm = normalize_vemcount_response(resp, kpi_keys=metric_map.keys()).rename(columns=metric_map)

        if df_norm is None or df_norm.empty:
            st.warning("Geen data ontvangen voor de gekozen selectie.")
            with st.expander("🔧 Debug response keys"):
                st.write("Params:", params_preview)
            return

        store_key_col = None
        for cand in ["shop_id", "id", "location_id"]:
            if cand in df_norm.columns:
                store_key_col = cand
                break
        if store_key_col is None:
            st.error("Geen store-id kolom gevonden in de response (shop_id/id/location_id).")
            with st.expander("🔧 Debug df_norm columns"):
                st.write(df_norm.columns.tolist())
            return

        # enrich sqm
        merged2 = enrich_merged_with_sqm_from_df_norm(merged, df_norm, store_key_col=store_key_col)

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

        join_cols = ["id", "store_display", "region", "sqm_effective"]
        if "store_type" in merged2.columns:
            join_cols.append("store_type")

        df_daily_store = df_daily_store.merge(
            merged2[join_cols].drop_duplicates(),
            left_on=store_key_col,
            right_on="id",
            how="left",
        )

        # ensure region exists (belt & suspenders)
        df_daily_store = ensure_region_column(df_daily_store, merged2, store_key_col)

        # --- Force sales_per_sqm if sqm exists (fix “leeg”)
        if "sales_per_sqm" not in df_daily_store.columns:
            df_daily_store["sales_per_sqm"] = np.nan
        df_daily_store["sales_per_sqm"] = np.where(
            pd.to_numeric(df_daily_store.get("sqm_effective", np.nan), errors="coerce") > 0,
            pd.to_numeric(df_daily_store.get("turnover", np.nan), errors="coerce") / pd.to_numeric(df_daily_store.get("sqm_effective", np.nan), errors="coerce"),
            df_daily_store["sales_per_sqm"],
        )

        # store in session_state
        st.session_state["has_data"] = True
        st.session_state["data_bundle"] = {
            "df_daily_store": df_daily_store,
            "merged": merged2,
            "store_key_col": store_key_col,
            "period_choice": period_choice,
            "start_period": start_period,
            "end_period": end_period,
            "selected_client": selected_client,
            "params_preview": params_preview,
            "region_choice": region_choice,
        }

    # If no data yet, show instruction
    if not st.session_state["has_data"]:
        st.info("Selecteer retailer/regio/periode en klik op **Analyseer**.")
        return

    # -------------------------
    # Use cached data
    # -------------------------
    bundle = st.session_state["data_bundle"]
    df_daily_store = bundle["df_daily_store"]
    merged2 = bundle["merged"]
    store_key_col = bundle["store_key_col"]
    start_period = bundle["start_period"]
    end_period = bundle["end_period"]
    selected_client = bundle["selected_client"]
    region_choice = bundle["region_choice"]

    df_region_daily = df_daily_store[df_daily_store["region"] == region_choice].copy()

    # ----------------------
    # Region KPI headline
    # ----------------------
    foot_total = float(df_region_daily["footfall"].sum()) if "footfall" in df_region_daily.columns else 0.0
    turn_total = float(df_region_daily["turnover"].sum()) if "turnover" in df_region_daily.columns else 0.0
    trans_total = float(df_region_daily["transactions"].sum()) if "transactions" in df_region_daily.columns else 0.0

    conv = (trans_total / foot_total * 100.0) if foot_total > 0 else np.nan
    atv = (turn_total / trans_total) if trans_total > 0 else np.nan

    # Capture weekly
    region_weekly = aggregate_weekly_region(df_region_daily)
    pathzz_weekly = fetch_region_street_traffic(region=region_choice, start_date=start_period, end_date=end_period)

    capture_weekly = pd.DataFrame()
    avg_capture = np.nan

    if not region_weekly.empty:
        region_weekly = region_weekly.copy()
        region_weekly["week_start"] = pd.to_datetime(region_weekly["week_start"], errors="coerce")
        region_weekly = region_weekly.dropna(subset=["week_start"])
        region_weekly = (
            region_weekly.groupby("week_start", as_index=False)
            .agg(
                footfall=("footfall", "sum"),
                turnover=("turnover", "sum") if "turnover" in region_weekly.columns else ("footfall", "sum"),
                transactions=("transactions", "sum") if "transactions" in region_weekly.columns else ("footfall", "sum"),
            )
        )

    if not pathzz_weekly.empty:
        pathzz_weekly = pathzz_weekly.copy()
        pathzz_weekly["week_start"] = pd.to_datetime(pathzz_weekly["week_start"], errors="coerce")
        pathzz_weekly = pathzz_weekly.dropna(subset=["week_start"])
        pathzz_weekly = (
            pathzz_weekly.groupby("week_start", as_index=False)
               .agg(street_footfall=("street_footfall", "mean"))
        )

    if not region_weekly.empty and not pathzz_weekly.empty:
        capture_weekly = pd.merge(region_weekly, pathzz_weekly, on="week_start", how="inner")

    if not capture_weekly.empty:
        capture_weekly["capture_rate"] = np.where(
            capture_weekly["street_footfall"] > 0,
            capture_weekly["footfall"] / capture_weekly["street_footfall"] * 100.0,
            np.nan,
        )
        avg_capture = float(pd.to_numeric(capture_weekly["capture_rate"], errors="coerce").dropna().mean())

    # ----------------------
    # Header
    # ----------------------
    st.markdown(f"## {selected_client['brand']} — Regio **{region_choice}** · {start_period} → {end_period}")

    # KPI row
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
        "<div class='muted'>Company baseline = alle shops in company (zelfde periode). Regio baseline = shops binnen de gekozen regio.</div>",
        unsafe_allow_html=True
    )

    # ----------------------
    # Store drilldown
    # ----------------------
    if show_drilldown:
        st.markdown("## Store drilldown")

        # store selector that does NOT force refetch
        region_store_options = (
            merged2[merged2["region"] == region_choice][["id", "store_display"]]
            .dropna()
            .drop_duplicates()
            .sort_values("store_display")
        )
        store_labels = region_store_options["store_display"].tolist()
        default_idx = 0

        store_choice = st.selectbox("Winkel", store_labels, index=default_idx, key="drill_store_choice")

        store_id = int(region_store_options.loc[region_store_options["store_display"] == store_choice, "id"].iloc[0])

        ds = df_daily_store.copy()
        ds["date"] = pd.to_datetime(ds["date"], errors="coerce")
        ds = ds.dropna(subset=["date"])

        store_daily = ds[ds["id"] == store_id].copy()
        region_daily = ds[ds["region"] == region_choice].copy()
        company_daily = ds.copy()

        # daily baselines (weighted)
        region_base, company_base = build_daily_baselines(ds, region_choice=region_choice)

        # store KPIs (period totals)
        def _period_kpis(df_: pd.DataFrame) -> dict:
            foot = float(pd.to_numeric(df_.get("footfall", 0), errors="coerce").fillna(0).sum())
            turn = float(pd.to_numeric(df_.get("turnover", 0), errors="coerce").fillna(0).sum())
            trans = float(pd.to_numeric(df_.get("transactions", 0), errors="coerce").fillna(0).sum())
            sqm = pd.to_numeric(df_.get("sqm_effective", np.nan), errors="coerce")
            sqm_eff = float(sqm.dropna().drop_duplicates().iloc[0]) if sqm.dropna().any() else np.nan

            out = {
                "footfall": foot,
                "turnover": turn,
                "transactions": trans,
                "conversion_rate": (trans / foot * 100.0) if foot > 0 else np.nan,
                "avg_basket_size": (turn / trans) if trans > 0 else np.nan,
                "sales_per_visitor": (turn / foot) if foot > 0 else np.nan,
                "sales_per_sqm": (turn / sqm_eff) if (pd.notna(sqm_eff) and sqm_eff > 0) else np.nan,
            }
            return out

        store_period = _period_kpis(store_daily)
        region_period = _period_kpis(region_daily)
        company_period = _period_kpis(company_daily)

        # headline
        st.markdown(f"### {store_choice} · storeID {store_id}")

        cA, cB, cC, cD = st.columns(4)
        with cA:
            kpi_card("Footfall", fmt_int(store_period["footfall"]), "Store · periode")
        with cB:
            kpi_card("Omzet", fmt_eur(store_period["turnover"]), "Store · periode")
        with cC:
            kpi_card("Conversion", fmt_pct(store_period["conversion_rate"]), "Store · periode")
        with cD:
            kpi_card("Sales / m²", fmt_eur(store_period["sales_per_sqm"]), "Store · periode (sqm_effective)")

        # Trends — Store vs Regio vs Company (daily)
        st.markdown('<div class="panel"><div class="panel-title">Trends — Store vs Regio vs Company (daily)</div>', unsafe_allow_html=True)

        if store_daily.empty:
            st.info("Geen daily data voor deze store in deze periode.")
        else:
            plot = store_daily[["date", "conversion_rate", "sales_per_visitor", "avg_basket_size"]].copy()
            plot = plot.merge(region_base, on="date", how="left").merge(company_base, on="date", how="left")

            # choose metric
            metric_choice = st.selectbox(
                "Metric",
                ["conversion_rate", "sales_per_visitor", "avg_basket_size"],
                format_func=lambda x: {
                    "conversion_rate": "Conversie (%)",
                    "sales_per_visitor": "SPV (€ / visitor)",
                    "avg_basket_size": "ATV (€)"
                }[x],
                key="drill_metric_choice",
            )

            # build long form
            col_map = {
                "conversion_rate": ("conversion_rate", "region_conversion_rate", "company_conversion_rate"),
                "sales_per_visitor": ("sales_per_visitor", "region_sales_per_visitor", "company_sales_per_visitor"),
                "avg_basket_size": ("avg_basket_size", "region_avg_basket_size", "company_avg_basket_size"),
            }
            a, b, c = col_map[metric_choice]

            long = pd.DataFrame({
                "date": pd.to_datetime(plot["date"]),
                "Store": pd.to_numeric(plot[a], errors="coerce"),
                "Regio": pd.to_numeric(plot[b], errors="coerce"),
                "Company": pd.to_numeric(plot[c], errors="coerce"),
            }).melt(id_vars=["date"], var_name="series", value_name="value").dropna(subset=["value"])

            chart = (
                alt.Chart(long)
                .mark_line(point=True, strokeWidth=2)
                .encode(
                    x=alt.X("date:T", title="Datum"),
                    y=alt.Y("value:Q", title="", axis=alt.Axis(format=".2f")),
                    color=alt.Color(
                        "series:N",
                        scale=alt.Scale(domain=["Store", "Regio", "Company"], range=[PFM_PURPLE, PFM_LINE, PFM_RED]),
                        legend=alt.Legend(title=None),
                    ),
                    tooltip=[
                        alt.Tooltip("date:T", title="Datum"),
                        alt.Tooltip("series:N", title="Reeks"),
                        alt.Tooltip("value:Q", title="Waarde", format=".2f"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)

            st.caption("Baselines: Regio/Company zijn *weighted* (som-gebaseerd) per dag, dus eerlijk bij verschillende winkelgroottes.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Best / worst days
        st.markdown('<div class="panel"><div class="panel-title">Best / worst days (store)</div>', unsafe_allow_html=True)

        if store_daily.empty:
            st.info("Geen dagen om te ranken.")
        else:
            rank_metric = st.selectbox(
                "Rank op",
                ["turnover", "sales_per_visitor", "conversion_rate"],
                format_func=lambda x: {
                    "turnover": "Omzet",
                    "sales_per_visitor": "SPV",
                    "conversion_rate": "Conversie"
                }[x],
                key="rank_metric",
            )

            cols_show = ["date", "turnover", "footfall", "transactions", "conversion_rate", "sales_per_visitor", "avg_basket_size", "sales_per_sqm"]
            tmp = store_daily.copy()
            for c in cols_show:
                if c in tmp.columns:
                    tmp[c] = pd.to_numeric(tmp[c], errors="coerce") if c != "date" else tmp[c]
            tmp = tmp.sort_values(rank_metric, ascending=False)

            left, right = st.columns(2)
            with left:
                st.markdown("**Top 5 days**")
                st.dataframe(tmp[cols_show].head(5), use_container_width=True)
            with right:
                st.markdown("**Bottom 5 days**")
                st.dataframe(tmp[cols_show].tail(5).sort_values(rank_metric, ascending=True), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Lever scan
        st.markdown('<div class="panel"><div class="panel-title">Lever scan — waar zit het gat?</div>', unsafe_allow_html=True)

        lev = lever_scan_scores(store_period, region_period, company_period)

        if lev.empty:
            st.info("Geen lever scan mogelijk.")
        else:
            view_mode = st.radio("Vergelijking", ["vs Regio", "vs Company"], horizontal=True, key="lever_view")
            score_col = "score_vs_region" if view_mode == "vs Regio" else "score_vs_company"
            ratio_col = "ratio_vs_region" if view_mode == "vs Regio" else "ratio_vs_company"

            b = lev.copy()
            b["score"] = pd.to_numeric(b[score_col], errors="coerce")
            b["ratio"] = pd.to_numeric(b[ratio_col], errors="coerce")

            # IMPORTANT: do NOT fill NaN scores with 100 (that caused “3x 100” confusion)
            b2 = b.dropna(subset=["score"]).copy()
            if b2.empty:
                st.info("Benchmarks ontbreken voor deze vergelijking (scores zijn NaN).")
            else:
                b2 = b2.sort_values("score", ascending=True)

                bar = (
                    alt.Chart(b2)
                    .mark_bar(cornerRadiusEnd=4, color=PFM_AMBER)
                    .encode(
                        x=alt.X("score:Q", title=f"Score {view_mode} (0–100)", scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y("metric:N", sort="-x", title=None),
                        tooltip=[
                            alt.Tooltip("metric:N", title="Metric"),
                            alt.Tooltip("score:Q", title="Score", format=".0f"),
                            alt.Tooltip("ratio:Q", title="Ratio vs benchmark (%)", format=".1f"),
                            alt.Tooltip("store_value:Q", title="Store value", format=".2f"),
                        ],
                    )
                    .properties(height=220)
                )
                st.altair_chart(bar, use_container_width=True)
                st.caption("Lager = grotere hefboom. Scores zijn ratio-gebaseerd en geclipt (70–130% → 0–100).")

        st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------
    # Debug
    # ----------------------
    with st.expander("🔧 Debug (v2)"):
        st.write("Retailer:", selected_client)
        st.write("Regio:", region_choice)
        st.write("Periode:", start_period, "→", end_period, f"({period_choice})")
        st.write("Fetch ids used:", len(all_shop_ids), "(ALL for benchmark correctness)")
        st.write("Store key col:", store_key_col)
        st.write("REPORT_URL:", REPORT_URL)
        st.write("Params used:", bundle.get("params_preview"))

        st.write("Merged columns:", merged2.columns.tolist())
        st.write("sqm_effective non-null:", int(pd.to_numeric(merged2["sqm_effective"], errors="coerce").notna().sum()) if "sqm_effective" in merged2.columns else 0)

        st.write("df_daily_store columns:", df_daily_store.columns.tolist())
        st.write("df_daily_store head:", df_daily_store.head())

if __name__ == "__main__":
    main()
