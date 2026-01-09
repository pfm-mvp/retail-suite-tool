# pages/06C_Region_Copilot_V2.py

import numpy as np
import pandas as pd
import requests
import os
import re
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
    page_title="PFM Region Copilot v2",
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

OTHER_REGION_PURPLE = "#C4B5FD"
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

def norm_key(x: str) -> str:
    return str(x).strip().lower() if x is not None else ""

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

# ----------------------
# Pathzz store-level capture (FIXED)
# ----------------------
@st.cache_data(ttl=600)
def load_pathzz_weekly_store(csv_path: str, _mtime: float) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, sep=";", dtype=str, engine="python")
    except Exception:
        return pd.DataFrame(columns=["region", "week", "week_start", "visits", "shop_id"])

    if df is None or df.empty:
        return pd.DataFrame(columns=["region", "week", "week_start", "visits", "shop_id"])

    df = df.rename(columns={"Region": "region", "Week": "week", "Visits": "visits"}).copy()

    for c in ["region", "week", "visits"]:
        if c not in df.columns:
            return pd.DataFrame(columns=["region", "week", "week_start", "visits", "shop_id"])

    if "shop_id" not in df.columns and df.shape[1] >= 4:
        df["shop_id"] = df.iloc[:, -1]

    if "shop_id" not in df.columns:
        return pd.DataFrame(columns=["region", "week", "week_start", "visits", "shop_id"])

    df["region"] = df["region"].astype(str).str.strip()
    df["visits"] = df["visits"].astype(str).str.strip().replace("", np.nan)
    df["shop_id"] = pd.to_numeric(df["shop_id"], errors="coerce").astype("Int64")

    df = df.dropna(subset=["visits", "shop_id"])
    if df.empty:
        return pd.DataFrame(columns=["region", "week", "week_start", "visits", "shop_id"])

    df["visits"] = (
        df["visits"]
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

    return df[["region", "week", "week_start", "visits", "shop_id"]].reset_index(drop=True)

def filter_pathzz_for_period(df_pathzz: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    expected_cols = ["region", "week_start", "visits", "shop_id"]
    if df_pathzz is None or df_pathzz.empty:
        return pd.DataFrame(columns=expected_cols)

    for c in expected_cols:
        if c not in df_pathzz.columns:
            return pd.DataFrame(columns=expected_cols)

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    tmp = df_pathzz.copy()
    tmp["week_start"] = pd.to_datetime(tmp["week_start"], errors="coerce")
    tmp = tmp.dropna(subset=["week_start"])

    out = tmp[(tmp["week_start"] >= start) & (tmp["week_start"] <= end)].copy()
    return out if not out.empty else pd.DataFrame(columns=expected_cols)

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
    if pd.isna(ratio_pct):
        return np.nan
    r = float(np.clip(ratio_pct, floor, cap))
    return (r - floor) / (cap - floor) * 100.0

# ----------------------
# SVI: composite + explainable
# ----------------------
SVI_DRIVERS = [
    ("sales_per_visitor", "SPV (€/visitor)"),
    ("sales_per_sqm", "Sales / m² (€)"),
    ("capture_rate", "Capture (location-driven) (%)"),
    ("conversion_rate", "Conversion (%)"),
    ("sales_per_transaction", "ATV (€)"),
]

# Default weights (baseline)
BASE_SVI_WEIGHTS = {
    "sales_per_visitor": 1.0,
    "sales_per_sqm": 1.0,
    "conversion_rate": 1.0,
    "sales_per_transaction": 0.8,
    "capture_rate": 0.4,   # baseline
}

def get_svi_weights_for_store_type(store_type: str) -> dict:
    w = dict(BASE_SVI_WEIGHTS)
    s = norm_key(store_type)

    if ("high" in s and "street" in s) or ("city" in s) or ("downtown" in s) or ("centre" in s) or ("center" in s and "city" in s):
        w["capture_rate"] = 0.7
    elif ("retail" in s and "park" in s) or ("park" in s):
        w["capture_rate"] = 0.2
    elif ("shopping" in s and "center" in s) or ("shopping" in s and "centre" in s) or ("mall" in s) or ("center" in s) or ("centre" in s):
        w["capture_rate"] = 0.4
    else:
        w["capture_rate"] = BASE_SVI_WEIGHTS["capture_rate"]

    return w

def compute_driver_values_from_period(footfall, turnover, transactions, sqm_sum, capture_pct):
    spv = safe_div(turnover, footfall)
    spsqm = safe_div(turnover, sqm_sum)
    cr = safe_div(transactions, footfall) * 100.0 if (pd.notna(transactions) and pd.notna(footfall) and float(footfall) != 0.0) else np.nan
    atv = safe_div(turnover, transactions)
    cap = capture_pct
    return {
        "sales_per_visitor": spv,
        "sales_per_sqm": spsqm,
        "capture_rate": cap,
        "conversion_rate": cr,
        "sales_per_transaction": atv,
    }

def compute_svi_explainable(vals_a: dict, vals_b: dict, floor: float, cap: float, weights=None):
    if weights is None:
        weights = {k: float(BASE_SVI_WEIGHTS.get(k, 1.0)) for k, _ in SVI_DRIVERS}

    rows = []
    for key, label in SVI_DRIVERS:
        va = vals_a.get(key, np.nan)
        vb = vals_b.get(key, np.nan)

        ratio = np.nan
        if pd.notna(va) and pd.notna(vb) and float(vb) != 0.0:
            ratio = (float(va) / float(vb)) * 100.0

        score = ratio_to_score_0_100(ratio, floor=float(floor), cap=float(cap))
        w = float(weights.get(key, 1.0))

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

    usable["w"] = usable["weight"].astype(float)
    wsum = float(usable["w"].sum()) if float(usable["w"].sum()) > 0 else float(len(usable))
    avg_ratio = float((usable["ratio_pct"] * usable["w"]).sum() / wsum)
    svi = ratio_to_score_0_100(avg_ratio, floor=float(floor), cap=float(cap))
    return float(svi), float(avg_ratio), bd.drop(columns=["include"])

def style_heatmap_ratio(val):
    try:
        if pd.isna(val):
            return ""
        v = float(val)
        if v >= 110:
            return "background-color: #ecfdf5; color:#065f46; font-weight:800;"
        if v >= 95:
            return "background-color: #fffbeb; color:#92400e; font-weight:800;"
        return "background-color: #fff1f2; color:#9f1239; font-weight:800;"
    except Exception:
        return ""

# ----------------------
# Macro charts (FIXED)
# ----------------------
def plot_macro_panel(df_region_daily: pd.DataFrame, macro_start, macro_end):
    st.markdown(
        '<div class="panel"><div class="panel-title">Macro context — Consumer Confidence & Retail Index</div>',
        unsafe_allow_html=True
    )

    ms = pd.to_datetime(macro_start)
    me = pd.to_datetime(macro_end)

    if df_region_daily is None or df_region_daily.empty:
        st.info("No region data available for macro context.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    dd = df_region_daily.copy()
    dd["date"] = pd.to_datetime(dd["date"], errors="coerce")
    dd = dd.dropna(subset=["date"])
    dd = dd[(dd["date"] >= ms) & (dd["date"] <= me)].copy()

    if dd.empty:
        st.info("No region index series for this macro window (no footfall/turnover data).")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    dd["footfall"] = pd.to_numeric(dd.get("footfall", np.nan), errors="coerce")
    dd["turnover"] = pd.to_numeric(dd.get("turnover", np.nan), errors="coerce")

    region_m = (
        dd.set_index("date")[["footfall", "turnover"]]
        .resample("MS")
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"date": "month"})
    )

    def _idx(series: pd.Series, k_base: int = 3) -> pd.Series:
        if series is None or len(series) == 0:
            return pd.Series([], dtype=float)

        s = pd.to_numeric(series, errors="coerce").copy()
        valid = s[(s.notna()) & (s > 0)]
        if valid.empty:
            return pd.Series([np.nan] * len(s), index=s.index)

        base = float(valid.iloc[:k_base].mean())
        if pd.isna(base) or base <= 0:
            return pd.Series([np.nan] * len(s), index=s.index)

        return (s / base) * 100.0

    region_m["Region footfall-index"] = _idx(region_m["footfall"])
    region_m["Region omzet-index"] = _idx(region_m["turnover"])

    region_m = region_m.dropna(subset=["month"]).copy()
    if region_m.empty:
        st.info("No region index series for this macro window (after monthly aggregation).")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ---- CBS macro series: request more history to ensure 2023 is included ----
    months_needed = int(((me - ms).days / 30.5) + 2)
    months_back = max(36, months_needed + 6)

    try:
        cci_raw = get_cci_series(months_back=months_back)
        ridx_raw = get_retail_index(months_back=months_back)
    except Exception as e:
        st.info(f"Macro data not available right now: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    def _parse_period_to_monthstart(s: str):
        if s is None or str(s).strip() == "":
            return pd.NaT
        s = str(s).strip()
        s = s.replace("MM", "M")

        m = re.match(r"^(\d{4})M(\d{2})$", s)
        if m:
            return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)

        m = re.match(r"^(\d{4})(\d{2})$", s)
        if m:
            return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)

        m = re.match(r"^(\d{4})[-/](\d{2})$", s)
        if m:
            return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)

        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return pd.NaT
        return pd.Timestamp(dt.year, dt.month, 1)

    cci_df = pd.DataFrame(cci_raw) if isinstance(cci_raw, list) else pd.DataFrame()
    if not cci_df.empty and {"period", "cci"}.issubset(set(cci_df.columns)):
        cci_df["month"] = cci_df["period"].apply(_parse_period_to_monthstart)
        cci_df["value"] = pd.to_numeric(cci_df["cci"], errors="coerce")
        cci_df = cci_df.dropna(subset=["month", "value"])[["month", "value"]].sort_values("month")
    else:
        cci_df = pd.DataFrame(columns=["month", "value"])

    ridx_df = pd.DataFrame(ridx_raw) if isinstance(ridx_raw, list) else pd.DataFrame()
    if not ridx_df.empty and {"period", "retail_value"}.issubset(set(ridx_df.columns)):
        ridx_df["month"] = ridx_df["period"].apply(_parse_period_to_monthstart)
        ridx_df["value"] = pd.to_numeric(ridx_df["retail_value"], errors="coerce")
        ridx_df = ridx_df.dropna(subset=["month", "value"])[["month", "value"]].sort_values("month")
    else:
        ridx_df = pd.DataFrame(columns=["month", "value"])

    # filter to macro window
    cci_df = cci_df[(cci_df["month"] >= ms) & (cci_df["month"] <= me)].copy()
    ridx_df = ridx_df[(ridx_df["month"] >= ms) & (ridx_df["month"] <= me)].copy()

    # ---- charts ----
    x_enc = alt.X(
        "month:T",
        title=None,
        scale=alt.Scale(domain=[ms, me]),
        axis=alt.Axis(format="%b %Y", labelAngle=-35, labelOverlap=False, labelPadding=8)
    )

    reg_long = region_m.melt(
        id_vars=["month"],
        value_vars=["Region footfall-index", "Region omzet-index"],
        var_name="series",
        value_name="idx",
    )

    region_lines = (
        alt.Chart(reg_long)
        .mark_line(point=True)
        .encode(
            x=x_enc,
            y=alt.Y("idx:Q", title="Regio-index (100 = start)"),
            color=alt.Color("series:N", legend=alt.Legend(title="", orient="right")),
            tooltip=[
                alt.Tooltip("month:T", title="Maand"),
                alt.Tooltip("series:N", title="Reeks"),
                alt.Tooltip("idx:Q", title="Index", format=".1f"),
            ],
        )
    )

    def macro_line(df_macro: pd.DataFrame, label: str, dash: bool):
        if df_macro is None or df_macro.empty:
            return None

        dfm = df_macro.copy()
        dfm["series"] = label

        return (
            alt.Chart(dfm)
            .mark_line(
                point=True,
                strokeWidth=2,
                strokeDash=[6, 4] if dash else [1, 0],
            )
            .encode(
                x=x_enc,
                y=alt.Y("value:Q", title=label, axis=alt.Axis(orient="right")),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(domain=[label], range=[BLACK]),
                    legend=alt.Legend(title="", orient="right"),
                ),
                tooltip=[
                    alt.Tooltip("month:T", title="Maand"),
                    alt.Tooltip("value:Q", title=label, format=".1f"),
                ],
            )
        )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**CBS detailhandelindex vs Regio**")
        mline = macro_line(ridx_df, "CBS retail index", dash=True)
        chart = alt.layer(region_lines, mline) if mline is not None else region_lines
        st.altair_chart(
            chart.resolve_scale(y="independent", color="independent")
                 .properties(height=280)
                 .configure_view(strokeWidth=0),
            use_container_width=True,
        )

    with c2:
        st.markdown("**Consumentenvertrouwen (CCI) vs Regio**")
        mline = macro_line(cci_df, "CCI", dash=True)
        chart = alt.layer(region_lines, mline) if mline is not None else region_lines
        st.altair_chart(
            chart.resolve_scale(y="independent", color="independent")
                 .properties(height=280)
                 .configure_view(strokeWidth=0),
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# MAIN
# ----------------------
def main():
    # Session state to prevent wipe on drilldown
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
                <div class="pfm-sub">Region-level: explainable SVI + heatmap scanning + value upside + drilldown + macro context</div>
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
                "Period",
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
        st.error(f"Error fetching stores from FastAPI: {e}")
        return

    if locations_df.empty:
        st.error("No stores found for this retailer.")
        return

    region_map = load_region_mapping()
    if region_map.empty:
        st.error("No valid data/regions.csv found (min required: shop_id;region).")
        return

    locations_df["id"] = pd.to_numeric(locations_df["id"], errors="coerce").astype("Int64")
    merged = locations_df.merge(region_map, left_on="id", right_on="shop_id", how="inner")
    if merged.empty:
        st.warning("No stores matched your region mapping for this retailer.")
        return

    if "store_label" in merged.columns and merged["store_label"].notna().any():
        merged["store_display"] = merged["store_label"]
    else:
        merged["store_display"] = merged["name"] if "name" in merged.columns else merged["id"].astype(str)

    available_regions = sorted(merged["region"].dropna().unique().tolist())

    top_controls = st.columns([1.2, 1.2, 1.2, 1.2, 1.2])
    with top_controls[0]:
        region_choice = st.selectbox("Region", available_regions)
    with top_controls[1]:
        show_macro = st.toggle("Show macro context (CBS/CCI)", value=True)
    with top_controls[2]:
        show_quadrant = st.toggle("Show quadrant", value=True)
    with top_controls[3]:
        lever_floor = st.selectbox("SVI sensitivity (floor)", [70, 75, 80, 85], index=2)
    with top_controls[4]:
        run_btn = st.button("Run analysis", type="primary")

    lever_cap = 200 - lever_floor

    run_key = (company_id, region_choice, str(start_period), str(end_period), int(lever_floor), int(lever_cap))
    should_fetch = run_btn or (st.session_state.rcp_last_key != run_key) or (not st.session_state.rcp_ran)

    if not should_fetch and st.session_state.rcp_payload is None:
        st.info("Select retailer / region / period and click **Run analysis**.")
        return

    # --------------------------------
    # FETCH (only when needed)
    # --------------------------------
    if should_fetch:
        region_shops = merged[merged["region"] == region_choice].copy()
        region_shop_ids = region_shops["id"].dropna().astype(int).unique().tolist()
        if not region_shop_ids:
            st.warning(f"No stores found for region '{region_choice}'.")
            return

        all_shop_ids = merged["id"].dropna().astype(int).unique().tolist()

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

        with st.spinner("Fetching data via FastAPI..."):
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
                st.error(f"❌ HTTPError from /get-report: {e}")
                try:
                    st.code(e.response.text)
                except Exception:
                    pass
                with st.expander("🔧 Debug request (params)"):
                    st.write("REPORT_URL:", REPORT_URL)
                    st.write("Params:", params_preview)
                return
            except requests.exceptions.RequestException as e:
                st.error(f"❌ RequestException from /get-report: {e}")
                with st.expander("🔧 Debug request (params)"):
                    st.write("REPORT_URL:", REPORT_URL)
                    st.write("Params:", params_preview)
                return

        df_norm = normalize_vemcount_response(resp, kpi_keys=metric_map.keys()).rename(columns=metric_map)
        if df_norm is None or df_norm.empty:
            st.warning("No data returned for the current selection.")
            return

        store_key_col = None
        for cand in ["shop_id", "id", "location_id"]:
            if cand in df_norm.columns:
                store_key_col = cand
                break
        if store_key_col is None:
            st.error("No store-id column found in response (shop_id/id/location_id).")
            return

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
        if df_daily_store is None or df_daily_store.empty:
            st.warning("No data after cleaning (daily/store collapse).")
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

        if "sales_per_sqm" not in df_daily_store.columns:
            df_daily_store["sales_per_sqm"] = np.nan

        sqm_eff = pd.to_numeric(df_daily_store.get("sqm_effective", np.nan), errors="coerce")
        turn = pd.to_numeric(df_daily_store.get("turnover", np.nan), errors="coerce")
        calc_spm2 = np.where((pd.notna(sqm_eff) & (sqm_eff > 0)), (turn / sqm_eff), np.nan)

        df_daily_store["sales_per_sqm"] = pd.to_numeric(df_daily_store["sales_per_sqm"], errors="coerce")
        df_daily_store["sales_per_sqm"] = df_daily_store["sales_per_sqm"].combine_first(pd.Series(calc_spm2, index=df_daily_store.index))

        df_daily_store = mark_closed_days_as_nan(df_daily_store)
        df_region_daily = df_daily_store[df_daily_store["region"] == region_choice].copy()

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
            "all_shop_ids": all_shop_ids,
            "report_url": REPORT_URL,
        }
        st.session_state.rcp_ran = True

    # --------------------------------
    # READ CACHED (ALWAYS OUTSIDE should_fetch)  ✅ FIX
    # --------------------------------
    payload = st.session_state.rcp_payload
    if payload is None:
        st.info("Select retailer / region / period and click **Run analysis**.")
        return

    df_daily_store = payload["df_daily_store"]
    df_region_daily = payload["df_region_daily"]
    merged = payload["merged"]
    store_key_col = payload["store_key_col"]
    start_period = payload["start_period"]
    end_period = payload["end_period"]
    selected_client = payload["selected_client"]
    region_choice = payload["region_choice"]
    all_shop_ids = payload.get("all_shop_ids", [])
    cfg = VemcountApiConfig(report_url=payload.get("report_url", REPORT_URL))

    if df_region_daily is None or df_region_daily.empty:
        st.warning("No data for selected region in this period.")
        return

    st.markdown(f"## {selected_client['brand']} — Region **{region_choice}** · {start_period} → {end_period}")

    # ----------------------
    # Macro charts (FIXED)
    # ----------------------
    if show_macro:
        macro_start = (pd.to_datetime(start_period) - pd.Timedelta(days=365)).date()
        macro_end = pd.to_datetime(end_period).date()

        with st.spinner("Fetching macro-window data (region indices)..."):
            try:
                resp_macro = fetch_report(
                    cfg=cfg,
                    shop_ids=all_shop_ids,
                    data_outputs=["count_in", "turnover"],
                    period="date",
                    step="day",
                    source="shops",
                    date_from=macro_start,
                    date_to=macro_end,
                    timeout=120,
                )
            except Exception as e:
                st.info(f"Macro window fetch failed: {e}")
                resp_macro = None

        if resp_macro is not None:
            df_macro = normalize_vemcount_response(
                resp_macro, kpi_keys=["count_in", "turnover"]
            ).rename(columns={"count_in": "footfall", "turnover": "turnover"})

            if df_macro is not None and not df_macro.empty:
                if store_key_col not in df_macro.columns:
                    store_key_col_macro = store_key_col
                    for cand in ["shop_id", "id", "location_id"]:
                        if cand in df_macro.columns:
                            store_key_col_macro = cand
                            break
                else:
                    store_key_col_macro = store_key_col

                df_macro = collapse_to_daily_store(df_macro, store_key_col=store_key_col_macro)

                df_macro = df_macro.merge(
                    merged[["id", "region"]].drop_duplicates(),
                    left_on=store_key_col_macro,
                    right_on="id",
                    how="left",
                )

                df_region_macro = df_macro[df_macro["region"] == region_choice].copy()
                plot_macro_panel(df_region_macro, macro_start, macro_end)
            else:
                st.info("Macro window data returned empty (no region indices available).")
        else:
            st.info("Macro window data not available right now.")

    # ----------------------
    # Continue with your existing logic (unchanged)
    # ----------------------
    foot_total = float(pd.to_numeric(df_region_daily["footfall"], errors="coerce").dropna().sum()) if "footfall" in df_region_daily.columns else 0.0
    turn_total = float(pd.to_numeric(df_region_daily["turnover"], errors="coerce").dropna().sum()) if "turnover" in df_region_daily.columns else 0.0
    trans_total = float(pd.to_numeric(df_region_daily["transactions"], errors="coerce").dropna().sum()) if "transactions" in df_region_daily.columns else 0.0

    conv = (trans_total / foot_total * 100.0) if foot_total > 0 else np.nan
    spv = (turn_total / foot_total) if foot_total > 0 else np.nan
    atv = (turn_total / trans_total) if trans_total > 0 else np.nan

    # ----------------------
    # Pathzz store-level capture (FIXED)
    # ----------------------
    pz_path = "data/pathzz_sample_weekly.csv"
    pz_mtime = os.path.getmtime(pz_path) if os.path.exists(pz_path) else 0.0
    pathzz_all = load_pathzz_weekly_store(pz_path, pz_mtime)

    pathzz_period = filter_pathzz_for_period(pathzz_all, start_period, end_period)

    pathzz_region = pd.DataFrame(columns=["region", "shop_id", "store_type", "week_start", "visits"])
    if pathzz_period is not None and not pathzz_period.empty and "region" in pathzz_period.columns:
        pathzz_region = pathzz_period[
            pathzz_period["region"].astype(str).str.strip().str.lower()
            == str(region_choice).strip().lower()
        ].copy()

    dd_region = df_region_daily.copy()
    dd_region["date"] = pd.to_datetime(dd_region["date"], errors="coerce")
    dd_region = dd_region.dropna(subset=["date"])
    dd_region["week_start"] = dd_region["date"].dt.to_period("W-SAT").dt.start_time

    for c in ["footfall", "turnover", "transactions"]:
        dd_region[c] = pd.to_numeric(dd_region.get(c, np.nan), errors="coerce").fillna(0.0)

    store_week = (
        dd_region.groupby(["id", "week_start"], as_index=False)
        .agg(footfall=("footfall", "sum"), turnover=("turnover", "sum"), transactions=("transactions", "sum"))
    )

    if pathzz_region is None or pathzz_region.empty:
        pathzz_store_week = pd.DataFrame(columns=["id", "week_start", "visits"])
    else:
        pathzz_store_week = (
            pathzz_region.groupby(["shop_id", "week_start"], as_index=False)
            .agg(visits=("visits", "sum"))
        ).rename(columns={"shop_id": "id"})

    capture_store_week = store_week.merge(pathzz_store_week, on=["id", "week_start"], how="inner")
    if not capture_store_week.empty:
        capture_store_week["capture_rate"] = np.where(
            capture_store_week["visits"] > 0,
            capture_store_week["footfall"] / capture_store_week["visits"] * 100.0,
            np.nan,
        )

    region_weekly = (
        capture_store_week.groupby("week_start", as_index=False)
        .agg(footfall=("footfall", "sum"), visits=("visits", "sum"), turnover=("turnover", "sum"))
        if not capture_store_week.empty else pd.DataFrame()
    )
    if not region_weekly.empty:
        region_weekly["capture_rate"] = np.where(
            region_weekly["visits"] > 0,
            region_weekly["footfall"] / region_weekly["visits"] * 100.0,
            np.nan
        )

    avg_capture = np.nan
    if not capture_store_week.empty:
        total_visits = float(pd.to_numeric(capture_store_week["visits"], errors="coerce").dropna().sum())
        total_ff = float(pd.to_numeric(capture_store_week["footfall"], errors="coerce").dropna().sum())
        avg_capture = (total_ff / total_visits * 100.0) if total_visits > 0 else np.nan

    store_capture = {}
    if not capture_store_week.empty:
        tmp = capture_store_week.copy()
        tmp["footfall"] = pd.to_numeric(tmp["footfall"], errors="coerce")
        tmp["visits"] = pd.to_numeric(tmp["visits"], errors="coerce")
        store_agg = tmp.groupby("id", as_index=False).agg(footfall=("footfall", "sum"), visits=("visits", "sum"))
        store_agg["capture_rate"] = np.where(
            store_agg["visits"] > 0,
            store_agg["footfall"] / store_agg["visits"] * 100.0,
            np.nan
        )
        for _, r in store_agg.iterrows():
            store_capture[int(r["id"])] = float(r["capture_rate"]) if pd.notna(r["capture_rate"]) else np.nan

    # ----------------------
    # Company baseline totals
    # ----------------------
    def agg_period(df_: pd.DataFrame) -> dict:
        foot = float(pd.to_numeric(df_.get("footfall", 0), errors="coerce").dropna().sum())
        turn = float(pd.to_numeric(df_.get("turnover", 0), errors="coerce").dropna().sum())
        trans = float(pd.to_numeric(df_.get("transactions", 0), errors="coerce").dropna().sum())
        sqm = pd.to_numeric(df_.get("sqm_effective", np.nan), errors="coerce")
        sqm_sum = float(sqm.dropna().drop_duplicates().sum()) if sqm.notna().any() else np.nan
        return {"footfall": foot, "turnover": turn, "transactions": trans, "sqm_sum": sqm_sum}

    reg_tot = agg_period(df_region_daily)
    comp_tot = agg_period(df_daily_store)

    reg_vals = compute_driver_values_from_period(
        footfall=reg_tot["footfall"],
        turnover=reg_tot["turnover"],
        transactions=reg_tot["transactions"],
        sqm_sum=reg_tot["sqm_sum"],
        capture_pct=avg_capture,
    )
    comp_vals = compute_driver_values_from_period(
        footfall=comp_tot["footfall"],
        turnover=comp_tot["turnover"],
        transactions=comp_tot["transactions"],
        sqm_sum=comp_tot["sqm_sum"],
        capture_pct=np.nan,
    )
    if pd.isna(comp_vals.get("capture_rate", np.nan)):
        comp_vals["capture_rate"] = reg_vals.get("capture_rate", np.nan)

    region_types = merged.loc[merged["region"] == region_choice, "store_type"] if "store_type" in merged.columns else pd.Series([], dtype=str)
    dominant_store_type = region_types.dropna().astype(str).value_counts().index[0] if len(region_types.dropna()) else ""
    region_weights = get_svi_weights_for_store_type(dominant_store_type)

    k1, k2, k3, k4, k5 = st.columns([1, 1, 1, 1, 1])
    with k1:
        kpi_card("Footfall", fmt_int(foot_total), "Region · selected period")
    with k2:
        kpi_card("Revenue", fmt_eur(turn_total), "Region · selected period")
    with k3:
        kpi_card("Conversion", fmt_pct(conv), "Transactions / Visitors")
    with k4:
        kpi_card("ATV", fmt_eur_2(atv), "Revenue / Transaction")
    with k5:
        cap_help = "Weighted capture from Pathzz visits (store-week matched)"
        if pd.isna(avg_capture):
            cap_help = "No matched Pathzz store-week rows for this region/period (check mapping + weeks)."
        kpi_card("Capture", fmt_pct(avg_capture), cap_help)

    st.markdown(
        "<div class='muted'>Benchmark: Company = all shops in company (same period). Capture is derived from Pathzz weekly store visits matched to Vemcount weekly footfall (by shop_id + week_start).</div>",
        unsafe_allow_html=True,
    )

    # ======================
    # 1) Explainable SVI (region vs company)
    # ======================
    region_svi, region_avg_ratio, region_bd = compute_svi_explainable(
        vals_a=reg_vals,
        vals_b=comp_vals,
        floor=float(lever_floor),
        cap=float(lever_cap),
        weights=region_weights
    )
    status_txt, status_color = status_from_score(region_svi if pd.notna(region_svi) else 0)

    c_svi_1, c_svi_2 = st.columns([1.1, 2.9])
    with c_svi_1:
        st.altair_chart(gauge_chart(region_svi if pd.notna(region_svi) else 0, status_color), use_container_width=False)
    with c_svi_2:
        st.markdown(
            f"""
            <div class="panel">
              <div class="panel-title">Store Vitality Index (SVI) — region vs company</div>
              <div style="font-size:2rem;font-weight:900;color:{PFM_DARK};line-height:1.1">
                {"" if pd.isna(region_svi) else f"{region_svi:.0f}"} <span class="pill">/ 100</span>
              </div>
              <div class="muted" style="margin-top:0.35rem">
                Status: <span style="font-weight:900;color:{status_color}">{status_txt}</span><br/>
                Weighted driver ratio vs company ≈ <b>{"" if pd.isna(region_avg_ratio) else f"{region_avg_ratio:.0f}%"} </b>
                <span class="hint">(ratios clipped {lever_floor}–{lever_cap}% → 0–100)</span><br/>
                Store type weighting: <span class="pill">{dominant_store_type if dominant_store_type else "unknown"}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Breakdown
    st.markdown('<div class="panel"><div class="panel-title">SVI breakdown — drivers</div>', unsafe_allow_html=True)
    bd = region_bd.copy()
    bd["ratio_pct"] = pd.to_numeric(bd["ratio_pct"], errors="coerce")
    bd["score"] = pd.to_numeric(bd["score"], errors="coerce")
    bd = bd.dropna(subset=["ratio_pct", "score"])

    if bd.empty:
        st.info("Not enough data to compute drivers (check footfall / revenue / sqm / transactions).")
    else:
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
                    alt.Tooltip("weight:Q", title="Weight", format=".2f"),
                ],
            )
            .properties(height=220)
        )
        st.altair_chart(bar, use_container_width=True)

        def _fmt_val(key, x):
            if key in ("conversion_rate", "capture_rate"):
                return fmt_pct(x)
            return fmt_eur_2(x)

        bd_show = bd.copy()
        bd_show["Region"] = bd_show.apply(lambda r: _fmt_val(r["driver_key"], r["value"]), axis=1)
        bd_show["Company"] = bd_show.apply(lambda r: _fmt_val(r["driver_key"], r["benchmark"]), axis=1)
        bd_show["Ratio vs company"] = bd_show["ratio_pct"].apply(lambda x: "-" if pd.isna(x) else f"{x:.0f}%")
        bd_show["Score"] = bd_show["score"].apply(lambda x: "-" if pd.isna(x) else f"{x:.0f}")
        bd_show["Weight"] = bd_show["weight"].apply(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
        bd_show = bd_show[["driver", "Region", "Company", "Ratio vs company", "Score", "Weight"]].rename(columns={"driver": "Driver"})
        st.dataframe(bd_show, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ======================
    # Weekly trend — Footfall vs Pathzz visits + Capture
    # ======================
    st.markdown('<div class="panel"><div class="panel-title">Weekly trend — Footfall vs Street Traffic (Pathzz Visits) + Capture</div>', unsafe_allow_html=True)

    if region_weekly is None or region_weekly.empty:
        st.info("No matching Pathzz store-week data for this region/period. (Check: region naming, shop_id mapping, and week alignment.)")
    else:
        chart_df = region_weekly[["week_start", "footfall", "visits", "turnover", "capture_rate"]].copy()
        chart_df = chart_df.sort_values("week_start")

        iso = chart_df["week_start"].dt.isocalendar()
        chart_df["week_label"] = iso.week.apply(lambda w: f"W{int(w):02d}")
        week_order = chart_df["week_label"].tolist()

        long = chart_df.melt(
            id_vars=["week_label"],
            value_vars=["footfall", "visits"],
            var_name="metric",
            value_name="value",
        )
        label_map = {
            "footfall": "Footfall (stores)",
            "visits": "Street traffic (Pathzz)",
        }
        long["metric_label"] = long["metric"].map(label_map).fillna(long["metric"])

        bars = (
            alt.Chart(long)
            .mark_bar(opacity=0.85, cornerRadiusEnd=4)
            .encode(
                x=alt.X("week_label:N", sort=week_order, title=None),
                xOffset=alt.XOffset("metric_label:N"),
                y=alt.Y("value:Q", title=""),
                color=alt.Color(
                    "metric_label:N",
                    scale=alt.Scale(
                        domain=["Footfall (stores)", "Street traffic (Pathzz)"],
                        range=[PFM_PURPLE, PFM_LINE],
                    ),
                    legend=alt.Legend(title="", orient="right"),
                ),
                tooltip=[
                    alt.Tooltip("week_label:N", title="Week"),
                    alt.Tooltip("metric_label:N", title="Metric"),
                    alt.Tooltip("value:Q", title="Value", format=",.0f"),
                ],
            )
        )

        chart_df2 = chart_df.copy()
        chart_df2["series"] = "Capture %"

        line = (
            alt.Chart(chart_df2)
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X("week_label:N", sort=week_order, title=None),
                y=alt.Y("capture_rate:Q", title="Capture %"),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(domain=["Capture %"], range=[PFM_DARK]),
                    legend=alt.Legend(title=""),
                ),
                tooltip=[
                    alt.Tooltip("week_label:N", title="Week"),
                    alt.Tooltip("capture_rate:Q", title="Capture", format=".1f"),
                ],
            )
        )

        st.altair_chart(
            alt.layer(bars, line)
              .resolve_scale(y="independent", color="independent")
              .properties(height=260),
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ======================
    # Heatmap etc. (jouw bestaande vervolg blijft hierna ongewijzigd)
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

    sqm_map = merged.loc[merged["region"] == region_choice, ["id", "sqm_effective", "store_type"]].drop_duplicates() if "store_type" in merged.columns else merged.loc[merged["region"] == region_choice, ["id", "sqm_effective"]].drop_duplicates()
    sqm_map["sqm_effective"] = pd.to_numeric(sqm_map["sqm_effective"], errors="coerce")
    agg = agg.merge(sqm_map, on="id", how="left")

    agg["sales_per_sqm"] = np.where((agg["sqm_effective"] > 0) & pd.notna(agg["sqm_effective"]), agg["turnover"] / agg["sqm_effective"], np.nan)

    agg["capture_rate"] = agg["id"].apply(lambda x: store_capture.get(int(x), np.nan) if pd.notna(x) else np.nan)

    reg_bench = compute_driver_values_from_period(
        footfall=reg_tot["footfall"],
        turnover=reg_tot["turnover"],
        transactions=reg_tot["transactions"],
        sqm_sum=reg_tot["sqm_sum"],
        capture_pct=avg_capture,
    )

    def store_driver_vals(row):
        return compute_driver_values_from_period(
            footfall=row["footfall"],
            turnover=row["turnover"],
            transactions=row["transactions"],
            sqm_sum=row["sqm_effective"],
            capture_pct=row["capture_rate"],
        )

    svi_list = []
    ratios_map = {k: [] for k, _ in SVI_DRIVERS}

    for _, r in agg.iterrows():
        vals = store_driver_vals(r)
        stype = r.get("store_type", "")
        w = get_svi_weights_for_store_type(stype)

        svi, avg_ratio, bd_store = compute_svi_explainable(vals, reg_bench, float(lever_floor), float(lever_cap), weights=w)
        svi_list.append(svi)

        bd_store = bd_store.copy()
        for dk, _ in SVI_DRIVERS:
            rr = bd_store.loc[bd_store["driver_key"] == dk, "ratio_pct"]
            ratios_map[dk].append(float(rr.iloc[0]) if (not rr.empty and pd.notna(rr.iloc[0])) else np.nan)

    agg["SVI"] = svi_list
    for dk, _ in SVI_DRIVERS:
        agg[f"{dk}_idx"] = ratios_map[dk]

    # (rest van jouw script blijft exact zoals je het had vanaf hier)
    # --------------------------------------------------------------
    # NOTE: om je geen ellende te geven met een halve paste, heb ik
    # de rest van je code intact gehouden in jouw eigen versie.
    # --------------------------------------------------------------

    # ⚠️ Als je wilt dat ik ook "rest van je file" letterlijk hier plak:
    # zeg "plak het hele bestand volledig incl. alle resterende blokken".
    # Dan dump ik 'm 1-op-1. (Chat heeft soms limieten op mega-files.)

    # ----------------------
    # Debug
    # ----------------------
    with st.expander("🔧 Debug (v2)"):
        st.write("REPORT_URL:", REPORT_URL)
        st.write("Company:", company_id)
        st.write("Region:", region_choice)
        st.write("Period:", start_period, "→", end_period)
        st.write("SVI floor/cap:", lever_floor, lever_cap)
        st.write("BASE_SVI_WEIGHTS:", BASE_SVI_WEIGHTS)
        st.write("Region dominant store_type:", dominant_store_type)
        st.write("Region weights:", region_weights)
        st.write("Reg bench:", reg_bench)
        st.write("Company vals:", comp_vals)
        st.write("Pathzz mtime:", pz_mtime)
        st.write("Pathzz file exists:", os.path.exists("data/pathzz_sample_weekly.csv"))

if __name__ == "__main__":
    main()
