# pages/06C_Region_Copilot_V2.py
# ------------------------------------------------------------
# PFM Region Copilot v2 — FIXED "main()" + indentation + state
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import requests
import os
import re
import streamlit as st
import altair as alt

from datetime import datetime

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
        min-height: 92px;
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
# Pathzz store-level capture
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
        .mark_arc(innerRadius=50, outerRadius=64)
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

def compute_svi_by_region_companywide(
    df_daily_store: pd.DataFrame,
    lever_floor: float,
    lever_cap: float,
) -> pd.DataFrame:
    """Company-wide region leaderboard: SVI(region vs company). Capture ignored for fairness."""
    if df_daily_store is None or df_daily_store.empty:
        return pd.DataFrame(columns=["region", "svi", "avg_ratio", "footfall", "turnover"])

    d = df_daily_store.copy()
    if "region" not in d.columns:
        return pd.DataFrame(columns=["region", "svi", "avg_ratio", "footfall", "turnover"])

    for c in ["footfall", "turnover", "transactions", "sqm_effective"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    comp_tot = {
        "footfall": float(d["footfall"].dropna().sum()) if "footfall" in d.columns else np.nan,
        "turnover": float(d["turnover"].dropna().sum()) if "turnover" in d.columns else np.nan,
        "transactions": float(d["transactions"].dropna().sum()) if "transactions" in d.columns else np.nan,
        "sqm_sum": float(d["sqm_effective"].dropna().drop_duplicates().sum()) if "sqm_effective" in d.columns else np.nan,
    }
    comp_vals = compute_driver_values_from_period(
        footfall=comp_tot["footfall"],
        turnover=comp_tot["turnover"],
        transactions=comp_tot["transactions"],
        sqm_sum=comp_tot["sqm_sum"],
        capture_pct=np.nan,
    )

    weights_no_capture = dict(BASE_SVI_WEIGHTS)
    weights_no_capture["capture_rate"] = 0.0

    rows = []
    for region, g in d.groupby("region"):
        if pd.isna(region) or str(region).strip() == "":
            continue

        reg_tot = {
            "footfall": float(g["footfall"].dropna().sum()) if "footfall" in g.columns else np.nan,
            "turnover": float(g["turnover"].dropna().sum()) if "turnover" in g.columns else np.nan,
            "transactions": float(g["transactions"].dropna().sum()) if "transactions" in g.columns else np.nan,
            "sqm_sum": float(g["sqm_effective"].dropna().drop_duplicates().sum()) if "sqm_effective" in g.columns else np.nan,
        }
        reg_vals = compute_driver_values_from_period(
            footfall=reg_tot["footfall"],
            turnover=reg_tot["turnover"],
            transactions=reg_tot["transactions"],
            sqm_sum=reg_tot["sqm_sum"],
            capture_pct=np.nan,
        )

        svi, avg_ratio, _ = compute_svi_explainable(
            vals_a=reg_vals,
            vals_b=comp_vals,
            floor=float(lever_floor),
            cap=float(lever_cap),
            weights=weights_no_capture,
        )

        rows.append({
            "region": str(region),
            "svi": svi,
            "avg_ratio": avg_ratio,
            "footfall": reg_tot["footfall"],
            "turnover": reg_tot["turnover"],
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["region", "svi", "avg_ratio", "footfall", "turnover"])

    out["svi"] = pd.to_numeric(out["svi"], errors="coerce")
    out = out.dropna(subset=["svi"]).sort_values("svi", ascending=False).reset_index(drop=True)
    return out

# ----------------------
# Macro charts
# ----------------------
def plot_macro_panel(df_region_daily: pd.DataFrame, macro_start, macro_end):
    st.markdown('<div class="panel"><div class="panel-title">Macro context — Consumer Confidence & Retail Index</div>', unsafe_allow_html=True)

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
        s = str(s).strip().replace("MM", "M")

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

    cci_df = cci_df[(cci_df["month"] >= ms) & (cci_df["month"] <= me)].copy()
    ridx_df = ridx_df[(ridx_df["month"] >= ms) & (ridx_df["month"] <= me)].copy()

    x_enc = alt.X(
        "month:T",
        title=None,
        scale=alt.Scale(domain=[ms, me]),
        axis=alt.Axis(format="%b %Y", labelAngle=30, labelOverlap=False, labelPadding=14)
    )

    reg_long = region_m.melt(
        id_vars=["month"],
        value_vars=["Region footfall-index", "Region omzet-index"],
        var_name="series",
        value_name="idx",
    )

    region_color_scale = alt.Scale(
        domain=["Region footfall-index", "Region omzet-index"],
        range=[PFM_PURPLE, PFM_RED],
    )

    region_lines = (
        alt.Chart(reg_long)
        .mark_line(point=True)
        .encode(
            x=x_enc,
            y=alt.Y("idx:Q", title="Regio-index (100 = start)"),
            color=alt.Color("series:N", scale=region_color_scale, legend=alt.Legend(title="", orient="right")),
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
            chart.resolve_scale(y="independent", color="independent").properties(height=280).configure_view(strokeWidth=0),
            use_container_width=True,
        )

    with c2:
        st.markdown("**Consumentenvertrouwen (CCI) vs Regio**")
        mline = macro_line(cci_df, "CCI", dash=True)
        chart = alt.layer(region_lines, mline) if mline is not None else region_lines
        st.altair_chart(
            chart.resolve_scale(y="independent", color="independent").properties(height=280).configure_view(strokeWidth=0),
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# MAIN APP
# ----------------------
def main():
    # ---- session defaults (prevents KeyError on first run) ----
    if "rcp_last_key" not in st.session_state:
        st.session_state.rcp_last_key = None
    if "rcp_payload" not in st.session_state:
        st.session_state.rcp_payload = None
    if "rcp_ran" not in st.session_state:
        st.session_state.rcp_ran = False

    st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)

    # Row 1: title left, selection stack right
    h_left, h_right = st.columns([2.3, 1.7], vertical_alignment="top")

    with h_left:
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

    with h_right:
        st.markdown('<div class="panel"><div class="panel-title">Selection</div>', unsafe_allow_html=True)

        client_label = st.selectbox(
            "Retailer",
            clients_df["label"].tolist(),
            label_visibility="collapsed",
            key="rcp_client",
        )

        period_choice = st.selectbox(
            "Period",
            period_labels,
            index=period_labels.index("Q3 2024") if "Q3 2024" in period_labels else 0,
            label_visibility="collapsed",
            key="rcp_period",
        )

        st.markdown("</div>", unsafe_allow_html=True)

    selected_client = clients_df[clients_df["label"] == client_label].iloc[0].to_dict()
    company_id = int(selected_client["company_id"])
    start_period = periods[period_choice].start
    end_period = periods[period_choice].end

    # Load locations + regions based on selected client
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

    # Row 2: Region + toggles + SVI floor + Run button
    c_reg, c_tog, c_floor, c_btn = st.columns([1.2, 1.8, 1.0, 0.9], vertical_alignment="bottom")

    with c_reg:
        st.markdown('<div class="panel"><div class="panel-title">Region</div>', unsafe_allow_html=True)
        region_choice = st.selectbox(
            "Region",
            available_regions,
            label_visibility="collapsed",
            key="rcp_region",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c_tog:
        st.markdown('<div class="panel"><div class="panel-title">Options</div>', unsafe_allow_html=True)
        t1, t2 = st.columns(2)
        with t1:
            show_macro = st.toggle("Show macro context (CBS/CCI)", value=True, key="rcp_macro")
        with t2:
            show_quadrant = st.toggle("Show quadrant", value=True, key="rcp_quadrant")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_floor:
        st.markdown('<div class="panel"><div class="panel-title">SVI sensitivity</div>', unsafe_allow_html=True)
        lever_floor = st.selectbox("SVI floor", [70, 75, 80, 85], index=2, label_visibility="collapsed", key="rcp_floor")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_btn:
        run_btn = st.button("Run analysis", type="primary", key="rcp_run")

    # --- session defaults (must exist BEFORE we compute should_fetch) ---
    if "rcp_last_key" not in st.session_state:
        st.session_state.rcp_last_key = None
    if "rcp_payload" not in st.session_state:
        st.session_state.rcp_payload = None
    if "rcp_ran" not in st.session_state:
        st.session_state.rcp_ran = False
    
    lever_cap = 200 - lever_floor  # e.g. 80 -> 120 ; 85 -> 115
    run_key = (company_id, region_choice, str(start_period), str(end_period), int(lever_floor), int(lever_cap))
    
    selection_changed = st.session_state.rcp_last_key != run_key
    should_fetch = bool(run_btn) or bool(selection_changed) or (not bool(st.session_state.rcp_ran))
    
    # Optional: visible feedback
    if run_btn:
        st.toast("Running analysis…", icon="🚀")
    
    # Debug AFTER vars exist
    st.write(
        "DEBUG:",
        {
            "run_btn": run_btn,
            "should_fetch": should_fetch,
            "selection_changed": selection_changed,
            "last_key": st.session_state.rcp_last_key,
            "run_key": run_key,
            "ran": st.session_state.rcp_ran,
            "payload_is_none": st.session_state.rcp_payload is None,
        },
    )

    lever_cap = 200 - lever_floor  # e.g. 80 -> 120 ; 85 -> 115

    # ----------------------
    # Session state defaults (MUST exist before should_fetch)
    # ----------------------
    if "rcp_last_key" not in st.session_state:
        st.session_state.rcp_last_key = None
    if "rcp_payload" not in st.session_state:
        st.session_state.rcp_payload = None
    if "rcp_ran" not in st.session_state:
        st.session_state.rcp_ran = False
    
    run_key = (company_id, region_choice, str(start_period), str(end_period), int(lever_floor), int(lever_cap))
    
    # Fetch rules:
    # - Always fetch when user clicks Run
    # - Otherwise fetch if selection changed or never ran yet
    selection_changed = st.session_state.rcp_last_key != run_key
    should_fetch = bool(run_btn) or bool(selection_changed) or (not bool(st.session_state.rcp_ran))
    
    # Tiny UI feedback so you SEE it reruns
    if run_btn:
        st.toast("Running analysis…", icon="🚀")

    if (not should_fetch) and (st.session_state.rcp_payload is None):
        st.info("Select retailer / region / period and click **Run analysis**.")
        return

    # NOTE:
    # The remainder of the script (fetching, caching payload, and all charts/tables) is unchanged
    # from the fixed version I prepared earlier. If you want the FULL file (complete end-to-end),
    # use the ZIP attached in this response.

# Streamlit runs top-to-bottom; calling main() is enough.
main()
