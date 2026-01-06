# pages/06C_Region_Copilot_V2_DRILLDOWN.py

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
    page_title="PFM Region Copilot v2 (Regio + Drilldown)",
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
      .small-note {{
        color: {PFM_GRAY};
        font-size: 0.80rem;
        line-height: 1.25rem;
      }}
      .divider {{
        height: 0.65rem;
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

    # recompute derived metrics robustly (prefer truth from sums)
    if "turnover" in out.columns and "footfall" in out.columns:
        out["sales_per_visitor"] = np.where(out["footfall"] > 0, out["turnover"] / out["footfall"], np.nan)
    if "transactions" in out.columns and "footfall" in out.columns:
        out["conversion_rate"] = np.where(out["footfall"] > 0, out["transactions"] / out["footfall"] * 100.0, np.nan)
    if "turnover" in out.columns and "transactions" in out.columns:
        out["avg_basket_size"] = np.where(out["transactions"] > 0, out["turnover"] / out["transactions"], np.nan)
        out["sales_per_transaction"] = out["avg_basket_size"]

    return out

# ----------------------
# Weekly (region) for capture chart
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
# SVI v2 (Explainable)
# ----------------------
def _ratio_to_score_0_100(ratio: float, floor: float = 70.0, cap: float = 130.0) -> float:
    if pd.isna(ratio):
        return np.nan
    r = float(np.clip(ratio, floor, cap))
    return (r - floor) / (cap - floor) * 100.0

def build_region_svi_v2(
    df_daily_store: pd.DataFrame,
    merged: pd.DataFrame,
    store_key_col: str,
    region_choice: str,
    capture_weekly: pd.DataFrame,
    weights: dict | None = None,
) -> dict:
    if weights is None:
        weights = {
            "sales_per_sqm": 0.30,
            "capture_rate": 0.15,
            "conversion_rate": 0.20,
            "avg_basket_size": 0.20,
            "sales_per_transaction": 0.15,
        }

    if df_daily_store is None or df_daily_store.empty:
        return {"region_svi": np.nan, "components": pd.DataFrame(), "benchmarks": {}}

    d = df_daily_store.copy()

    try:
        d = ensure_region_column(d, merged, store_key_col)
    except Exception:
        pass

    if "region" not in d.columns:
        region_like = [c for c in d.columns if str(c).lower().startswith("region")]
        if region_like:
            d["region"] = d[region_like[0]]

    if "region" not in d.columns and merged is not None and not merged.empty:
        if "id" in merged.columns and "region" in merged.columns and store_key_col in d.columns:
            tmp_map = merged[["id", "region"]].drop_duplicates().copy()
            tmp_map["id"] = pd.to_numeric(tmp_map["id"], errors="coerce").astype("Int64")
            d[store_key_col] = pd.to_numeric(d[store_key_col], errors="coerce").astype("Int64")
            d = d.merge(tmp_map, left_on=store_key_col, right_on="id", how="left")

    if "region" not in d.columns:
        if "region_x" in d.columns and "region_y" in d.columns:
            d["region"] = d["region_x"].combine_first(d["region_y"])
        elif "region_x" in d.columns:
            d["region"] = d["region_x"]
        elif "region_y" in d.columns:
            d["region"] = d["region_y"]

    if "region" not in d.columns:
        return {
            "region_svi": np.nan,
            "components": pd.DataFrame(),
            "benchmarks": {"error": "region_missing_in_df_daily_store", "cols": d.columns.tolist()},
        }

    d_reg = d[d["region"] == region_choice].copy()
    d_all = d.copy()

    if d_reg.empty or d_all.empty:
        return {"region_svi": np.nan, "components": pd.DataFrame(), "benchmarks": {}}

    def _agg_period(df_: pd.DataFrame) -> dict:
        out = {}
        foot = float(pd.to_numeric(df_.get("footfall", 0), errors="coerce").fillna(0).sum())
        turn = float(pd.to_numeric(df_.get("turnover", 0), errors="coerce").fillna(0).sum())
        trans = float(pd.to_numeric(df_.get("transactions", 0), errors="coerce").fillna(0).sum())

        out["footfall"] = foot
        out["turnover"] = turn
        out["transactions"] = trans

        out["conversion_rate"] = (trans / foot * 100.0) if foot > 0 else np.nan
        out["avg_basket_size"] = (turn / trans) if trans > 0 else np.nan
        out["sales_per_visitor"] = (turn / foot) if foot > 0 else np.nan

        sqm = pd.to_numeric(df_.get("sqm_effective", np.nan), errors="coerce")
        sqm_sum = float(sqm.dropna().drop_duplicates().sum()) if sqm.notna().any() else np.nan
        out["sales_per_sqm"] = (turn / sqm_sum) if (pd.notna(sqm_sum) and sqm_sum > 0) else np.nan

        out["sales_per_transaction"] = out["avg_basket_size"]
        return out

    reg_vals = _agg_period(d_reg)
    all_vals = _agg_period(d_all)

    reg_capture = np.nan
    if capture_weekly is not None and not capture_weekly.empty and "capture_rate" in capture_weekly.columns:
        reg_capture = float(pd.to_numeric(capture_weekly["capture_rate"], errors="coerce").dropna().mean())

    reg_vals["capture_rate"] = reg_capture
    all_vals["capture_rate"] = np.nan  # company capture benchmark unknown

    components = []
    for k, w in weights.items():
        v_reg = reg_vals.get(k, np.nan)
        v_bench = all_vals.get(k, np.nan)

        if k == "capture_rate":
            ratio = 100.0
            bench = np.nan
        else:
            if pd.notna(v_reg) and pd.notna(v_bench) and float(v_bench) != 0.0:
                ratio = (float(v_reg) / float(v_bench)) * 100.0
                bench = float(v_bench)
            else:
                ratio = np.nan
                bench = v_bench

        score = _ratio_to_score_0_100(ratio)
        components.append({
            "component": k,
            "weight": w,
            "value": v_reg,
            "benchmark": bench,
            "ratio_vs_benchmark": ratio,
            "score_0_100": score,
        })

    comp_df = pd.DataFrame(components)

    if comp_df.empty or comp_df["score_0_100"].dropna().empty:
        region_svi = np.nan
    else:
        tmp = comp_df.dropna(subset=["score_0_100", "weight"]).copy()
        wsum = float(tmp["weight"].sum()) if not tmp.empty else 0.0
        region_svi = float((tmp["score_0_100"] * tmp["weight"]).sum() / wsum) if wsum > 0 else np.nan

    return {
        "region_svi": region_svi,
        "components": comp_df,
        "benchmarks": {"region": reg_vals, "company": all_vals},
    }

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
# Opportunities v2 (1 lever per store)
# ----------------------
def build_opportunities_v2_one_lever(
    df_region_daily: pd.DataFrame,
    store_key_col: str,
    store_name_col: str = "store_display",
    min_days: int = 20,
) -> pd.DataFrame:
    if df_region_daily is None or df_region_daily.empty:
        return pd.DataFrame()

    d = df_region_daily.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])

    g = (
        d.groupby([store_key_col, store_name_col], as_index=False)
        .agg(
            days=("date", "nunique"),
            footfall=("footfall", "sum"),
            turnover=("turnover", "sum"),
            transactions=("transactions", "sum"),
        )
    )

    if g.empty:
        return pd.DataFrame()

    g["conversion_rate"] = np.where(g["footfall"] > 0, g["transactions"] / g["footfall"] * 100.0, np.nan)
    g["avg_basket_size"] = np.where(g["transactions"] > 0, g["turnover"] / g["transactions"], np.nan)
    g["sales_per_visitor"] = np.where(g["footfall"] > 0, g["turnover"] / g["footfall"], np.nan)

    conv_target = float(pd.to_numeric(g["conversion_rate"], errors="coerce").dropna().median()) if g["conversion_rate"].notna().any() else np.nan
    spv_target = float(pd.to_numeric(g["sales_per_visitor"], errors="coerce").dropna().median()) if g["sales_per_visitor"].notna().any() else np.nan

    period_days = int(d["date"].nunique())
    year_factor = (365.0 / period_days) if period_days > 0 else 1.0

    def conf(dcount: int) -> str:
        if dcount >= 60:
            return "●●●●●"
        if dcount >= 45:
            return "●●●●○"
        if dcount >= 30:
            return "●●●○○"
        if dcount >= 20:
            return "●●○○○"
        return "●○○○○"

    rows = []
    for _, r in g.iterrows():
        days = int(r["days"]) if pd.notna(r["days"]) else 0
        if days < min_days:
            continue

        foot = float(r["footfall"]) if pd.notna(r["footfall"]) else 0.0
        conv = float(r["conversion_rate"]) if pd.notna(r["conversion_rate"]) else np.nan
        spv = float(r["sales_per_visitor"]) if pd.notna(r["sales_per_visitor"]) else np.nan
        atv = float(r["avg_basket_size"]) if pd.notna(r["avg_basket_size"]) else np.nan

        driver = None
        upside_period = 0.0

        if pd.notna(conv_target) and pd.notna(conv) and conv < conv_target and foot > 0 and pd.notna(atv):
            delta_pp = conv_target - conv
            extra_transactions = (delta_pp / 100.0) * foot
            upside_period = extra_transactions * atv
            driver = f"Conversion → regio-median (+{delta_pp:.1f}pp)"
        elif pd.notna(spv_target) and pd.notna(spv) and spv < spv_target and foot > 0:
            delta_spv = spv_target - spv
            upside_period = delta_spv * foot
            driver = f"SPV → regio-median (+{delta_spv:.2f} €/visitor)"
        else:
            continue

        if upside_period <= 0:
            continue

        rows.append({
            store_key_col: r[store_key_col],
            "store_display": r[store_name_col],
            "days": days,
            "confidence": conf(days),
            "driver": driver,
            "theoretical_upside_period": float(upside_period),
            "theoretical_upside_year": float(upside_period * year_factor),
            "annualised_from_days": period_days,
        })

    return pd.DataFrame(rows)

# ----------------------
# Macro helper: dual axis chart
# ----------------------
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

# ----------------------
# NEW: Drilldown helpers
# ----------------------
def safe_div(a: float, b: float) -> float:
    if b is None or pd.isna(b) or float(b) == 0.0:
        return np.nan
    if a is None or pd.isna(a):
        return np.nan
    return float(a) / float(b)

def compute_period_kpis(df: pd.DataFrame, sqm_sum_mode: str = "unique") -> dict:
    """
    df is a slice already containing:
      footfall, turnover, transactions, sqm_effective (optional)
    sqm_sum_mode:
      - "unique": sum unique sqm_effective values (good for region/company)
      - "single": take first sqm_effective (good for single store slice)
    """
    if df is None or df.empty:
        return {}

    foot = float(pd.to_numeric(df.get("footfall", 0), errors="coerce").fillna(0).sum())
    turn = float(pd.to_numeric(df.get("turnover", 0), errors="coerce").fillna(0).sum())
    trans = float(pd.to_numeric(df.get("transactions", 0), errors="coerce").fillna(0).sum())

    sqm_val = np.nan
    if "sqm_effective" in df.columns:
        sqm = pd.to_numeric(df["sqm_effective"], errors="coerce")
        if sqm_sum_mode == "single":
            sqm_val = float(sqm.dropna().iloc[0]) if sqm.dropna().shape[0] else np.nan
        else:
            sqm_val = float(sqm.dropna().drop_duplicates().sum()) if sqm.dropna().shape[0] else np.nan

    conv = (trans / foot * 100.0) if foot > 0 else np.nan
    atv = (turn / trans) if trans > 0 else np.nan
    spv = (turn / foot) if foot > 0 else np.nan
    spm2 = (turn / sqm_val) if (pd.notna(sqm_val) and sqm_val > 0) else np.nan

    return {
        "footfall": foot,
        "turnover": turn,
        "transactions": trans,
        "conversion_rate": conv,
        "avg_basket_size": atv,
        "sales_per_visitor": spv,
        "sales_per_sqm": spm2,
    }

def daily_aggregate(df: pd.DataFrame, level_label: str, store_key_col: str | None = None) -> pd.DataFrame:
    """
    Return daily time series with derived KPIs based on sums (not averages).
    df should include date, footfall, turnover, transactions, sqm_effective (+ store_key if needed).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])

    # daily sums
    agg_cols = {}
    for c in ["footfall", "turnover", "transactions"]:
        if c in d.columns:
            agg_cols[c] = "sum"

    if "sqm_effective" in d.columns:
        # region/company: sum unique sqm across stores per day (stores fixed, so duplicates exist)
        # We'll compute per date: sum unique sqm per store first, then sum.
        if store_key_col and store_key_col in d.columns:
            tmp = d[[store_key_col, "date", "sqm_effective"]].copy()
            tmp[store_key_col] = pd.to_numeric(tmp[store_key_col], errors="coerce")
            tmp["sqm_effective"] = pd.to_numeric(tmp["sqm_effective"], errors="coerce")
            tmp = tmp.dropna(subset=[store_key_col, "date", "sqm_effective"])
            sqm_day = (
                tmp.groupby(["date", store_key_col], as_index=False)["sqm_effective"].first()
                .groupby("date", as_index=False)["sqm_effective"].sum()
                .rename(columns={"sqm_effective": "sqm_sum"})
            )
        else:
            sqm_day = (
                d.groupby("date", as_index=False)["sqm_effective"]
                .apply(lambda s: pd.to_numeric(s, errors="coerce").dropna().drop_duplicates().sum())
                .reset_index()
                .rename(columns={0: "sqm_sum"})
            )
    else:
        sqm_day = pd.DataFrame()

    out = d.groupby("date", as_index=False).agg(agg_cols) if agg_cols else pd.DataFrame()
    if out.empty:
        return pd.DataFrame()

    if not sqm_day.empty:
        out = out.merge(sqm_day, on="date", how="left")
    else:
        out["sqm_sum"] = np.nan

    out["conversion_rate"] = np.where(out["footfall"] > 0, (out["transactions"] / out["footfall"]) * 100.0, np.nan)
    out["avg_basket_size"] = np.where(out["transactions"] > 0, out["turnover"] / out["transactions"], np.nan)
    out["sales_per_visitor"] = np.where(out["footfall"] > 0, out["turnover"] / out["footfall"], np.nan)
    out["sales_per_sqm"] = np.where(out["sqm_sum"] > 0, out["turnover"] / out["sqm_sum"], np.nan)

    out["level"] = level_label
    return out.sort_values("date")

def melt_metrics(df: pd.DataFrame, metric_keys: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    keep = ["date", "level"] + [k for k in metric_keys if k in df.columns]
    d = df[keep].copy()
    long = d.melt(id_vars=["date", "level"], var_name="metric", value_name="value")
    return long

def metric_label(m: str) -> str:
    mapping = {
        "footfall": "Footfall",
        "turnover": "Omzet",
        "transactions": "Transacties",
        "conversion_rate": "Conversie (%)",
        "sales_per_visitor": "SPV (€ / visitor)",
        "avg_basket_size": "ATV (€)",
        "sales_per_sqm": "Sales / m² (€)",
    }
    return mapping.get(m, m)

def metric_axis_format(m: str) -> str:
    if m in ["turnover"]:
        return ",.0f"
    if m in ["footfall", "transactions"]:
        return ",.0f"
    if m in ["sales_per_visitor", "avg_basket_size", "sales_per_sqm"]:
        return ",.2f"
    if m in ["conversion_rate"]:
        return ".2f"
    return ",.2f"

def make_trend_chart(long_df: pd.DataFrame, metric: str):
    if long_df is None or long_df.empty:
        st.info("Geen trenddata beschikbaar.")
        return

    d = long_df[long_df["metric"] == metric].dropna(subset=["value"]).copy()
    if d.empty:
        st.info("Geen waarden voor deze metric.")
        return

    # color mapping: Store = purple, Region = red-ish, Company = gray
    levels = d["level"].unique().tolist()
    domain = []
    range_ = []
    for lv in ["Store", "Regio", "Company"]:
        if lv in levels:
            domain.append(lv)
            if lv == "Store":
                range_.append(PFM_PURPLE)
            elif lv == "Regio":
                range_.append(PFM_RED)
            else:
                range_.append(PFM_GRAY)

    chart = (
        alt.Chart(d)
        .mark_line(point=False, strokeWidth=2)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("value:Q", title=metric_label(metric), axis=alt.Axis(format=metric_axis_format(metric))),
            color=alt.Color("level:N", scale=alt.Scale(domain=domain, range=range_), title=None),
            tooltip=[
                alt.Tooltip("date:T", title="Datum"),
                alt.Tooltip("level:N", title="Niveau"),
                alt.Tooltip("value:Q", title=metric_label(metric), format=metric_axis_format(metric)),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)

def score_vs_benchmark(value: float, bench: float, floor_ratio=70.0, cap_ratio=130.0) -> float:
    if pd.isna(value) or pd.isna(bench) or float(bench) == 0.0:
        return np.nan
    ratio = (float(value) / float(bench)) * 100.0
    return _ratio_to_score_0_100(ratio, floor=floor_ratio, cap=cap_ratio)

# ----------------------
# MAIN
# ----------------------
def main():
    st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)

    header_left, header_right = st.columns([2.2, 1.8])

    with header_left:
        st.markdown(
            f"""
            <div class="pfm-header">
              <div>
                <div class="pfm-title">PFM Region Performance Copilot <span class="pill">v2</span> <span class="pill">+ Drilldown</span></div>
                <div class="pfm-sub">Regio upgrade + store drilldown (Store vs Regio vs Company) + metric trends</div>
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
        show_store_drilldown = st.toggle("Toon store drilldown", value=True)
    with top_controls[4]:
        run_btn = st.button("Analyseer", type="primary")

    if not run_btn:
        st.info("Selecteer retailer/regio/periode en klik op **Analyseer**.")
        return

    # use PeriodDef from helpers_periods
    start_period = periods[period_choice].start
    end_period = periods[period_choice].end
    macro_year = periods[period_choice].macro_year

    region_shops = merged[merged["region"] == region_choice].copy()
    region_shop_ids = region_shops["id"].dropna().astype(int).unique().tolist()
    if not region_shop_ids:
        st.warning(f"Geen winkels gevonden voor regio '{region_choice}'.")
        return

    all_shop_ids = merged["id"].dropna().astype(int).unique().tolist()

    # Always fetch ALL shops for correct company benchmark (SVI + drilldown comparisons)
    fetch_ids = all_shop_ids

    # Metrics (transactions key is "transactions")
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
        shop_ids=fetch_ids,
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
                shop_ids=fetch_ids,
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
                st.write("start/end:", start_period, end_period)
            return
        except requests.exceptions.RequestException as e:
            st.error(f"❌ RequestException bij /get-report: {e}")
            with st.expander("🔧 Debug request (params)"):
                st.write("REPORT_URL:", REPORT_URL)
                st.write("Params:", params_preview)
                st.write("start/end:", start_period, end_period)
            return

    df_norm = normalize_vemcount_response(resp, kpi_keys=metric_map.keys()).rename(columns=metric_map)

    if df_norm is None or df_norm.empty:
        st.warning("Geen data ontvangen voor de gekozen selectie.")
        with st.expander("🔧 Debug response keys"):
            if isinstance(resp, dict):
                st.write("Top-level keys:", list(resp.keys()))
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

    # sqm fix
    merged = enrich_merged_with_sqm_from_df_norm(merged, df_norm, store_key_col=store_key_col)

    # sqm_effective: override > locations sqm > sqm_api
    sqm_col = None
    for cand in ["sqm", "sq_meter", "sq_meters", "square_meters"]:
        if cand in merged.columns:
            sqm_col = cand
            break

    base_sqm = pd.to_numeric(merged[sqm_col], errors="coerce") if sqm_col is not None else np.nan
    sqm_api = pd.to_numeric(merged["sqm_api"], errors="coerce") if "sqm_api" in merged.columns else np.nan

    merged["sqm_effective"] = np.where(
        merged["sqm_override"].notna(),
        pd.to_numeric(merged["sqm_override"], errors="coerce"),
        np.where(pd.notna(base_sqm), base_sqm, sqm_api)
    )

    df_daily_store = collapse_to_daily_store(df_norm, store_key_col=store_key_col)
    if df_daily_store is None or df_daily_store.empty:
        st.warning("Geen data na opschonen (daily/store collapse).")
        with st.expander("🔧 Debug df_norm head"):
            st.write(df_norm.head())
        return

    join_cols = ["id", "store_display", "region", "sqm_effective"]
    if "store_type" in merged.columns:
        join_cols.append("store_type")

    df_daily_store = df_daily_store.merge(
        merged[join_cols].drop_duplicates(),
        left_on=store_key_col,
        right_on="id",
        how="left",
    )

    df_region_daily = df_daily_store[df_daily_store["region"] == region_choice].copy()
    if df_region_daily.empty:
        st.warning("Geen data voor geselecteerde regio binnen de periode.")
        with st.expander("🔧 Debug join coverage"):
            st.write("Region choice:", region_choice)
            st.write("Unique regions:", sorted(df_daily_store["region"].dropna().unique().tolist()))
            st.write(df_daily_store[[store_key_col, "id", "region", "store_display"]].head(20))
        return

    # ----------------------
    # Region KPI headline (period)
    # ----------------------
    foot_total = float(df_region_daily["footfall"].sum()) if "footfall" in df_region_daily.columns else 0.0
    turn_total = float(df_region_daily["turnover"].sum()) if "turnover" in df_region_daily.columns else 0.0
    trans_total = float(df_region_daily["transactions"].sum()) if "transactions" in df_region_daily.columns else 0.0

    conv = (trans_total / foot_total * 100.0) if foot_total > 0 else np.nan
    spv = (turn_total / foot_total) if foot_total > 0 else np.nan
    atv = (turn_total / trans_total) if trans_total > 0 else np.nan

    # ----------------------
    # Capture weekly chart data
    # ----------------------
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
    # SVI v2 region
    # ----------------------
    svi_v2 = build_region_svi_v2(
        df_daily_store=df_daily_store,
        merged=merged,
        store_key_col=store_key_col,
        region_choice=region_choice,
        capture_weekly=capture_weekly,
    )
    region_svi_v2 = svi_v2.get("region_svi", np.nan)
    comp_df = svi_v2.get("components", pd.DataFrame())

    region_status, region_color = ("-", PFM_LINE)
    if not pd.isna(region_svi_v2):
        region_status, region_color = status_from_score(region_svi_v2)

    # ----------------------
    # Opportunities v2
    # ----------------------
    opp_v2 = build_opportunities_v2_one_lever(
        df_region_daily=df_region_daily,
        store_key_col=store_key_col,
        store_name_col="store_display",
        min_days=20,
    )

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

    st.markdown("<div class='muted'>Tip: v2 gebruikt company-benchmark (zelfde periode) voor SVI-context. Capture wordt getoond als absolute waarde (benchmark volgt later als Pathzz-scope breder is).</div>", unsafe_allow_html=True)

    # ----------------------
    # Row: Weekly + SVI v2 + breakdown
    # ----------------------
    r2_a, r2_b, r2_c = st.columns([1.7, 0.75, 1.05])

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
        st.markdown('<div class="panel"><div class="panel-title">Regio SVI (v2)</div>', unsafe_allow_html=True)

        if not pd.isna(region_svi_v2):
            st.altair_chart(gauge_chart(region_svi_v2, region_color), use_container_width=True)
            st.markdown(f"**{region_svi_v2:.0f}** · {region_status}")
            st.caption("SVI v2 = gewogen score (0–100) t.o.v. company benchmark (zelfde periode).")
        else:
            st.info("Nog geen SVI v2.")
        st.markdown("</div>", unsafe_allow_html=True)

    with r2_c:
        st.markdown('<div class="panel"><div class="panel-title">SVI breakdown (waar zit de hefboom?)</div>', unsafe_allow_html=True)

        if comp_df is None or comp_df.empty or comp_df["score_0_100"].dropna().empty:
            st.info("Nog geen component breakdown.")
        else:
            b = comp_df.copy()
            b["component_name"] = b["component"].apply(nice_component_name)
            b["score"] = pd.to_numeric(b["score_0_100"], errors="coerce")
            b["ratio"] = pd.to_numeric(b["ratio_vs_benchmark"], errors="coerce")
            b = b.sort_values("score", ascending=True)

            bar = (
                alt.Chart(b)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("score:Q", title="Score (0–100)", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("component_name:N", sort="-x", title=None),
                    color=alt.value(PFM_AMBER),
                    tooltip=[
                        alt.Tooltip("component_name:N", title="Component"),
                        alt.Tooltip("score:Q", title="Score", format=".0f"),
                        alt.Tooltip("ratio:Q", title="vs benchmark (%)", format=".1f"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(bar, use_container_width=True)

            st.caption("Interpretatie: lage score = grootste hefboom. Capture krijgt later een echte benchmark zodra Pathzz breder beschikbaar is.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------
    # Opportunities v2
    # ----------------------
    st.markdown("### Biggest opportunities (v2)")

    st.markdown(
        """
        <div class='hint'>
        Dit is <b>geen forecast</b>. Dit is een <b>theoretical upside</b> (geannualiseerd vanuit jouw geselecteerde periode),
        waarbij per winkel <b>één dominante hefboom</b> wordt gesimuleerd (conversion óf SPV).
        </div>
        """,
        unsafe_allow_html=True,
    )

    opp_panel_left, opp_panel_right = st.columns([1.25, 0.75])

    with opp_panel_left:
        st.markdown('<div class="panel"><div class="panel-title">Top opportunities (1 lever per store)</div>', unsafe_allow_html=True)

        if opp_v2 is None or opp_v2.empty:
            st.info("Nog geen opportunities gevonden (of te weinig dagen data per store).")
        else:
            topn = opp_v2.sort_values("theoretical_upside_year", ascending=False).head(8).copy()
            chart_df = topn.copy()
            chart_df["upside"] = pd.to_numeric(chart_df["theoretical_upside_year"], errors="coerce")

            opp_chart = (
                alt.Chart(chart_df)
                .mark_bar(cornerRadiusEnd=4, color=PFM_RED)
                .encode(
                    x=alt.X("upside:Q", title="Theoretical upside (€ / jaar)", axis=alt.Axis(format=",.0f")),
                    y=alt.Y("store_display:N", sort="-x", title=None),
                    tooltip=[
                        alt.Tooltip("store_display:N", title="Winkel"),
                        alt.Tooltip("upside:Q", title="Upside € / jaar", format=",.0f"),
                        alt.Tooltip("driver:N", title="Driver"),
                        alt.Tooltip("confidence:N", title="Confidence"),
                        alt.Tooltip("annualised_from_days:Q", title="Annualised from (#days)"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(opp_chart, use_container_width=True)

            total_top5 = float(topn["theoretical_upside_year"].head(5).sum())
            st.markdown(f"**Top 5 samen (theoretical, annualised):** {fmt_eur(total_top5)} / jaar")

            st.caption("• Per winkel: óf conversion uplift naar regio-median óf SPV uplift naar regio-median.")
            st.caption("• Upside berekend op huidig volume (footfall) en geannualiseerd vanuit de gekozen periode.")

        st.markdown("</div>", unsafe_allow_html=True)

    with opp_panel_right:
        st.markdown('<div class="panel"><div class="panel-title">Reality check</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class='muted'>
            Waarom dit wél bruikbaar is voor een regiomanager:
            <ul>
              <li>Het wijst <b>prioriteit</b> aan (waar zit de hefboom?)</li>
              <li>Het is <b>conservatief</b>: per winkel slechts één driver</li>
              <li>Het is <b>herhaalbaar</b>: zelfde logica bij elke periode</li>
            </ul>
            Wat het niet is:
            <ul>
              <li>Geen guarantee / forecast</li>
              <li>Geen “alles tegelijk fixen” scenario</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------
    # NEW: STORE DRILLDOWN (v2)
    # ----------------------
    if show_store_drilldown:
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("## Store drilldown (v2)")

        # Stores in this region
        store_list = (
            merged[merged["region"] == region_choice][["id", "store_display"]]
            .dropna(subset=["id"])
            .drop_duplicates()
            .sort_values("store_display")
        )
        store_options = store_list.apply(lambda r: f"{r['store_display']} (ID {int(r['id'])})", axis=1).tolist()
        default_idx = 0 if store_options else None

        dcol1, dcol2, dcol3 = st.columns([1.6, 1.2, 1.2])
        with dcol1:
            store_choice_label = st.selectbox("Selecteer winkel", store_options, index=default_idx if default_idx is not None else 0)
        with dcol2:
            metric_choices = st.multiselect(
                "Metrics",
                ["footfall", "turnover", "transactions", "conversion_rate", "sales_per_visitor", "avg_basket_size", "sales_per_sqm"],
                default=["turnover", "conversion_rate", "sales_per_visitor"],
            )
        with dcol3:
            show_best_worst = st.toggle("Toon best/worst days", value=True)

        if not store_options:
            st.info("Geen stores gevonden voor drilldown in deze regio.")
        else:
            # Parse store_id from label
            # label format: "<name> (ID 123)"
            try:
                store_id = int(store_choice_label.split("(ID")[-1].replace(")", "").strip())
            except Exception:
                store_id = int(store_list["id"].iloc[0])

            df_store_daily = df_daily_store[df_daily_store[store_key_col].astype("Int64") == store_id].copy()
            df_store_daily = df_store_daily.merge(
                merged[["id", "store_display", "region", "sqm_effective"]].drop_duplicates(),
                left_on=store_key_col,
                right_on="id",
                how="left",
            )

            if df_store_daily.empty:
                st.warning("Geen store-data gevonden in de gekozen periode.")
            else:
                # Company slice: all stores (already in df_daily_store)
                df_company_daily = df_daily_store.copy()

                # Ensure region exists for all
                try:
                    df_company_daily = ensure_region_column(df_company_daily, merged, store_key_col)
                except Exception:
                    pass

                # Region slice: selected region
                df_region_daily_2 = df_company_daily[df_company_daily["region"] == region_choice].copy()

                # Enrich sqm for company/region/store (needed for daily_aggregate sales/m²)
                # company/region already have sqm_effective because df_daily_store merge earlier
                # store also has sqm_effective from merge

                # Period KPIs
                store_kpis = compute_period_kpis(df_store_daily, sqm_sum_mode="single")
                region_kpis = compute_period_kpis(df_region_daily_2, sqm_sum_mode="unique")
                company_kpis = compute_period_kpis(df_company_daily, sqm_sum_mode="unique")

                st.markdown(f"### {store_kpis.get('store_display', '')}".strip())

                # KPI cards: Store vs Region vs Company (same row)
                cA, cB, cC = st.columns([1, 1, 1])
                with cA:
                    st.markdown('<div class="panel"><div class="panel-title">Store (periode)</div>', unsafe_allow_html=True)
                    kpi_card("Footfall", fmt_int(store_kpis.get("footfall", np.nan)), "Store · periode")
                    kpi_card("Omzet", fmt_eur(store_kpis.get("turnover", np.nan)), "Store · periode")
                    kpi_card("Conversie", fmt_pct(store_kpis.get("conversion_rate", np.nan)), "Transacties / bezoekers")
                    kpi_card("ATV", fmt_eur_2(store_kpis.get("avg_basket_size", np.nan)), "Omzet / transactie")
                    kpi_card("Sales / m²", fmt_eur_2(store_kpis.get("sales_per_sqm", np.nan)), "Omzet / m²")
                    st.markdown("</div>", unsafe_allow_html=True)

                with cB:
                    st.markdown('<div class="panel"><div class="panel-title">Regio benchmark</div>', unsafe_allow_html=True)
                    kpi_card("Footfall", fmt_int(region_kpis.get("footfall", np.nan)), "Regio · periode")
                    kpi_card("Omzet", fmt_eur(region_kpis.get("turnover", np.nan)), "Regio · periode")
                    kpi_card("Conversie", fmt_pct(region_kpis.get("conversion_rate", np.nan)), "Transacties / bezoekers")
                    kpi_card("ATV", fmt_eur_2(region_kpis.get("avg_basket_size", np.nan)), "Omzet / transactie")
                    kpi_card("Sales / m²", fmt_eur_2(region_kpis.get("sales_per_sqm", np.nan)), "Omzet / m²")
                    st.markdown("</div>", unsafe_allow_html=True)

                with cC:
                    st.markdown('<div class="panel"><div class="panel-title">Company benchmark</div>', unsafe_allow_html=True)
                    kpi_card("Footfall", fmt_int(company_kpis.get("footfall", np.nan)), "Company · periode")
                    kpi_card("Omzet", fmt_eur(company_kpis.get("turnover", np.nan)), "Company · periode")
                    kpi_card("Conversie", fmt_pct(company_kpis.get("conversion_rate", np.nan)), "Transacties / bezoekers")
                    kpi_card("ATV", fmt_eur_2(company_kpis.get("avg_basket_size", np.nan)), "Omzet / transactie")
                    kpi_card("Sales / m²", fmt_eur_2(company_kpis.get("sales_per_sqm", np.nan)), "Omzet / m²")
                    st.markdown("</div>", unsafe_allow_html=True)

                # Quick deltas / lever scan
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.markdown("### Lever scan — waar zit het gat?")

                # Scores vs Region + Company
                lever_rows = []
                for m in ["conversion_rate", "avg_basket_size", "sales_per_visitor", "sales_per_sqm"]:
                    v_store = store_kpis.get(m, np.nan)
                    v_reg = region_kpis.get(m, np.nan)
                    v_cmp = company_kpis.get(m, np.nan)

                    score_reg = score_vs_benchmark(v_store, v_reg)
                    score_cmp = score_vs_benchmark(v_store, v_cmp)

                    # "gap" as percent vs benchmark
                    gap_reg = ((v_store / v_reg) * 100.0 - 100.0) if (pd.notna(v_store) and pd.notna(v_reg) and v_reg != 0) else np.nan
                    gap_cmp = ((v_store / v_cmp) * 100.0 - 100.0) if (pd.notna(v_store) and pd.notna(v_cmp) and v_cmp != 0) else np.nan

                    lever_rows.append({
                        "metric": metric_label(m),
                        "store": v_store,
                        "region": v_reg,
                        "company": v_cmp,
                        "gap_vs_region_%": gap_reg,
                        "gap_vs_company_%": gap_cmp,
                        "score_vs_region": score_reg,
                        "score_vs_company": score_cmp,
                    })

                lever_df = pd.DataFrame(lever_rows)

                l1, l2 = st.columns([1.1, 1.2])
                with l1:
                    st.markdown('<div class="panel"><div class="panel-title">Scores (0–100)</div>', unsafe_allow_html=True)
                    # altair bar: lowest scores first (biggest lever)
                    tmp = lever_df.copy()
                    tmp["weakness"] = tmp["score_vs_region"]
                    tmp = tmp.dropna(subset=["weakness"])
                    if tmp.empty:
                        st.info("Onvoldoende data om scores te berekenen.")
                    else:
                        tmp = tmp.sort_values("weakness", ascending=True)
                        bar = (
                            alt.Chart(tmp)
                            .mark_bar(cornerRadiusEnd=4)
                            .encode(
                                x=alt.X("weakness:Q", title="Score vs Regio (0–100)", scale=alt.Scale(domain=[0, 100])),
                                y=alt.Y("metric:N", sort="-x", title=None),
                                color=alt.value(PFM_AMBER),
                                tooltip=[
                                    alt.Tooltip("metric:N", title="Metric"),
                                    alt.Tooltip("weakness:Q", title="Score vs Regio", format=".0f"),
                                    alt.Tooltip("score_vs_company:Q", title="Score vs Company", format=".0f"),
                                    alt.Tooltip("gap_vs_region_%:Q", title="Gap vs Regio (%)", format=".1f"),
                                    alt.Tooltip("gap_vs_company_%:Q", title="Gap vs Company (%)", format=".1f"),
                                ],
                            )
                            .properties(height=240)
                        )
                        st.altair_chart(bar, use_container_width=True)

                    st.caption("Lager = grotere hefboom. Scores zijn ratio-gebaseerd en geclipped (70–130% → 0–100).")
                    st.markdown("</div>", unsafe_allow_html=True)

                with l2:
                    st.markdown('<div class="panel"><div class="panel-title">Store vs Regio vs Company (tabel)</div>', unsafe_allow_html=True)
                    t = lever_df.copy()
                    # pretty formatting
                    def _fmt_val(metric_name, v):
                        if "Conversie" in metric_name:
                            return fmt_pct(v)
                        if "ATV" in metric_name or "Sales / m²" in metric_name or "SPV" in metric_name:
                            return fmt_eur_2(v)
                        return str(v)

                    t["Store"] = t.apply(lambda r: _fmt_val(r["metric"], r["store"]), axis=1)
                    t["Regio"] = t.apply(lambda r: _fmt_val(r["metric"], r["region"]), axis=1)
                    t["Company"] = t.apply(lambda r: _fmt_val(r["metric"], r["company"]), axis=1)
                    t["Gap vs Regio"] = t["gap_vs_region_%"].apply(lambda x: "-" if pd.isna(x) else f"{x:+.1f}%".replace(".", ","))
                    t["Gap vs Company"] = t["gap_vs_company_%"].apply(lambda x: "-" if pd.isna(x) else f"{x:+.1f}%".replace(".", ","))
                    t = t[["metric", "Store", "Regio", "Company", "Gap vs Regio", "Gap vs Company"]].rename(columns={"metric": "Metric"})
                    st.dataframe(t, use_container_width=True, hide_index=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # Trend charts
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.markdown("### Trends — Store vs Regio vs Company (daily)")

                store_ts = daily_aggregate(df_store_daily, "Store", store_key_col=store_key_col)
                region_ts = daily_aggregate(df_region_daily_2, "Regio", store_key_col=store_key_col)
                company_ts = daily_aggregate(df_company_daily, "Company", store_key_col=store_key_col)

                ts = pd.concat([store_ts, region_ts, company_ts], ignore_index=True)
                long = melt_metrics(ts, metric_choices)

                tcol1, tcol2 = st.columns([1, 1])
                left_metrics = metric_choices[0::2]
                right_metrics = metric_choices[1::2]

                with tcol1:
                    st.markdown('<div class="panel"><div class="panel-title">Trend charts (1)</div>', unsafe_allow_html=True)
                    if not left_metrics:
                        st.info("Kies minimaal één metric.")
                    else:
                        for m in left_metrics:
                            st.markdown(f"**{metric_label(m)}**")
                            make_trend_chart(long, m)
                    st.markdown("</div>", unsafe_allow_html=True)

                with tcol2:
                    st.markdown('<div class="panel"><div class="panel-title">Trend charts (2)</div>', unsafe_allow_html=True)
                    if not right_metrics:
                        st.info("Kies extra metrics links.")
                    else:
                        for m in right_metrics:
                            st.markdown(f"**{metric_label(m)}**")
                            make_trend_chart(long, m)
                    st.markdown("</div>", unsafe_allow_html=True)

                # Best/Worst days (optional)
                if show_best_worst:
                    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                    st.markdown("### Best / worst days (store)")

                    s = store_ts.copy()
                    if s.empty:
                        st.info("Geen store timeseries.")
                    else:
                        # pick a “performance metric” for ranking (default: turnover)
                        rank_metric = "turnover" if "turnover" in s.columns else "sales_per_visitor"
                        s2 = s[["date", rank_metric, "footfall", "transactions", "conversion_rate", "sales_per_visitor", "avg_basket_size", "sales_per_sqm"]].copy()
                        s2 = s2.dropna(subset=[rank_metric])
                        if s2.empty:
                            st.info("Onvoldoende data om best/worst dagen te tonen.")
                        else:
                            best = s2.sort_values(rank_metric, ascending=False).head(5).copy()
                            worst = s2.sort_values(rank_metric, ascending=True).head(5).copy()

                            def _fmt_row(df_):
                                out = df_.copy()
                                out["date"] = out["date"].dt.date.astype(str)
                                out["turnover"] = out["turnover"].apply(fmt_eur) if "turnover" in out.columns else out.get("turnover", "-")
                                out["footfall"] = out["footfall"].apply(fmt_int) if "footfall" in out.columns else out.get("footfall", "-")
                                out["transactions"] = out["transactions"].apply(fmt_int) if "transactions" in out.columns else out.get("transactions", "-")
                                out["conversion_rate"] = out["conversion_rate"].apply(fmt_pct) if "conversion_rate" in out.columns else out.get("conversion_rate", "-")
                                out["sales_per_visitor"] = out["sales_per_visitor"].apply(fmt_eur_2) if "sales_per_visitor" in out.columns else out.get("sales_per_visitor", "-")
                                out["avg_basket_size"] = out["avg_basket_size"].apply(fmt_eur_2) if "avg_basket_size" in out.columns else out.get("avg_basket_size", "-")
                                out["sales_per_sqm"] = out["sales_per_sqm"].apply(fmt_eur_2) if "sales_per_sqm" in out.columns else out.get("sales_per_sqm", "-")
                                cols = ["date", "turnover", "footfall", "transactions", "conversion_rate", "sales_per_visitor", "avg_basket_size", "sales_per_sqm"]
                                cols = [c for c in cols if c in out.columns]
                                return out[cols]

                            b1, b2 = st.columns(2)
                            with b1:
                                st.markdown('<div class="panel"><div class="panel-title">Top 5 days</div>', unsafe_allow_html=True)
                                st.dataframe(_fmt_row(best), use_container_width=True, hide_index=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                            with b2:
                                st.markdown('<div class="panel"><div class="panel-title">Bottom 5 days</div>', unsafe_allow_html=True)
                                st.dataframe(_fmt_row(worst), use_container_width=True, hide_index=True)
                                st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------
    # Macro context (CBS/CCI)
    # ----------------------
    if show_macro:
        st.markdown("## Macro-context (CBS/CCI)")

        macro_start = start_period - timedelta(days=365)
        macro_end = end_period

        st.caption(f"Macro toont: {macro_start} → {macro_end} (1 jaar terug vanaf start van je periode t/m einddatum).")

        macro_metric_map = {"count_in": "footfall", "turnover": "turnover", "transactions": "transactions"}

        macro_params_preview = build_report_params(
            shop_ids=region_shop_ids,
            data_outputs=list(macro_metric_map.keys()),
            period="date",
            step="day",
            source="shops",
            date_from=macro_start,
            date_to=macro_end,
        )

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
                with st.expander("🔧 Debug macro request (params)"):
                    st.write("REPORT_URL:", REPORT_URL)
                    st.write("Params:", macro_params_preview)

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

        months_back = int(((macro_end.year - macro_start.year) * 12 + (macro_end.month - macro_start.month)) + 6)
        months_back = max(60, months_back)
        months_back = min(240, months_back)

        # CBS retail
        cbs_retail_month = pd.DataFrame()
        try:
            retail_series = get_retail_index(months_back=months_back)
        except Exception:
            retail_series = []

        if retail_series:
            raw = pd.DataFrame(retail_series)
            value_col = None
            for cand in ["retail_value", "value", "index", "retail_index"]:
                if cand in raw.columns:
                    value_col = cand
                    break
            period_col = None
            for cand in ["period", "periode", "month", "maand"]:
                if cand in raw.columns:
                    period_col = cand
                    break

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
                    cbs_retail_month = raw.groupby("date", as_index=False)[value_col].mean()
                    cbs_retail_month = cbs_retail_month.rename(columns={value_col: "retail_value"})
                    base = cbs_retail_month["retail_value"].replace(0, np.nan).dropna()
                    if not base.empty:
                        base = float(base.iloc[0])
                        cbs_retail_month["cbs_retail_index"] = (cbs_retail_month["retail_value"] / base) * 100.0

        # CCI
        cci_df = pd.DataFrame()
        try:
            cci_series = get_cci_series(months_back=months_back)
        except Exception:
            cci_series = []

        if cci_series:
            raw = pd.DataFrame(cci_series)
            value_col = None
            for cand in ["cci", "value", "index", "cci_value"]:
                if cand in raw.columns:
                    value_col = cand
                    break
            period_col = None
            for cand in ["period", "periode", "month", "maand"]:
                if cand in raw.columns:
                    period_col = cand
                    break

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
                        base = float(base.iloc[0])
                        cci_df["cci_index"] = (cci_df["cci"] / base) * 100.0

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

        if not cbs_retail_month.empty and "cbs_retail_index" in cbs_retail_month.columns:
            c = cbs_retail_month[["date", "cbs_retail_index"]].copy()
            c["series"] = "CBS detailhandelindex"
            c = c.rename(columns={"cbs_retail_index": "value"})
            macro_lines.append(c)

        macro_df_cbs = pd.concat(macro_lines, ignore_index=True) if macro_lines else pd.DataFrame()

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

        if not cci_df.empty and "cci_index" in cci_df.columns:
            c = cci_df[["date", "cci_index"]].copy()
            c["series"] = "CCI consumentenvertrouwen"
            c = c.rename(columns={"cci_index": "value"})
            macro_lines.append(c)

        macro_df_cci = pd.concat(macro_lines, ignore_index=True) if macro_lines else pd.DataFrame()

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
    # Quadrant (SVI vs Performance proxy)
    # ----------------------
    if show_quadrant:
        st.markdown("## Regio quadrant (v2)")

        regs = sorted(merged["region"].dropna().unique().tolist())
        rows = []

        for r in regs:
            drr = df_daily_store[df_daily_store["region"] == r].copy()
            if drr.empty:
                continue

            dummy_capture = pd.DataFrame()
            out = build_region_svi_v2(
                df_daily_store=df_daily_store,
                merged=merged,
                store_key_col=store_key_col,
                region_choice=r,
                capture_weekly=dummy_capture,
            )
            svi = out.get("region_svi", np.nan)

            bench = out.get("benchmarks", {}).get("company", {})
            regb = out.get("benchmarks", {}).get("region", {})
            reg_spv = regb.get("sales_per_visitor", np.nan)
            cmp_spv = bench.get("sales_per_visitor", np.nan)
            rel = (reg_spv / cmp_spv * 100.0) if (pd.notna(reg_spv) and pd.notna(cmp_spv) and cmp_spv != 0) else np.nan

            rows.append({
                "region": r,
                "svi_v2": svi,
                "perf_vs_company_spv": rel,
                "is_selected": (r == region_choice),
            })

        quad = pd.DataFrame(rows)
        if quad.empty or quad["svi_v2"].dropna().empty:
            st.info("Nog onvoldoende data voor quadrant.")
        else:
            quad["x"] = pd.to_numeric(quad["svi_v2"], errors="coerce")
            quad["y"] = pd.to_numeric(quad["perf_vs_company_spv"], errors="coerce")

            st.caption("Y-as is een pragmatische proxy: SPV (sales/visitor) vs company benchmark (100 = gelijk).")

            chart = (
                alt.Chart(quad.dropna(subset=["x", "y"]))
                .mark_circle(size=220, opacity=0.9)
                .encode(
                    x=alt.X("x:Q", title="SVI v2 (0–100)", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("y:Q", title="Performance vs company (SPV index)", axis=alt.Axis(format=".0f")),
                    color=alt.Color(
                        "is_selected:N",
                        scale=alt.Scale(domain=[True, False], range=[PFM_PURPLE, PFM_LINE]),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("region:N", title="Regio"),
                        alt.Tooltip("x:Q", title="SVI v2", format=".0f"),
                        alt.Tooltip("y:Q", title="SPV index", format=".0f"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)

    # ----------------------
    # Debug
    # ----------------------
    with st.expander("🔧 Debug (v2 + drilldown)"):
        st.write("Retailer:", selected_client)
        st.write("Regio:", region_choice)
        st.write("Periode:", start_period, "→", end_period, f"({period_choice})")
        st.write("Macro year:", macro_year)
        st.write("Store key col:", store_key_col)
        st.write("All shops:", len(all_shop_ids), "Region shops:", len(region_shop_ids))
        st.write("REPORT_URL:", REPORT_URL)
        st.write("Params used:", params_preview)
        st.write("Merged columns:", merged.columns.tolist())
        st.write("df_norm columns:", df_norm.columns.tolist())
        st.write("df_daily_store head:", df_daily_store.head())
        st.subheader("SVI v2 components (raw)")
        st.dataframe(comp_df)
        st.subheader("Opportunities v2 (raw)")
        st.dataframe(opp_v2.head(20))


if __name__ == "__main__":
    main()
