# pages/07A_Region_Reasoner.py
# ------------------------------------------------------------
# PFM Region Reasoner — Agentic Outcome Edition
# Outcome-first, apples-to-apples, deterministic before LLM
# ------------------------------------------------------------

from __future__ import annotations

import os
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import requests

from datetime import datetime

# --- helpers / services (existing in repo) ---
from helpers_clients import load_clients
from helpers_periods import period_catalog
from helpers_vemcount_api import (
    VemcountApiConfig,
    fetch_report,
    build_report_params,
)
from helpers_normalize import normalize_vemcount_response

from services.svi_service import (
    compute_driver_values_from_period,
    compute_svi_explainable,
    BASE_SVI_WEIGHTS,
    get_svi_weights_for_store_type,
)

from services.value_upside import estimate_upside
from services.advisor import make_actions

from outcome_explainer import OutcomeExplainer


# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="PFM Region Reasoner — Agentic",
    layout="wide"
)

# ------------------------------------------------------------
# Formatting helpers (EU)
# ------------------------------------------------------------
def fmt_eur(x):
    if pd.isna(x):
        return "-"
    return f"€ {x:,.0f}".replace(",", ".")

def fmt_pct(x, d=0):
    if pd.isna(x):
        return "-"
    return f"{x:.{d}f}%".replace(".", ",")

def fmt_int(x):
    if pd.isna(x):
        return "-"
    return f"{int(x):,}".replace(",", ".")

# ------------------------------------------------------------
# Region mapping
# ------------------------------------------------------------
@st.cache_data(ttl=600)
def load_region_mapping(path="data/regions.csv") -> pd.DataFrame:
    """
    Expected:
    shop_id;region;sqm_override;store_type
    """
    df = pd.read_csv(path, sep=";")
    df["shop_id"] = pd.to_numeric(df["shop_id"], errors="coerce").astype("Int64")
    df["region"] = df["region"].astype(str).str.strip().str.lower()
    df["sqm_override"] = pd.to_numeric(df.get("sqm_override"), errors="coerce")
    df["store_type"] = (
        df.get("store_type", "Unknown")
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )
    return df.dropna(subset=["shop_id"])


# ------------------------------------------------------------
# Outcome builder (DETERMINISTIC)
# ------------------------------------------------------------
def build_region_outcomes(payload: dict) -> dict:
    df_region = payload["df_region"]
    store_dim = payload["store_dim"]
    meta = payload["meta"]
    scores = payload["scores"]

    # --- Opportunities ---
    opp_rows = []
    for _, r in payload["store_summary"].iterrows():
        up, driver = estimate_upside(r, payload["company_type_bench"].get(r["store_type"], {}))
        if pd.notna(up) and up > 0:
            opp_rows.append({
                "store_name": r["store_display"],
                "store_type": r["store_type"],
                "driver": driver,
                "impact_period_eur": up,
                "impact_annual_eur": up * 52 / payload["weeks"],
                "evidence": [
                    f"Traffic idx: {fmt_pct(r['Traffic idx vs type'])}",
                    f"CR idx: {fmt_pct(r['CR idx vs type'])}",
                    f"Sales/m² idx: {fmt_pct(r['Sales/m² idx vs type'])}",
                ],
                "actions": make_actions(r),
            })

    opportunities = sorted(
        opp_rows,
        key=lambda x: x["impact_period_eur"],
        reverse=True
    )[:3]

    # --- Risks ---
    risks = []
    if (payload["region_svi"] < 60):
        risks.append({
            "driver": "Low regional SVI",
            "severity": "high",
            "why": [
                f"Region SVI {int(payload['region_svi'])}/100",
                "Multiple stores underperform store_type benchmark"
            ],
            "actions": [
                {"label": "Review", "text": "Focus on lowest SVI stores first"}
            ]
        })

    return {
        "meta": meta,
        "scores": scores,
        "opportunities": opportunities,
        "risks": risks,
        "notes": [
            "Benchmarked on same store_type",
            "sqm-calibrated (override → sq_meter)"
        ]
    }


# ------------------------------------------------------------
# Render Outcome Feed
# ------------------------------------------------------------
def render_outcome_feed(outcomes: dict, typing: bool = True):
    explainer = OutcomeExplainer()

    cards = explainer.build_cards(
        outcomes,
        persona="region_manager",
        style="crisp"
    )

    st.markdown("## Agentic outcome feed")

    for card in cards:
        box = st.empty()
        if typing:
            txt = ""
            for chunk in explainer.stream_typing(card["body"]):
                txt = chunk
                box.markdown(
                    f"### {card['title']}\n\n{txt}",
                    unsafe_allow_html=True
                )
        else:
            box.markdown(
                f"### {card['title']}\n\n{card['body']}",
                unsafe_allow_html=True
            )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    # --------------------
    # Selectors
    # --------------------
    clients = load_clients("clients.json")
    client = st.selectbox(
        "Client",
        clients,
        format_func=lambda c: f"{c['brand']} ({c['company_id']})"
    )

    periods = period_catalog(today=datetime.now().date())
    period_label = st.selectbox("Period", list(periods.keys()))
    period = periods[period_label]

    region_map = load_region_mapping()
    region = st.selectbox(
        "Region",
        sorted(region_map["region"].unique())
    )

    run = st.button("Run analysis", type="primary")
    if not run:
        st.stop()

    # --------------------
    # Fetch data
    # --------------------
    shop_ids = region_map["shop_id"].astype(int).tolist()

    cfg = VemcountApiConfig(
        report_url=st.secrets["API_URL"]
    )

    resp = fetch_report(
        cfg=cfg,
        shop_ids=shop_ids,
        data_outputs=[
            "turnover",
            "count_in",
            "transactions",
            "sales_per_sqm",
        ],
        period="date",
        step="day",
        source="shops",
        date_from=period.start,
        date_to=period.end,
    )

    df = normalize_vemcount_response(resp)
    if df.empty:
        st.warning("No data returned")
        st.stop()

    # --------------------
    # Join region + sqm
    # --------------------
    df["shop_id"] = pd.to_numeric(df["shop_id"], errors="coerce").astype("Int64")
    df = df.merge(
        region_map,
        on="shop_id",
        how="inner"
    )

    # --- sqm_effective (WORKING LOGIC) ---
    base_sqm = pd.to_numeric(df.get("sq_meter"), errors="coerce")
    df["sqm_effective"] = df["sqm_override"].combine_first(base_sqm)

    # --- sales_per_sqm fallback ---
    if "sales_per_sqm" not in df.columns:
        df["sales_per_sqm"] = np.nan

    df["sales_per_sqm"] = pd.to_numeric(df["sales_per_sqm"], errors="coerce")
    df["sales_per_sqm"] = df["sales_per_sqm"].combine_first(
        df["turnover"] / df["sqm_effective"]
    )

    # --------------------
    # Region subset
    # --------------------
    df_region = df[df["region"] == region].copy()

    # --------------------
    # Store summary
    # --------------------
    store_summary = (
        df_region
        .groupby(["shop_id", "store_type", "sqm_effective"], as_index=False)
        .agg(
            turnover=("turnover", "sum"),
            footfall=("count_in", "sum"),
            transactions=("transactions", "sum"),
            sales_per_sqm=("sales_per_sqm", "mean"),
        )
    )

    store_summary["sales_per_visitor"] = store_summary["turnover"] / store_summary["footfall"]
    store_summary["conversion_rate"] = store_summary["transactions"] / store_summary["footfall"] * 100
    store_summary["store_display"] = store_summary["shop_id"].astype(str)

    # --------------------
    # Company store_type benchmarks
    # --------------------
    company_type_bench = {}
    for stype, g in df.groupby("store_type"):
        company_type_bench[stype] = {
            "sales_per_visitor": g["turnover"].sum() / g["count_in"].sum(),
            "conversion_rate": g["transactions"].sum() / g["count_in"].sum() * 100,
            "sales_per_sqm": g["turnover"].sum() / g["sqm_effective"].sum(),
        }

    # --------------------
    # Region SVI
    # --------------------
    region_vals = compute_driver_values_from_period(
        footfall=df_region["count_in"].sum(),
        turnover=df_region["turnover"].sum(),
        transactions=df_region["transactions"].sum(),
        sqm_sum=df_region["sqm_effective"].sum(),
        capture_pct=np.nan,
    )

    comp_vals = compute_driver_values_from_period(
        footfall=df["count_in"].sum(),
        turnover=df["turnover"].sum(),
        transactions=df["transactions"].sum(),
        sqm_sum=df["sqm_effective"].sum(),
        capture_pct=np.nan,
    )

    region_svi, avg_ratio, _ = compute_svi_explainable(
        vals_a=region_vals,
        vals_b=comp_vals,
        weights=BASE_SVI_WEIGHTS,
        floor=80,
        cap=120,
    )

    # --------------------
    # Payload
    # --------------------
    payload = {
        "df_region": df_region,
        "store_dim": region_map,
        "store_summary": store_summary,
        "company_type_bench": company_type_bench,
        "region_svi": region_svi,
        "weeks": max(1, (period.end - period.start).days / 7),
        "meta": {
            "client": client["brand"],
            "region": region,
            "period_label": period_label,
        },
        "scores": {
            "region_svi": region_svi,
            "avg_ratio_vs_company": avg_ratio,
        }
    }

    # --------------------
    # Render outcomes
    # --------------------
    outcomes = build_region_outcomes(payload)
    render_outcome_feed(outcomes, typing=True)


if __name__ == "__main__":
    main()