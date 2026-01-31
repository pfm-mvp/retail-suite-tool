# pages/07B_Region_Reasoner_AI.py
# ------------------------------------------------------------
# PFM Region Reasoner — Agentic Outcome Edition (KNALLER v1)
#
# Adds:
# - Region SVI vs other regions (rank + table)
# - Store-type performance vs other regions (same store_type, excluding current region)
# - Upside aggregated by store_type + top stores
# - Strengths: best store_types + best stores (copy what works)
#
# Fixes:
# - OutcomeExplainer import robust (repo root on sys.path)
# - sqm_effective: sqm_override overrides, else sq_meter from report payload
# - typing loop indentation fixed
# - avoids duplicate debug columns in st.write
# ------------------------------------------------------------

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import streamlit as st
import requests
from pathlib import Path
from datetime import datetime

# ------------------------------------------------------------
# Ensure repo root is on sys.path (for outcome_explainer.py in root)
# pages/.. -> repo root
# ------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
)

# Optional: advisor may not exist in all repos → safe import
try:
    from services.advisor import make_actions
except Exception:
    make_actions = None

# Optional: LLM explainer wrapper (root/outcome_explainer.py)
try:
    from outcome_explainer import OutcomeExplainer
except Exception:
    OutcomeExplainer = None


# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(page_title="PFM Region Reasoner — Agentic", layout="wide")


# ------------------------------------------------------------
# Formatting helpers (EU)
# ------------------------------------------------------------
def fmt_eur(x):
    if pd.isna(x):
        return "-"
    return f"€ {float(x):,.0f}".replace(",", ".")

def fmt_pct(x, d=0):
    if pd.isna(x):
        return "-"
    return f"{float(x):.{d}f}%".replace(".", ",")

def safe_div(a, b):
    try:
        if pd.isna(a) or pd.isna(b) or float(b) == 0.0:
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan

def idx_vs(a, b):
    return (a / b * 100.0) if (pd.notna(a) and pd.notna(b) and float(b) != 0.0) else np.nan


# ------------------------------------------------------------
# Region mapping
# ------------------------------------------------------------
@st.cache_data(ttl=600)
def load_region_mapping(path="data/regions.csv") -> pd.DataFrame:
    """
    Expected (semicolon separated):
    shop_id;region;sqm_override;store_type
    """
    try:
        df = pd.read_csv(path, sep=";")
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()
    if "shop_id" not in df.columns or "region" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["shop_id"] = pd.to_numeric(df["shop_id"], errors="coerce").astype("Int64")
    df["region"] = df["region"].astype(str).str.strip().str.lower()

    if "sqm_override" in df.columns:
        df["sqm_override"] = pd.to_numeric(df["sqm_override"], errors="coerce")
    else:
        df["sqm_override"] = np.nan

    if "store_type" in df.columns:
        df["store_type"] = (
            df["store_type"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
        )
    else:
        df["store_type"] = "Unknown"

    return df.dropna(subset=["shop_id"]).copy()


# ------------------------------------------------------------
# FastAPI locations endpoint (names; sometimes sqm not included)
# ------------------------------------------------------------
@st.cache_data(ttl=600)
def get_locations_by_company(fastapi_base_url: str, company_id: int) -> pd.DataFrame:
    url = f"{fastapi_base_url.rstrip('/')}/company/{company_id}/location"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "locations" in data:
        return pd.DataFrame(data["locations"])
    return pd.DataFrame(data)


# ------------------------------------------------------------
# Benchmarks
# ------------------------------------------------------------
def compute_company_type_bench(df_all: pd.DataFrame) -> dict:
    """
    Bench per store_type across ALL rows in df_all.
    df_all should have:
      id, store_type, footfall, turnover, transactions, sqm_effective
    """
    out = {}
    if df_all is None or df_all.empty:
        return out

    tmp = df_all.copy()
    if "store_type" not in tmp.columns:
        tmp["store_type"] = "Unknown"

    tmp["store_type"] = (
        tmp["store_type"]
        .fillna("Unknown")
        .astype(str).str.strip()
        .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    )

    for stype, g in tmp.groupby("store_type"):
        foot = pd.to_numeric(g.get("footfall"), errors="coerce").dropna().sum()
        turn = pd.to_numeric(g.get("turnover"), errors="coerce").dropna().sum()
        trans = pd.to_numeric(g.get("transactions"), errors="coerce").dropna().sum()

        # unique sqm per store (avoid daily duplication)
        sqm_sum = np.nan
        if "id" in g.columns:
            sqm = (
                g[["id", "sqm_effective"]]
                .dropna(subset=["id"])
                .drop_duplicates(subset=["id"])
            )
            sqm_sum = pd.to_numeric(sqm["sqm_effective"], errors="coerce").dropna().sum()
        else:
            sqm_sum = pd.to_numeric(g.get("sqm_effective"), errors="coerce").dropna().sum()

        out[stype] = {
            "footfall": float(foot),
            "turnover": float(turn),
            "transactions": float(trans),
            "sales_per_visitor": safe_div(turn, foot),
            "conversion_rate": safe_div(trans, foot) * 100.0 if pd.notna(foot) else np.nan,
            "sales_per_transaction": safe_div(turn, trans),
            "sales_per_sqm": safe_div(turn, sqm_sum),
            "sqm_sum": float(sqm_sum) if pd.notna(sqm_sum) else np.nan,
        }
    return out


# ------------------------------------------------------------
# Upside estimator (deterministic)
# ------------------------------------------------------------
def estimate_upside(store_row: pd.Series, bench_vals: dict) -> tuple[float, str]:
    """
    Upside vs same store_type benchmark:
      - Low SPV
      - Low Sales/m²
      - Low Conversion (uses store ATV if possible)
    """
    foot = pd.to_numeric(store_row.get("footfall", np.nan), errors="coerce")
    turn = pd.to_numeric(store_row.get("turnover", np.nan), errors="coerce")
    trans = pd.to_numeric(store_row.get("transactions", np.nan), errors="coerce")
    sqm = pd.to_numeric(store_row.get("sqm_effective", np.nan), errors="coerce")

    spv_s = safe_div(turn, foot)
    spm2_s = safe_div(turn, sqm)
    cr_s = safe_div(trans, foot) * 100.0 if pd.notna(trans) and pd.notna(foot) else np.nan
    atv_s = safe_div(turn, trans)

    spv_b = bench_vals.get("sales_per_visitor", np.nan)
    spm2_b = bench_vals.get("sales_per_sqm", np.nan)
    cr_b = bench_vals.get("conversion_rate", np.nan)
    atv_b = bench_vals.get("sales_per_transaction", np.nan)
    atv_use = atv_s if pd.notna(atv_s) else atv_b

    candidates = []

    if pd.notna(foot) and foot > 0 and pd.notna(spv_s) and pd.notna(spv_b) and spv_s < spv_b:
        candidates.append(("Low SPV vs type", float(foot) * float(spv_b - spv_s)))

    if pd.notna(sqm) and sqm > 0 and pd.notna(spm2_s) and pd.notna(spm2_b) and spm2_s < spm2_b:
        candidates.append(("Low Sales/m² vs type", float(sqm) * float(spm2_b - spm2_s)))

    if pd.notna(foot) and foot > 0 and pd.notna(cr_s) and pd.notna(cr_b) and cr_s < cr_b and pd.notna(atv_use):
        extra_trans = float(foot) * (float(cr_b - cr_s) / 100.0)
        candidates.append(("Low Conversion vs type", max(0.0, extra_trans) * float(atv_use)))

    if not candidates:
        return np.nan, ""

    best = sorted(candidates, key=lambda x: x[1], reverse=True)[0]
    upside = float(best[1]) if best[1] > 0 else np.nan
    return upside, best[0]


# ------------------------------------------------------------
# Region-wide SVI table (rank across regions)
# ------------------------------------------------------------
def compute_region_svi_table(df_all: pd.DataFrame, region_list: list[str]) -> pd.DataFrame:
    """
    Computes region SVI for each region in region_list vs company totals.
    Returns DataFrame: region, svi, avg_ratio_vs_company, turnover, footfall, transactions, sqm_sum
    """
    if df_all is None or df_all.empty:
        return pd.DataFrame()

    comp_vals = compute_driver_values_from_period(
        footfall=float(pd.to_numeric(df_all["footfall"], errors="coerce").dropna().sum()),
        turnover=float(pd.to_numeric(df_all["turnover"], errors="coerce").dropna().sum()),
        transactions=float(pd.to_numeric(df_all["transactions"], errors="coerce").dropna().sum()),
        sqm_sum=float(pd.to_numeric(df_all["sqm_effective"], errors="coerce").dropna().sum()),
        capture_pct=np.nan,
    )

    rows = []
    for reg in region_list:
        dfr = df_all[df_all["region"] == reg].copy()
        if dfr.empty:
            continue

        reg_vals = compute_driver_values_from_period(
            footfall=float(pd.to_numeric(dfr["footfall"], errors="coerce").dropna().sum()),
            turnover=float(pd.to_numeric(dfr["turnover"], errors="coerce").dropna().sum()),
            transactions=float(pd.to_numeric(dfr["transactions"], errors="coerce").dropna().sum()),
            sqm_sum=float(pd.to_numeric(dfr["sqm_effective"], errors="coerce").dropna().sum()),
            capture_pct=np.nan,
        )

        svi, avg_ratio, _ = compute_svi_explainable(
            vals_a=reg_vals,
            vals_b=comp_vals,
            weights=BASE_SVI_WEIGHTS,
            floor=80,
            cap=120,
        )

        rows.append({
            "region": reg,
            "svi": float(svi) if pd.notna(svi) else np.nan,
            "avg_ratio_vs_company": float(avg_ratio) if pd.notna(avg_ratio) else np.nan,
            "turnover": float(pd.to_numeric(dfr["turnover"], errors="coerce").dropna().sum()),
            "footfall": float(pd.to_numeric(dfr["footfall"], errors="coerce").dropna().sum()),
            "transactions": float(pd.to_numeric(dfr["transactions"], errors="coerce").dropna().sum()),
            "sqm_sum": float(pd.to_numeric(dfr["sqm_effective"], errors="coerce").dropna().sum()),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("svi", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


# ------------------------------------------------------------
# Store-type comparison: current region vs other regions
# ------------------------------------------------------------
def compute_storetype_vs_other_regions(df_all: pd.DataFrame, region_choice: str) -> pd.DataFrame:
    """
    For each store_type:
      - compute KPIs for current region stores
      - compute KPIs for same store_type across OTHER regions
      - return indices (region / other_regions * 100)
    """
    if df_all is None or df_all.empty:
        return pd.DataFrame()

    df_all = df_all.copy()
    df_all["store_type"] = (
        df_all.get("store_type", "Unknown")
        .fillna("Unknown")
        .astype(str).str.strip()
        .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    )

    def kpis_from(df_: pd.DataFrame) -> dict:
        foot = pd.to_numeric(df_["footfall"], errors="coerce").dropna().sum()
        turn = pd.to_numeric(df_["turnover"], errors="coerce").dropna().sum()
        trans = pd.to_numeric(df_["transactions"], errors="coerce").dropna().sum()

        sqm_sum = np.nan
        if "id" in df_.columns:
            sqm = (
                df_[["id", "sqm_effective"]]
                .drop_duplicates(subset=["id"])
            )
            sqm_sum = pd.to_numeric(sqm["sqm_effective"], errors="coerce").dropna().sum()
        else:
            sqm_sum = pd.to_numeric(df_.get("sqm_effective"), errors="coerce").dropna().sum()

        return {
            "footfall": float(foot),
            "turnover": float(turn),
            "transactions": float(trans),
            "spv": safe_div(turn, foot),
            "cr": safe_div(trans, foot) * 100.0 if pd.notna(foot) else np.nan,
            "atv": safe_div(turn, trans),
            "spm2": safe_div(turn, sqm_sum),
            "sqm_sum": float(sqm_sum) if pd.notna(sqm_sum) else np.nan,
        }

    rows = []
    for stype in sorted(df_all["store_type"].dropna().unique().tolist()):
        dfr = df_all[(df_all["region"] == region_choice) & (df_all["store_type"] == stype)]
        dfo = df_all[(df_all["region"] != region_choice) & (df_all["store_type"] == stype)]

        if dfr.empty or dfo.empty:
            continue

        a = kpis_from(dfr)
        b = kpis_from(dfo)

        rows.append({
            "store_type": stype,
            "region_turnover": a["turnover"],
            "other_turnover": b["turnover"],
            "SPV idx vs other regions": idx_vs(a["spv"], b["spv"]),
            "CR idx vs other regions": idx_vs(a["cr"], b["cr"]),
            "Sales/m² idx vs other regions": idx_vs(a["spm2"], b["spm2"]),
            "Traffic idx vs other regions": idx_vs(a["footfall"], b["footfall"]),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Sales/m² idx vs other regions", ascending=True).reset_index(drop=True)


# ------------------------------------------------------------
# Outcome builder (agentic: region rank + store-type focus + upside by type)
# ------------------------------------------------------------
def build_region_outcomes(payload: dict) -> dict:
    meta = payload["meta"]
    scores = payload["scores"]
    store_summary = payload["store_summary"]
    company_type_bench = payload["company_type_bench"]
    region_rank = payload.get("region_rank", {})
    stype_vs_other = payload.get("stype_vs_other", pd.DataFrame())

    # --- store-level opportunities (same as before) ---
    opps_store = []
    for _, r in store_summary.iterrows():
        stype = str(r.get("store_type", "Unknown")).strip() or "Unknown"
        bench = company_type_bench.get(stype, {})
        up, driver = estimate_upside(r, bench)
        if pd.notna(up) and up > 0:
            actions = []
            if callable(make_actions):
                try:
                    actions = make_actions(r) or []
                except Exception:
                    actions = []
            opps_store.append({
                "store_name": r.get("store_display", "") or str(r.get("id", "")),
                "store_id": r.get("id", np.nan),
                "store_type": stype,
                "driver": driver,
                "impact_period_eur": float(up),
                "impact_annual_eur": float(up) * payload["annual_factor"],
                "actions": actions,
            })

    opps_store = sorted(opps_store, key=lambda x: x["impact_period_eur"], reverse=True)

    # --- aggregate upside by store_type (this is the “manager view”) ---
    opps_type = []
    if opps_store:
        df_o = pd.DataFrame(opps_store)
        g = df_o.groupby("store_type", as_index=False).agg(
            impact_period_eur=("impact_period_eur", "sum"),
            impact_annual_eur=("impact_annual_eur", "sum"),
            stores=("store_name", "count"),
        ).sort_values("impact_period_eur", ascending=False)

        for _, rr in g.head(3).iterrows():
            stype = rr["store_type"]
            top_store = df_o[df_o["store_type"] == stype].sort_values("impact_period_eur", ascending=False).head(2)
            examples = [f"{row['store_name']} ({fmt_eur(row['impact_period_eur'])})" for _, row in top_store.iterrows()]
            opps_type.append({
                "store_type": stype,
                "impact_period_eur": float(rr["impact_period_eur"]),
                "impact_annual_eur": float(rr["impact_annual_eur"]),
                "stores": int(rr["stores"]),
                "examples": examples,
            })

    # --- strengths: best stores (where potential is already used well) ---
    strengths = []
    if store_summary is not None and not store_summary.empty:
        ss = store_summary.copy()
        # define "strong" as high sales/m² and high SPV relative to type (simple deterministic)
        ss["score_strength"] = (
            pd.to_numeric(ss.get("Sales/m² idx vs type"), errors="coerce").fillna(100)
            + pd.to_numeric(ss.get("SPV idx vs type"), errors="coerce").fillna(100)
        )
        best = ss.sort_values("score_strength", ascending=False).head(3)
        for _, r in best.iterrows():
            strengths.append({
                "store_name": r.get("store_display", "") or str(r.get("id", "")),
                "store_type": str(r.get("store_type", "Unknown")),
                "spv_idx": r.get("SPV idx vs type", np.nan),
                "spm2_idx": r.get("Sales/m² idx vs type", np.nan),
                "cr_idx": r.get("CR idx vs type", np.nan),
            })

    # --- store-type focus vs other regions ---
    focus_types = []
    if isinstance(stype_vs_other, pd.DataFrame) and not stype_vs_other.empty:
        tmp = stype_vs_other.copy()
        # weakest store_types first (lowest Sales/m² idx vs other regions)
        weak = tmp.sort_values("Sales/m² idx vs other regions", ascending=True).head(3)
        for _, r in weak.iterrows():
            focus_types.append({
                "store_type": r["store_type"],
                "spv_idx_other": r.get("SPV idx vs other regions", np.nan),
                "cr_idx_other": r.get("CR idx vs other regions", np.nan),
                "spm2_idx_other": r.get("Sales/m² idx vs other regions", np.nan),
                "traffic_idx_other": r.get("Traffic idx vs other regions", np.nan),
            })

    # --- risks ---
    risks = []
    svi = scores.get("region_svi", np.nan)
    if pd.notna(svi) and float(svi) < 60:
        risks.append({
            "driver": "Regional performance pressure",
            "severity": "high",
            "why": [
                f"Region SVI: {int(round(float(svi)))} / 100",
                "Multiple store_types underperform vs other regions",
            ],
        })

    return {
        "meta": meta,
        "scores": scores,
        "region_rank": region_rank,
        "opportunities_by_type": opps_type,
        "opportunities_top_stores": opps_store[:3],
        "focus_store_types": focus_types,
        "strengths": strengths,
        "risks": risks,
        "notes": [
            "Benchmarked on same store_type (company-wide)",
            "Store-type comparisons exclude current region (vs other regions)",
            "sqm_effective: sqm_override overrides; else sq_meter from report payload"
        ]
    }


# ------------------------------------------------------------
# Render outcome feed
# ------------------------------------------------------------
def render_outcome_feed(outcomes: dict, typing: bool = True):
    st.markdown("## Agentic outcome feed")

    # If OutcomeExplainer exists, use it; else deterministic UI cards
    if OutcomeExplainer is None:
        st.warning("OutcomeExplainer not available — using deterministic cards (still fully usable).")
        render_deterministic_cards(outcomes)
        return

    explainer = OutcomeExplainer()
    cards = explainer.build_cards(outcomes, persona="region_manager", style="crisp", use_llm=False)

    for card in cards:
        box = st.empty()
        title = str(card.get("title", "") or "")
        body_full = str(card.get("body", "") or "")

        if typing:
            for chunk in explainer.stream_typing(body_full, chunk_size=24, delay=0.01):
                box.markdown(f"### {title}\n\n{chunk}")
        else:
            box.markdown(f"### {title}\n\n{body_full}")


def render_deterministic_cards(outcomes: dict):
    meta = outcomes.get("meta", {})
    scores = outcomes.get("scores", {})
    rr = outcomes.get("region_rank", {})

    st.markdown(f"### {meta.get('client','Client')} · {meta.get('region','Region')} · {meta.get('period_label','Period')}")

    cols = st.columns(3)
    svi = scores.get("region_svi", np.nan)
    cols[0].metric("Region SVI", f"{int(round(float(svi)))} / 100" if pd.notna(svi) else "-")

    if rr:
        cols[1].metric("Rank vs regions", f"{rr.get('rank','-')} / {rr.get('total','-')}")
        cols[2].metric("Gap to #1", fmt_pct(rr.get("gap_to_best", np.nan), 0))
    else:
        cols[1].metric("Rank vs regions", "-")
        cols[2].metric("Gap to #1", "-")

    st.markdown("#### 1) Where to focus (store types vs other regions)")
    focus = outcomes.get("focus_store_types", [])
    if focus:
        for f in focus:
            st.markdown(
                f"- **{f['store_type']}** · Sales/m² idx: **{fmt_pct(f['spm2_idx_other'],0)}** · "
                f"SPV idx: **{fmt_pct(f['spv_idx_other'],0)}** · CR idx: **{fmt_pct(f['cr_idx_other'],0)}** · "
                f"Traffic idx: **{fmt_pct(f['traffic_idx_other'],0)}**"
            )
    else:
        st.info("No store-type vs other regions comparison available (need at least 2 regions with same store_type).")

    st.markdown("#### 2) Biggest upside (aggregated by store type)")
    opp_t = outcomes.get("opportunities_by_type", [])
    if opp_t:
        for o in opp_t:
            st.markdown(
                f"- **{o['store_type']}** · **{fmt_eur(o['impact_period_eur'])}** (period) · "
                f"**{fmt_eur(o['impact_annual_eur'])}/yr** · stores impacted: {o['stores']} "
                f"· examples: {', '.join(o['examples'])}"
            )
    else:
        st.info("No upside found (or missing KPIs).")

    st.markdown("#### 3) Top stores to act on this week")
    opp_s = outcomes.get("opportunities_top_stores", [])
    if opp_s:
        for o in opp_s:
            st.markdown(
                f"- **{o['store_name']}** ({o['store_type']}) — {o['driver']} · "
                f"**{fmt_eur(o['impact_period_eur'])}** (period) · **{fmt_eur(o['impact_annual_eur'])}/yr**"
            )
    else:
        st.info("No store-level opportunities detected.")

    st.markdown("#### 4) What’s already working (copy/paste playbook)")
    strengths = outcomes.get("strengths", [])
    if strengths:
        for s in strengths:
            st.markdown(
                f"- **{s['store_name']}** ({s['store_type']}) · "
                f"SPV idx: **{fmt_pct(s['spv_idx'],0)}** · Sales/m² idx: **{fmt_pct(s['spm2_idx'],0)}** · "
                f"CR idx: **{fmt_pct(s['cr_idx'],0)}**"
            )
    else:
        st.info("No strengths summary available.")

    if outcomes.get("risks"):
        st.markdown("#### Risks / alerts")
        for r in outcomes["risks"]:
            st.error(f"**{r.get('driver','Risk')}** ({r.get('severity','')}) — " + " · ".join(r.get("why", [])))


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    # ---- API URL setup ----
    raw_api_url = st.secrets["API_URL"].rstrip("/")
    if raw_api_url.endswith("/get-report"):
        report_url = raw_api_url
        fastapi_base = raw_api_url.rsplit("/get-report", 1)[0]
    else:
        fastapi_base = raw_api_url
        report_url = raw_api_url + "/get-report"

    st.title("PFM Region Reasoner — Agentic Outcome Edition")

    # ---- Clients ----
    clients = load_clients("clients.json")
    clients_df = pd.DataFrame(clients)
    if clients_df.empty:
        st.error("No clients found in clients.json")
        st.stop()

    clients_df["label"] = clients_df.apply(
        lambda r: f"{r['brand']} – {r['name']} (company_id {r['company_id']})",
        axis=1
    )
    client_label = st.selectbox("Client", clients_df["label"].tolist())
    selected_client = clients_df[clients_df["label"] == client_label].iloc[0].to_dict()
    company_id = int(selected_client["company_id"])

    # ---- Periods ----
    periods = period_catalog(today=datetime.now().date())
    period_label = st.selectbox("Period", list(periods.keys()))
    p = periods[period_label]
    start_date, end_date = p.start, p.end

    # ---- Region mapping ----
    region_map = load_region_mapping()
    if region_map.empty:
        st.error("No valid data/regions.csv found (need shop_id;region;store_type).")
        st.stop()

    region_choice = st.selectbox("Region", sorted(region_map["region"].unique()))
    region_choice = str(region_choice).strip().lower()

    run = st.button("Run analysis", type="primary")
    if not run:
        st.stop()

    # ---- Locations (names only; sqm will come from report payload sq_meter) ----
    locations_df = get_locations_by_company(fastapi_base, company_id)
    if locations_df is None or locations_df.empty:
        st.error("No locations returned from /company/{company}/location")
        st.stop()

    if "id" not in locations_df.columns:
        st.error(f"Locations payload has no 'id' column. Columns: {locations_df.columns.tolist()}")
        st.stop()

    locations_df = locations_df.copy()
    locations_df["id"] = pd.to_numeric(locations_df["id"], errors="coerce").astype("Int64")

    # join region mapping onto locations (for region + store_type + sqm_override)
    merged = locations_df.merge(region_map, left_on="id", right_on="shop_id", how="inner")
    if merged.empty:
        st.warning("No stores matched regions.csv mapping.")
        st.stop()

    merged["region"] = merged["region"].astype(str).str.strip().str.lower()

    # store_display
    if "name" in merged.columns and merged["name"].notna().any():
        merged["store_display"] = merged["name"]
    else:
        merged["store_display"] = merged["id"].astype(str)

    # store_type hygiene
    if "store_type" not in merged.columns:
        merged["store_type"] = "Unknown"
    merged["store_type"] = (
        merged["store_type"]
        .fillna("Unknown")
        .astype(str).str.strip()
        .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    )

    # ---- Fetch report ----
    all_shop_ids = merged["id"].dropna().astype(int).unique().tolist()
    if not all_shop_ids:
        st.error("No shop IDs after mapping.")
        st.stop()

    cfg = VemcountApiConfig(report_url=report_url)

    # IMPORTANT: include sq_meter in data_outputs (you confirmed it's in response)
    metric_map = {
        "count_in": "footfall",
        "turnover": "turnover",
        "transactions": "transactions",
        "sales_per_sqm": "sales_per_sqm",
        "sq_meter": "sq_meter",
    }

    params_preview = build_report_params(
        shop_ids=all_shop_ids,
        data_outputs=list(metric_map.keys()),
        period="date",
        step="day",
        source="shops",
        date_from=start_date,
        date_to=end_date,
    )

    try:
        resp = fetch_report(
            cfg=cfg,
            shop_ids=all_shop_ids,
            data_outputs=list(metric_map.keys()),
            period="date",
            step="day",
            source="shops",
            date_from=start_date,
            date_to=end_date,
            timeout=120,
        )
    except Exception as e:
        st.error(f"Report fetch failed: {e}")
        st.write("Params preview:", params_preview)
        st.stop()

    df_norm = normalize_vemcount_response(resp, kpi_keys=metric_map.keys()).rename(columns=metric_map)
    if df_norm is None or df_norm.empty:
        st.warning("No data returned for this selection.")
        st.stop()

    # store key
    store_key = None
    for cand in ["shop_id", "id", "location_id"]:
        if cand in df_norm.columns:
            store_key = cand
            break
    if store_key is None:
        st.error("No store id column in report response.")
        st.stop()

    df_norm[store_key] = pd.to_numeric(df_norm[store_key], errors="coerce").astype("Int64")

    # Join dimensions (region + store_type + store_display + sqm_override)
    df = df_norm.merge(
        merged[["id", "store_display", "region", "store_type", "sqm_override"]].drop_duplicates(),
        left_on=store_key,
        right_on="id",
        how="left"
    )

    # Ensure numeric
    for c in ["footfall", "turnover", "transactions", "sales_per_sqm", "sq_meter", "sqm_override"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # sqm_effective: sqm_override overrides; else sq_meter from report payload
    if "sq_meter" not in df.columns:
        df["sq_meter"] = np.nan
    sqm_eff = pd.to_numeric(df["sqm_override"], errors="coerce").combine_first(pd.to_numeric(df["sq_meter"], errors="coerce"))
    sqm_eff = pd.Series(sqm_eff, index=df.index, dtype="float64")
    sqm_eff.loc[sqm_eff <= 0] = np.nan
    df["sqm_effective"] = sqm_eff

    # sales_per_sqm: keep API value, only backfill where missing/NaN
    if "sales_per_sqm" not in df.columns:
        df["sales_per_sqm"] = np.nan
    df["sales_per_sqm"] = pd.to_numeric(df["sales_per_sqm"], errors="coerce")

    computed_spm2 = pd.Series(
        np.where((df["sqm_effective"] > 0) & df["turnover"].notna(), df["turnover"] / df["sqm_effective"], np.nan),
        index=df.index
    )
    df["sales_per_sqm"] = df["sales_per_sqm"].combine_first(computed_spm2)

    # Region subset
    df_region = df[df["region"] == region_choice].copy()
    if df_region.empty:
        st.warning("No rows for selected region after join.")
        st.stop()

    # Store summary (period totals)
    store_summary = (
        df_region
        .groupby(["id", "store_display", "store_type", "sqm_effective"], as_index=False)
        .agg(
            turnover=("turnover", "sum"),
            footfall=("footfall", "sum"),
            transactions=("transactions", "sum"),
        )
    )

    store_summary["conversion_rate"] = np.where(
        store_summary["footfall"] > 0,
        store_summary["transactions"] / store_summary["footfall"] * 100.0,
        np.nan
    )
    store_summary["sales_per_visitor"] = np.where(
        store_summary["footfall"] > 0,
        store_summary["turnover"] / store_summary["footfall"],
        np.nan
    )
    store_summary["sales_per_transaction"] = np.where(
        store_summary["transactions"] > 0,
        store_summary["turnover"] / store_summary["transactions"],
        np.nan
    )
    store_summary["sales_per_sqm"] = np.where(
        store_summary["sqm_effective"] > 0,
        store_summary["turnover"] / store_summary["sqm_effective"],
        np.nan
    )

    # Company type benchmarks (company-wide)
    df_all = df.copy()
    company_type_bench = compute_company_type_bench(df_all)

    # Indices vs type bench (company-wide)
    store_summary["Traffic idx vs type"] = store_summary.apply(
        lambda r: idx_vs(r["footfall"], company_type_bench.get(r["store_type"], {}).get("footfall", np.nan)),
        axis=1
    )
    store_summary["CR idx vs type"] = store_summary.apply(
        lambda r: idx_vs(r["conversion_rate"], company_type_bench.get(r["store_type"], {}).get("conversion_rate", np.nan)),
        axis=1
    )
    store_summary["SPV idx vs type"] = store_summary.apply(
        lambda r: idx_vs(r["sales_per_visitor"], company_type_bench.get(r["store_type"], {}).get("sales_per_visitor", np.nan)),
        axis=1
    )
    store_summary["ATV idx vs type"] = store_summary.apply(
        lambda r: idx_vs(r["sales_per_transaction"], company_type_bench.get(r["store_type"], {}).get("sales_per_transaction", np.nan)),
        axis=1
    )
    store_summary["Sales/m² idx vs type"] = store_summary.apply(
        lambda r: idx_vs(r["sales_per_sqm"], company_type_bench.get(r["store_type"], {}).get("sales_per_sqm", np.nan)),
        axis=1
    )

    # Region SVI vs company
    reg_vals = compute_driver_values_from_period(
        footfall=float(pd.to_numeric(df_region["footfall"], errors="coerce").dropna().sum()),
        turnover=float(pd.to_numeric(df_region["turnover"], errors="coerce").dropna().sum()),
        transactions=float(pd.to_numeric(df_region["transactions"], errors="coerce").dropna().sum()),
        sqm_sum=float(pd.to_numeric(df_region["sqm_effective"], errors="coerce").dropna().sum()),
        capture_pct=np.nan,
    )
    comp_vals = compute_driver_values_from_period(
        footfall=float(pd.to_numeric(df_all["footfall"], errors="coerce").dropna().sum()),
        turnover=float(pd.to_numeric(df_all["turnover"], errors="coerce").dropna().sum()),
        transactions=float(pd.to_numeric(df_all["transactions"], errors="coerce").dropna().sum()),
        sqm_sum=float(pd.to_numeric(df_all["sqm_effective"], errors="coerce").dropna().sum()),
        capture_pct=np.nan,
    )

    region_svi, avg_ratio, _ = compute_svi_explainable(
        vals_a=reg_vals,
        vals_b=comp_vals,
        weights=BASE_SVI_WEIGHTS,
        floor=80,
        cap=120,
    )

    # Annual factor
    days = max(1, (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1)
    annual_factor = 365.0 / float(days)

    # Region SVI rank table
    region_list = sorted(df_all["region"].dropna().unique().tolist())
    svi_table = compute_region_svi_table(df_all, region_list)

    region_rank = {}
    if svi_table is not None and not svi_table.empty:
        row = svi_table[svi_table["region"] == region_choice]
        if not row.empty:
            rank = int(row.iloc[0]["rank"])
            total = int(svi_table["rank"].max())
            best = float(svi_table.iloc[0]["svi"]) if pd.notna(svi_table.iloc[0]["svi"]) else np.nan
            cur = float(row.iloc[0]["svi"]) if pd.notna(row.iloc[0]["svi"]) else np.nan
            region_rank = {
                "rank": rank,
                "total": total,
                "best_svi": best,
                "gap_to_best": (cur / best * 100.0) if (pd.notna(cur) and pd.notna(best) and best != 0) else np.nan,
            }

    # Store-type comparison vs other regions
    stype_vs_other = compute_storetype_vs_other_regions(df_all, region_choice)

    payload = {
        "store_summary": store_summary,
        "company_type_bench": company_type_bench,
        "annual_factor": annual_factor,
        "region_rank": region_rank,
        "stype_vs_other": stype_vs_other,
        "meta": {
            "client": selected_client.get("brand", ""),
            "region": region_choice,
            "period_label": f"{period_label} ({start_date} → {end_date})"
        },
        "scores": {
            "region_svi": float(region_svi) if pd.notna(region_svi) else np.nan,
            "avg_ratio_vs_company": float(avg_ratio) if pd.notna(avg_ratio) else np.nan
        }
    }

    outcomes = build_region_outcomes(payload)

    # --- Layout: show comparisons BEFORE the narrative ---
    with st.expander("📊 Region comparison (SVI vs other regions)", expanded=True):
        if svi_table is None or svi_table.empty:
            st.info("No SVI table available.")
        else:
            show = svi_table.copy()
            show["SVI"] = show["svi"].round(0)
            show["Avg ratio vs company"] = show["avg_ratio_vs_company"].round(0)
            show["Turnover"] = show["turnover"].map(fmt_eur)
            show["Footfall"] = show["footfall"].round(0)
            show["Transactions"] = show["transactions"].round(0)
            show = show[["rank", "region", "SVI", "Avg ratio vs company", "Turnover", "Footfall", "Transactions"]]
            st.dataframe(show, use_container_width=True)

    with st.expander("🧩 Store types vs other regions (same type)", expanded=True):
        if stype_vs_other is None or stype_vs_other.empty:
            st.info("No store-type comparison available (need same store_type across >=2 regions).")
        else:
            st.dataframe(
                stype_vs_other.assign(
                    **{
                        "SPV idx vs other regions": stype_vs_other["SPV idx vs other regions"].round(0),
                        "CR idx vs other regions": stype_vs_other["CR idx vs other regions"].round(0),
                        "Sales/m² idx vs other regions": stype_vs_other["Sales/m² idx vs other regions"].round(0),
                        "Traffic idx vs other regions": stype_vs_other["Traffic idx vs other regions"].round(0),
                        "region_turnover": stype_vs_other["region_turnover"].map(fmt_eur),
                        "other_turnover": stype_vs_other["other_turnover"].map(fmt_eur),
                    }
                ),
                use_container_width=True
            )

    # --- The “agentic” feed ---
    render_outcome_feed(outcomes, typing=True)

    # Debug
    with st.expander("🔧 Debug"):
        st.write("REPORT_URL:", report_url)
        st.write("FASTAPI_BASE:", fastapi_base)
        st.write("Company:", company_id)
        st.write("Region:", region_choice)
        st.write("Period:", start_date, "→", end_date)
        st.write("Params preview:", params_preview)
        st.write("merged cols:", merged.columns.tolist())
        st.write("df_norm cols:", df_norm.columns.tolist())
        st.write("sqm_effective filled:", int(pd.to_numeric(df["sqm_effective"], errors="coerce").notna().sum()), "/", len(df))
        st.write("Example sqm (from report):", df[["id", "store_display", "sqm_override", "sq_meter", "sqm_effective"]].head(10))


main()