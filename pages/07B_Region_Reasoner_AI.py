# pages/07B_Region_Reasoner_AI.py
# ------------------------------------------------------------
# PFM Region Reasoner — Agentic Outcome Edition (IMPORT-SAFE)
# Fixes (this version):
# - Pull sq_meter (and optional name) from /get-report response shop metadata
# - Use that as base sqm when locations endpoint lacks sq_meter
# - Debug expander no longer crashes on duplicate columns
# - sqm column detection never picks sqm_override/sqm_effective
# - OutcomeExplainer always available (fallback) + debug shows import error
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
_REPO_ROOT = _THIS_FILE.parents[1]  # retail-suite-tool/
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


# ------------------------------------------------------------
# OutcomeExplainer import + fallback
# ------------------------------------------------------------
_OUTCOME_IMPORT_ERROR = None

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


class _FallbackOutcomeExplainer:
    def build_cards(self, outcomes: dict, persona: str = "region_manager", style: str = "crisp", use_llm: bool = False):
        meta = outcomes.get("meta", {})
        scores = outcomes.get("scores", {})
        cards = []

        title = f"{meta.get('client','Client')} · {meta.get('region','Region')} · {meta.get('period_label','Period')}"
        body = [f"**{title}**"]

        if pd.notna(scores.get("region_svi", np.nan)):
            body.append(f"- **Region SVI:** {int(round(float(scores['region_svi'])))} / 100")

        opps = outcomes.get("opportunities", []) or []
        if opps:
            body.append("\n**Top opportunities:**")
            for o in opps[:3]:
                body.append(
                    f"- **{o.get('store_name','')}** — {o.get('driver','')} · "
                    f"{fmt_eur(o.get('impact_period_eur'))} (period) · {fmt_eur(o.get('impact_annual_eur'))}/yr"
                )
        else:
            body.append("\nNo upside opportunities detected.")

        cards.append({"title": "Region summary", "body": "\n".join(body)})

        return cards

    def stream_typing(self, text: str, chunk_size: int = 24, delay: float = 0.0):
        acc = ""
        for i in range(0, len(text), chunk_size):
            acc = text[: i + chunk_size]
            yield acc


try:
    from outcome_explainer import OutcomeExplainer  # noqa
except Exception as e:
    _OUTCOME_IMPORT_ERROR = repr(e)
    OutcomeExplainer = _FallbackOutcomeExplainer


# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(page_title="PFM Region Reasoner — Agentic", layout="wide")


# ------------------------------------------------------------
# Region mapping
# ------------------------------------------------------------
@st.cache_data(ttl=600)
def load_region_mapping(path="data/regions.csv") -> pd.DataFrame:
    """
    Expected:
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
# Vemcount locations via FastAPI (optional)
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
# Extract store meta (sq_meter) from report response
# ------------------------------------------------------------
def extract_store_meta_from_report(resp: dict) -> pd.DataFrame:
    """
    Expected structure (based on your JSON example):
      resp["data"][period_key][shop_id]["data"]["sq_meter"]
      resp["data"][period_key][shop_id]["data"]["name"] (optional)
    Returns columns: id, sq_meter, name
    """
    rows = []
    if not isinstance(resp, dict):
        return pd.DataFrame()

    data = resp.get("data")
    if not isinstance(data, dict) or not data:
        return pd.DataFrame()

    # period_key could be "last_year" OR "date_YYYY-MM-DD" etc. We'll iterate all top-level keys.
    for period_key, period_block in data.items():
        if not isinstance(period_block, dict):
            continue
        for shop_id, shop_payload in period_block.items():
            if not isinstance(shop_payload, dict):
                continue
            meta = shop_payload.get("data") or {}
            if not isinstance(meta, dict):
                continue

            sq = meta.get("sq_meter", np.nan)
            nm = meta.get("name", None)
            try:
                sid = int(shop_id)
            except Exception:
                sid = pd.to_numeric(shop_id, errors="coerce")
                if pd.isna(sid):
                    continue
                sid = int(sid)

            rows.append({
                "id": sid,
                "sq_meter": sq,
                "name": nm
            })

    if not rows:
        return pd.DataFrame()

    dfm = pd.DataFrame(rows).drop_duplicates(subset=["id"])
    dfm["id"] = pd.to_numeric(dfm["id"], errors="coerce").astype("Int64")
    dfm["sq_meter"] = pd.to_numeric(dfm["sq_meter"], errors="coerce")
    return dfm


# ------------------------------------------------------------
# KPI / Benchmark helpers
# ------------------------------------------------------------
def idx_vs(a, b):
    return (a / b * 100.0) if (pd.notna(a) and pd.notna(b) and float(b) != 0.0) else np.nan


def compute_company_type_bench(df_all: pd.DataFrame) -> dict:
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


def estimate_upside(store_row: pd.Series, bench_vals: dict) -> tuple[float, str]:
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


def build_region_outcomes(payload: dict) -> dict:
    meta = payload["meta"]
    scores = payload["scores"]
    store_summary = payload["store_summary"]
    company_type_bench = payload["company_type_bench"]

    opportunities = []
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
            opportunities.append({
                "store_name": r.get("store_display", ""),
                "store_type": stype,
                "driver": driver,
                "impact_period_eur": float(up),
                "impact_annual_eur": float(up) * payload["annual_factor"],
                "actions": actions,
            })

    opportunities = sorted(opportunities, key=lambda x: x["impact_period_eur"], reverse=True)[:3]

    return {
        "meta": meta,
        "scores": scores,
        "opportunities": opportunities,
        "risks": [],
        "notes": [
            "Benchmarked on same store_type (company-wide)",
            "sqm source: regions.csv sqm_override → report meta sq_meter"
        ]
    }


def render_outcome_feed(outcomes: dict, typing: bool = True):
    st.markdown("## Agentic outcome feed")

    explainer = OutcomeExplainer()
    cards = explainer.build_cards(outcomes, persona="region_manager", style="crisp", use_llm=False)

    for card in cards:
        box = st.empty()
        title = str(card.get("title", "") or "")
        body_full = str(card.get("body", "") or "")

        if typing:
            for chunk in explainer.stream_typing(body_full, chunk_size=24, delay=0.0):
                box.markdown(f"### {title}\n\n{chunk}")
        else:
            box.markdown(f"### {title}\n\n{body_full}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    raw_api_url = st.secrets["API_URL"].rstrip("/")
    if raw_api_url.endswith("/get-report"):
        report_url = raw_api_url
        fastapi_base = raw_api_url.rsplit("/get-report", 1)[0]
    else:
        fastapi_base = raw_api_url
        report_url = raw_api_url + "/get-report"

    st.title("PFM Region Reasoner — Agentic Outcome Edition")

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

    periods = period_catalog(today=datetime.now().date())
    period_label = st.selectbox("Period", list(periods.keys()))
    p = periods[period_label]
    start_date, end_date = p.start, p.end

    region_map = load_region_mapping()
    if region_map.empty:
        st.error("No valid data/regions.csv found (need shop_id;region;store_type).")
        st.stop()

    region_choice = st.selectbox("Region", sorted(region_map["region"].unique()))
    region_choice = str(region_choice).strip().lower()

    run = st.button("Run analysis", type="primary")
    if not run:
        st.stop()

    # --- Locations only gives id/name in your setup; still useful for list of shops ---
    locations_df = get_locations_by_company(fastapi_base, company_id)
    if locations_df is None or locations_df.empty:
        st.error("No locations returned from /company/{company}/location")
        st.stop()

    # normalize locations cols
    locations_df = locations_df.copy()
    locations_df.columns = [str(c).strip().lower() for c in locations_df.columns]

    if "id" not in locations_df.columns:
        st.error(f"Locations payload has no 'id' column. Columns: {locations_df.columns.tolist()}")
        st.stop()

    locations_df["id"] = pd.to_numeric(locations_df["id"], errors="coerce").astype("Int64")

    merged = locations_df.merge(region_map, left_on="id", right_on="shop_id", how="inner")
    if merged.empty:
        st.warning("No stores matched regions.csv mapping.")
        st.stop()

    merged.columns = [str(c).strip().lower() for c in merged.columns]
    merged["region"] = merged["region"].astype(str).str.strip().str.lower()

    if "name" in merged.columns:
        merged["store_display"] = merged["name"]
    else:
        merged["store_display"] = merged["id"].astype(str)

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

    metric_map = {
        "count_in": "footfall",
        "turnover": "turnover",
        "transactions": "transactions",
        "sales_per_sqm": "sales_per_sqm",
        "sq_meter": "sq_meter",  # IMPORTANT: request sq_meter so it's present in meta anyway
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

    # --- NEW: extract sq_meter from report meta ---
    df_meta = extract_store_meta_from_report(resp)
    if df_meta is None:
        df_meta = pd.DataFrame()

    if not df_meta.empty:
        merged = merged.merge(df_meta[["id", "sq_meter"]].drop_duplicates(), on="id", how="left")
    else:
        merged["sq_meter"] = np.nan

    # sqm_effective:
    # priority: sqm_override (regions.csv) -> sq_meter (report meta)
    sqm_override = pd.to_numeric(merged.get("sqm_override", np.nan), errors="coerce")
    if not isinstance(sqm_override, pd.Series):
        sqm_override = pd.Series(sqm_override, index=merged.index, dtype="float64")

    base_sqm = pd.to_numeric(merged.get("sq_meter", np.nan), errors="coerce")
    if not isinstance(base_sqm, pd.Series):
        base_sqm = pd.Series(base_sqm, index=merged.index, dtype="float64")

    merged["sqm_effective"] = sqm_override.combine_first(base_sqm)
    merged["sqm_effective"] = pd.to_numeric(merged["sqm_effective"], errors="coerce")
    merged.loc[merged["sqm_effective"] <= 0, "sqm_effective"] = np.nan

    # normalize KPI rows
    df_norm = normalize_vemcount_response(resp, kpi_keys=["count_in", "turnover", "transactions", "sales_per_sqm"]).rename(
        columns={
            "count_in": "footfall",
            "turnover": "turnover",
            "transactions": "transactions",
            "sales_per_sqm": "sales_per_sqm",
        }
    )
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

    df = df_norm.merge(
        merged[["id", "store_display", "region", "store_type", "sqm_effective"]].drop_duplicates(),
        left_on=store_key,
        right_on="id",
        how="left"
    )

    for c in ["footfall", "turnover", "transactions", "sales_per_sqm", "sqm_effective"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # backfill sales_per_sqm only if missing/NaN
    if "sales_per_sqm" not in df.columns:
        df["sales_per_sqm"] = np.nan

    sqm_eff = pd.to_numeric(df.get("sqm_effective", np.nan), errors="coerce")
    turn = pd.to_numeric(df.get("turnover", np.nan), errors="coerce")
    computed_spm2 = pd.Series(np.where((sqm_eff > 0) & turn.notna(), turn / sqm_eff, np.nan), index=df.index)
    df["sales_per_sqm"] = pd.to_numeric(df["sales_per_sqm"], errors="coerce").combine_first(computed_spm2)

    df_region = df[df["region"] == region_choice].copy()
    if df_region.empty:
        st.warning("No rows for selected region after join.")
        st.stop()

    store_summary = (
        df_region
        .groupby(["id", "store_display", "store_type", "sqm_effective"], as_index=False)
        .agg(
            turnover=("turnover", "sum"),
            footfall=("footfall", "sum"),
            transactions=("transactions", "sum"),
        )
    )

    store_summary["conversion_rate"] = np.where(store_summary["footfall"] > 0, store_summary["transactions"] / store_summary["footfall"] * 100.0, np.nan)
    store_summary["sales_per_visitor"] = np.where(store_summary["footfall"] > 0, store_summary["turnover"] / store_summary["footfall"], np.nan)
    store_summary["sales_per_sqm"] = np.where(store_summary["sqm_effective"] > 0, store_summary["turnover"] / store_summary["sqm_effective"], np.nan)

    df_all = df.copy()
    company_type_bench = compute_company_type_bench(df_all)

    store_summary["Traffic idx vs type"] = store_summary.apply(
        lambda r: idx_vs(r["footfall"], company_type_bench.get(r["store_type"], {}).get("footfall", np.nan)), axis=1
    )
    store_summary["CR idx vs type"] = store_summary.apply(
        lambda r: idx_vs(r["conversion_rate"], company_type_bench.get(r["store_type"], {}).get("conversion_rate", np.nan)), axis=1
    )
    store_summary["Sales/m² idx vs type"] = store_summary.apply(
        lambda r: idx_vs(r["sales_per_sqm"], company_type_bench.get(r["store_type"], {}).get("sales_per_sqm", np.nan)), axis=1
    )

    reg_vals = compute_driver_values_from_period(
        footfall=float(df_region["footfall"].dropna().sum()),
        turnover=float(df_region["turnover"].dropna().sum()),
        transactions=float(df_region["transactions"].dropna().sum()),
        sqm_sum=float(pd.to_numeric(df_region["sqm_effective"], errors="coerce").dropna().sum()),
        capture_pct=np.nan,
    )
    comp_vals = compute_driver_values_from_period(
        footfall=float(df_all["footfall"].dropna().sum()),
        turnover=float(df_all["turnover"].dropna().sum()),
        transactions=float(df_all["transactions"].dropna().sum()),
        sqm_sum=float(pd.to_numeric(df_all["sqm_effective"], errors="coerce").dropna().sum()),
        capture_pct=np.nan,
    )

    region_svi, avg_ratio, _ = compute_svi_explainable(vals_a=reg_vals, vals_b=comp_vals, weights=BASE_SVI_WEIGHTS, floor=80, cap=120)

    days = max(1, (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1)
    annual_factor = 365.0 / float(days)

    payload = {
        "store_summary": store_summary,
        "company_type_bench": company_type_bench,
        "annual_factor": annual_factor,
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
    render_outcome_feed(outcomes, typing=True)

    with st.expander("🔧 Debug"):
        st.write("REPORT_URL:", report_url)
        st.write("FASTAPI_BASE:", fastapi_base)
        st.write("Company:", company_id)
        st.write("Region:", region_choice)
        st.write("Period:", start_date, "→", end_date)
        st.write("Params preview:", params_preview)
        st.write("OutcomeExplainer import error:", _OUTCOME_IMPORT_ERROR)

        st.write("locations_df columns:", locations_df.columns.tolist())
        st.write("merged columns:", merged.columns.tolist())
        st.write("df_meta columns:", df_meta.columns.tolist() if isinstance(df_meta, pd.DataFrame) else None)

        st.write(
            "sqm_effective filled:",
            int(pd.to_numeric(merged["sqm_effective"], errors="coerce").notna().sum()),
            "/",
            len(merged)
        )
        cols_show = ["id", "store_display", "sqm_override", "sq_meter", "sqm_effective"]
        cols_show = [c for c in cols_show if c in merged.columns]
        st.write("Example sqm:", merged[cols_show].head(10))

main()