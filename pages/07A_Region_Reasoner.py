# pages/07A_Region_Reasoner.py
# ------------------------------------------------------------
# Region Reasoner — Agentic Workload (NO CHARTS)
# - Uses SAME selectors as 06C_Regio_Copilot_SVI.py
# - Uses SAME helpers_vemcount_api interface (cfg=..., shop_ids=..., data_outputs=...)
# - Uses regions.csv for region mapping
# - Minimal output: region KPI + per-store "workload" (text + table)
# - Staffing: only if data exists
# ------------------------------------------------------------

import os
import json
import numpy as np
import pandas as pd
import requests
import streamlit as st

from datetime import datetime

from helpers_clients import load_clients
from helpers_periods import period_catalog
from helpers_normalize import normalize_vemcount_response
from helpers_vemcount_api import VemcountApiConfig, fetch_report, build_report_params

# Optional OpenAI (if you want the wording)
# - Only used if OPENAI_API_KEY exists
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ----------------------
# Page config
# ----------------------
st.set_page_config(page_title="Region Reasoner (Agentic)", layout="wide")


# ----------------------
# Small format helpers (EU-ish)
# ----------------------
def fmt_eur(x):
    try:
        if pd.isna(x):
            return "-"
        return f"€ {float(x):,.0f}".replace(",", ".")
    except Exception:
        return "-"

def fmt_int(x):
    try:
        if pd.isna(x):
            return "-"
        return f"{float(x):,.0f}".replace(",", ".")
    except Exception:
        return "-"

def fmt_pct(x):
    try:
        if pd.isna(x):
            return "-"
        return f"{float(x):.1f}%".replace(".", ",")
    except Exception:
        return "-"

def safe_div(a, b):
    try:
        a = float(a)
        b = float(b)
        if b == 0:
            return np.nan
        return a / b
    except Exception:
        return np.nan


# ----------------------
# API URL / endpoints (same logic as your working script)
# ----------------------
raw_api_url = st.secrets["API_URL"].rstrip("/")

if raw_api_url.endswith("/get-report"):
    REPORT_URL = raw_api_url
    FASTAPI_BASE_URL = raw_api_url.rsplit("/get-report", 1)[0]
else:
    FASTAPI_BASE_URL = raw_api_url
    REPORT_URL = raw_api_url + "/get-report"


# ----------------------
# Region mapping (same sep=";")
# ----------------------
@st.cache_data(ttl=600)
def load_region_mapping(path: str = "data/regions.csv") -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    if "shop_id" not in df.columns or "region" not in df.columns:
        return pd.DataFrame()
    df["shop_id"] = pd.to_numeric(df["shop_id"], errors="coerce").astype("Int64")
    df["region"] = df["region"].astype(str)
    # optional cols
    if "store_type" in df.columns:
        df["store_type"] = df["store_type"].astype(str)
    if "store_label" in df.columns:
        df["store_label"] = df["store_label"].astype(str)
    if "sqm_override" in df.columns:
        df["sqm_override"] = pd.to_numeric(df["sqm_override"], errors="coerce")
    else:
        df["sqm_override"] = np.nan
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
# Agentic logic
# ----------------------
def compute_store_kpis(df_store: pd.DataFrame) -> dict:
    foot = float(pd.to_numeric(df_store.get("footfall", np.nan), errors="coerce").dropna().sum())
    turn = float(pd.to_numeric(df_store.get("turnover", np.nan), errors="coerce").dropna().sum())
    trans = float(pd.to_numeric(df_store.get("transactions", np.nan), errors="coerce").dropna().sum())
    conv = (trans / foot * 100.0) if foot > 0 else np.nan
    spv = (turn / foot) if foot > 0 else np.nan
    atv = (turn / trans) if trans > 0 else np.nan
    return {
        "footfall": foot,
        "turnover": turn,
        "transactions": trans,
        "conversion_rate": conv,
        "sales_per_visitor": spv,
        "sales_per_transaction": atv,
    }


def pick_best_actions(store_k: dict, bench_k: dict, staffing_k: dict | None = None) -> list[dict]:
    """
    Data-driven actions only.
    No industry cliches (no "upsell perfume", etc.)

    We select 1–2 actions based on the largest relative gap vs benchmark,
    and we attach a crisp measurement plan.
    """
    actions = []

    # Build comparable metrics
    metrics = [
        ("conversion_rate", "Conversion", "%", "CR"),
        ("sales_per_visitor", "Sales per Visitor", "€", "SPV"),
        ("sales_per_transaction", "Avg Transaction Value", "€", "ATV"),
    ]

    gaps = []
    for key, label, unit, short in metrics:
        a = store_k.get(key, np.nan)
        b = bench_k.get(key, np.nan)
        if pd.notna(a) and pd.notna(b) and float(b) != 0.0:
            idx = (float(a) / float(b)) * 100.0
            gaps.append((key, label, unit, short, idx, a, b))

    # Sort by worst index first
    gaps = sorted(gaps, key=lambda x: x[4])

    # Action logic
    for key, label, unit, short, idx, a, b in gaps:
        if idx >= 95:
            continue  # close enough
        if key == "conversion_rate":
            actions.append({
                "action": "Fix conversion leakage",
                "why": f"{label} is {idx:.0f}% of benchmark ({fmt_pct(a)} vs {fmt_pct(b)}).",
                "how_to_test": "Run 2-week A/B: service script / queue management / hero product placement. Measure CR uplift vs baseline.",
                "kpi": "conversion_rate",
            })
        elif key == "sales_per_visitor":
            actions.append({
                "action": "Lift spend per visitor (without discounting by default)",
                "why": f"{label} is {idx:.0f}% of benchmark ({fmt_eur(a)} vs {fmt_eur(b)}).",
                "how_to_test": "Audit basket builders: bundles, attach-rate prompts, layout friction. Measure SPV uplift week-over-week.",
                "kpi": "sales_per_visitor",
            })
        elif key == "sales_per_transaction":
            actions.append({
                "action": "Increase ATV through guided selling (not blanket promo)",
                "why": f"{label} is {idx:.0f}% of benchmark ({fmt_eur(a)} vs {fmt_eur(b)}).",
                "how_to_test": "Coach top-3 add-on prompts + assortment availability check. Measure ATV uplift and margin impact.",
                "kpi": "sales_per_transaction",
            })
        if len(actions) >= 2:
            break

    # Staffing (only if present)
    if staffing_k and staffing_k.get("available", False):
        staff_idx = staffing_k.get("coverage_idx", np.nan)
        if pd.notna(staff_idx) and staff_idx < 90:
            actions.append({
                "action": "Rebalance staffing to demand peaks",
                "why": f"Staffing coverage index {staff_idx:.0f}% vs target. (Data available)",
                "how_to_test": "Shift 1–2 FTE blocks to top-footfall hours; track CR/SPV change during peak windows.",
                "kpi": "staffing_coverage",
            })

    if not actions:
        actions = [{
            "action": "Maintain & replicate what's working",
            "why": "This store is within ~5% of benchmark on key commercial KPIs.",
            "how_to_test": "Document best practices and replicate to bottom-25% stores in region.",
            "kpi": "playbook",
        }]

    return actions


def llm_summarize(openai_client, context: dict) -> str:
    """
    If OpenAI is available, create a short executive summary.
    Safe fallback if not.
    """
    if openai_client is None:
        return ""

    prompt = f"""
You are a retail performance analyst. Write a crisp 6-10 line summary for a region manager.
No generic advice, no industry clichés. Only refer to metrics in the context.

Context JSON:
{json.dumps(context, ensure_ascii=False, default=str)}
"""
    try:
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You write concise, data-driven retail actions."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return ""


# ----------------------
# MAIN
# ----------------------
def main():
    # Session defaults
    st.session_state.setdefault("rr_last_key", None)
    st.session_state.setdefault("rr_payload", None)
    st.session_state.setdefault("rr_ran", False)

    # Title row like Copilot (simple)
    st.markdown("## 🧠 Region Reasoner (Agentic Workload)")
    st.caption("Same selectors as Copilot v2 · No charts · Focus on what to do next per store")

    # Clients + periods (same as your working script)
    clients = load_clients("clients.json")
    clients_df = pd.DataFrame(clients)

    if clients_df.empty:
        st.error("No clients found in clients.json")
        return

    required_cols = {"brand", "name", "company_id"}
    if not required_cols.issubset(set(clients_df.columns)):
        st.error(f"clients.json missing columns. Required: {sorted(required_cols)}")
        return

    clients_df["label"] = clients_df.apply(
        lambda r: f"{r['brand']} – {r['name']} (company_id {r['company_id']})",
        axis=1,
    )

    periods = period_catalog(today=datetime.now().date())
    if not isinstance(periods, dict) or len(periods) == 0:
        st.error("period_catalog() returned no periods.")
        return
    period_labels = list(periods.keys())

    # Row selectors
    c1, c2, c3, c4 = st.columns([2.2, 1.4, 1.4, 0.8], vertical_alignment="center")
    with c1:
        client_label = st.selectbox("Client", clients_df["label"].tolist(), key="rr_client")
    with c2:
        period_choice = st.selectbox("Period", period_labels, key="rr_period")
    with c3:
        # region selector depends on mapping; placeholder for now
        st.write("")
    with c4:
        run_btn = st.button("Run", type="primary", key="rr_run")

    selected_client = clients_df[clients_df["label"] == client_label].iloc[0].to_dict()
    company_id = int(selected_client["company_id"])

    start_period = periods[period_choice].start
    end_period = periods[period_choice].end

    # Load locations and region mapping
    try:
        locations_df = get_locations_by_company(company_id)
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching stores from FastAPI: {e}")
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

    # Store display name (same pattern)
    if "store_label" in merged.columns and merged["store_label"].notna().any():
        merged["store_display"] = merged["store_label"]
    else:
        merged["store_display"] = merged["name"] if "name" in merged.columns else merged["id"].astype(str)

    available_regions = sorted(merged["region"].dropna().unique().tolist())

    # Put region selector in column 3 now that we have options
    with c3:
        region_choice = st.selectbox("Region", available_regions, key="rr_region")

    # Key
    run_key = (company_id, region_choice, str(start_period), str(end_period), period_choice)
    selection_changed = st.session_state.rr_last_key != run_key
    should_fetch = bool(run_btn) or bool(selection_changed) or (not bool(st.session_state.rr_ran))

    # Debug (always available)
    with st.expander("Debug", expanded=False):
        st.write({
            "REPORT_URL": REPORT_URL,
            "FASTAPI_BASE_URL": FASTAPI_BASE_URL,
            "company_id": company_id,
            "region": region_choice,
            "period_choice": period_choice,
            "start_period": str(start_period),
            "end_period": str(end_period),
            "selection_changed": selection_changed,
        })

    if not should_fetch and st.session_state.rr_payload is None:
        st.info("Select client / region / period and click **Run**.")
        return

    # ----------------------
    # FETCH
    # ----------------------
    if should_fetch:
        metric_map = {
            "count_in": "footfall",
            "turnover": "turnover",
            "transactions": "transactions",
            "conversion_rate": "conversion_rate",
            "sales_per_visitor": "sales_per_visitor",
            "sales_per_transaction": "sales_per_transaction",
        }

        all_shop_ids = merged["id"].dropna().astype(int).unique().tolist()

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

        with st.spinner("Fetching KPI data…"):
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
                    st.write("Params:", params_preview)
                return
            except requests.exceptions.RequestException as e:
                st.error(f"❌ RequestException from /get-report: {e}")
                with st.expander("🔧 Debug request (params)"):
                    st.write("Params:", params_preview)
                return

        df_norm = normalize_vemcount_response(resp, kpi_keys=metric_map.keys()).rename(columns=metric_map)
        if df_norm is None or df_norm.empty:
            st.warning("No data returned for the current selection.")
            return

        # detect store key
        store_key_col = None
        for cand in ["shop_id", "id", "location_id"]:
            if cand in df_norm.columns:
                store_key_col = cand
                break
        if store_key_col is None:
            st.error("No store-id column found in response (shop_id/id/location_id).")
            return

        df_norm[store_key_col] = pd.to_numeric(df_norm[store_key_col], errors="coerce").astype("Int64")
        df_norm["date"] = pd.to_datetime(df_norm["date"], errors="coerce")
        df_norm = df_norm.dropna(subset=["date", store_key_col])

        # merge region/store labels into df_norm
        dim_cols = ["id", "region", "store_display"]
        if "store_type" in merged.columns:
            dim_cols.append("store_type")

        dim = merged[dim_cols].drop_duplicates().copy()
        dim["id"] = pd.to_numeric(dim["id"], errors="coerce").astype("Int64")

        df = df_norm.merge(dim, left_on=store_key_col, right_on="id", how="left")

        # filter to region
        df_region = df[df["region"] == region_choice].copy()
        if df_region.empty:
            st.warning("No rows for this region in the selected period.")
            return

        st.session_state.rr_last_key = run_key
        st.session_state.rr_payload = {
            "df_region": df_region,
            "df_all": df,
            "store_key_col": store_key_col,
            "selected_client": selected_client,
            "region_choice": region_choice,
            "start_period": start_period,
            "end_period": end_period,
        }
        st.session_state.rr_ran = True

    payload = st.session_state.rr_payload
    if payload is None:
        st.info("Select client / region / period and click **Run**.")
        return

    df_region = payload["df_region"]
    df_all = payload["df_all"]
    start_period = payload["start_period"]
    end_period = payload["end_period"]
    region_choice = payload["region_choice"]
    selected_client = payload["selected_client"]

    st.markdown(f"### {selected_client['brand']} — Region **{region_choice}** · {start_period} → {end_period}")

    # Region KPI rollup
    region_k = compute_store_kpis(df_region)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Footfall", fmt_int(region_k["footfall"]))
    k2.metric("Revenue", fmt_eur(region_k["turnover"]))
    k3.metric("Conversion", fmt_pct(region_k["conversion_rate"]))
    k4.metric("SPV", fmt_eur(region_k["sales_per_visitor"]))

    st.markdown("---")

    # Benchmarks: region vs company (same period)
    company_k = compute_store_kpis(df_all)

    # Build per-store table
    stores = (
        df_region.groupby(["store_display", "id"], as_index=False)
        .apply(lambda g: pd.Series(compute_store_kpis(g)))
        .reset_index(drop=True)
    )

    # Add indices vs region benchmark
    def idx(a, b):
        return (a / b * 100.0) if (pd.notna(a) and pd.notna(b) and float(b) != 0.0) else np.nan

    stores["CR idx vs region"] = stores.apply(lambda r: idx(r["conversion_rate"], region_k["conversion_rate"]), axis=1)
    stores["SPV idx vs region"] = stores.apply(lambda r: idx(r["sales_per_visitor"], region_k["sales_per_visitor"]), axis=1)
    stores["ATV idx vs region"] = stores.apply(lambda r: idx(r["sales_per_transaction"], region_k["sales_per_transaction"]), axis=1)

    # Rank "pain" (lower idx is worse)
    stores["worst_idx"] = stores[["CR idx vs region", "SPV idx vs region", "ATV idx vs region"]].min(axis=1, skipna=True)
    stores = stores.sort_values("worst_idx", ascending=True).reset_index(drop=True)

    # Optional: OpenAI client
    openai_client = None
    if OpenAI is not None:
        api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()
        if api_key:
            try:
                openai_client = OpenAI(api_key=api_key)
            except Exception:
                openai_client = None

    # Staffing placeholder detection (only if present in df_region columns)
    staffing_available = any(c.lower() in [x.lower() for x in df_region.columns] for c in ["staff", "staffing", "fte", "employees"])

    # UI: select a store to see workload output
    st.subheader("Agentic workload — per store")

    store_labels = (stores["store_display"] + " · " + stores["id"].astype(str)).tolist()
    pick = st.selectbox("Pick store", store_labels, index=0, key="rr_store_pick")
    chosen_id = int(pick.split("·")[-1].strip())

    df_store = df_region[df_region["id"].astype("Int64") == chosen_id].copy()
    store_k = compute_store_kpis(df_store)

    left, right = st.columns([1.2, 1.0], vertical_alignment="top")
    with left:
        st.markdown("#### Store KPI snapshot")
        a, b, c, d = st.columns(4)
        a.metric("Footfall", fmt_int(store_k["footfall"]))
        b.metric("Revenue", fmt_eur(store_k["turnover"]))
        c.metric("CR", fmt_pct(store_k["conversion_rate"]))
        d.metric("SPV", fmt_eur(store_k["sales_per_visitor"]))

        st.caption("Benchmark used for actions: **Region average (same period)**. (Company KPI is available for context.)")

    # Build staffing context only if present
    staffing_k = {"available": False}
    if staffing_available:
        staffing_k = {"available": True, "coverage_idx": np.nan}  # extend later with real logic

    actions = pick_best_actions(store_k, region_k, staffing_k=staffing_k)

    with right:
        st.markdown("#### Recommended actions (data-driven)")
        for i, ac in enumerate(actions[:3], start=1):
            st.markdown(f"**{i}. {ac['action']}**")
            st.write(f"- Why: {ac['why']}")
            st.write(f"- Test: {ac['how_to_test']}")

    # Executive summary (LLM optional)
    ctx = {
        "brand": selected_client["brand"],
        "region": region_choice,
        "period": {"start": str(start_period), "end": str(end_period)},
        "store_id": chosen_id,
        "store_kpis": {k: float(v) if pd.notna(v) else None for k, v in store_k.items()},
        "region_kpis": {k: float(v) if pd.notna(v) else None for k, v in region_k.items()},
        "company_kpis": {k: float(v) if pd.notna(v) else None for k, v in company_k.items()},
        "actions": actions[:2],
        "staffing_available": staffing_available,
    }

    summary = llm_summarize(openai_client, ctx)
    if summary:
        st.markdown("#### Executive summary (LLM)")
        st.write(summary)

    st.markdown("---")
    st.subheader("Region store list (sorted by worst index vs region)")
    show = stores.copy()
    show["Revenue"] = show["turnover"].apply(fmt_eur)
    show["Footfall"] = show["footfall"].apply(fmt_int)
    show["CR"] = show["conversion_rate"].apply(fmt_pct)
    show["SPV"] = show["sales_per_visitor"].apply(fmt_eur)
    show["ATV"] = show["sales_per_transaction"].apply(fmt_eur)
    for col in ["CR idx vs region", "SPV idx vs region", "ATV idx vs region", "worst_idx"]:
        show[col] = pd.to_numeric(show[col], errors="coerce").apply(lambda x: "-" if pd.isna(x) else f"{float(x):.0f}%")

    show = show.rename(columns={"store_display": "Store", "id": "Store ID"})
    show = show[["Store", "Store ID", "Revenue", "Footfall", "CR", "SPV", "ATV", "CR idx vs region", "SPV idx vs region", "ATV idx vs region", "worst_idx"]]
    st.dataframe(show, use_container_width=True, hide_index=True)


main()