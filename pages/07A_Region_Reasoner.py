# pages/06D_Region_Agentic_Workload.py
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ---------- Defensive imports (match your repo) ----------
# Period helpers (you already have these)
try:
    from helpers_periods import PRESETS, default_date_range
except Exception:
    PRESETS = [
        ("last_week", "Last week"),
        ("this_month", "This month"),
        ("last_month", "Last month"),
        ("this_quarter", "This quarter"),
        ("last_quarter", "Last quarter"),
        ("this_year", "This year"),
        ("last_year", "Last year"),
        ("date", "Custom date range"),
    ]

    from datetime import date, timedelta

    def default_date_range(days: int = 28):
        end = date.today()
        start = end - timedelta(days=days)
        return start, end

# Vemcount / FastAPI helpers (you already have these)
# Expect: build_report_params(...) and fetch_report(...)
build_report_params = None
fetch_report = None
normalize_vemcount_response = None

try:
    from helpers_vemcount_api import build_report_params, fetch_report
    # sometimes normalize is in same helper
    try:
        from helpers_vemcount_api import normalize_vemcount_response
    except Exception:
        normalize_vemcount_response = None
except Exception:
    build_report_params = None
    fetch_report = None

# Optional: shop mapping helper (if you already have it somewhere)
get_locations_by_company = None
try:
    # common pattern in your stack
    from vemcount import get_locations_by_company  # type: ignore
except Exception:
    try:
        from shop_mapping import get_locations_by_company  # type: ignore
    except Exception:
        get_locations_by_company = None

# OpenAI (you said integration is ready; we use secrets)
from openai import OpenAI


# ---------- UI / formatting ----------
def fmt_eur(x: Any) -> str:
    try:
        x = float(x)
    except Exception:
        return "-"
    s = f"{x:,.0f}"
    # EU format: 1.234.567
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"€{s}"


def fmt_pct(x: Any) -> str:
    try:
        x = float(x)
    except Exception:
        return "-"
    s = f"{x*100:.1f}".replace(".", ",")
    return f"{s}%"


def name_for(shop_id: int, mapping: Dict[int, str]) -> str:
    return mapping.get(int(shop_id), f"Shop {shop_id}")


# ---------- Agent: normalize -> score -> brief ----------
def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def compute_rollup(df: pd.DataFrame) -> Dict[str, Any]:
    for col in ["count_in", "turnover", "conversion_rate", "sales_per_visitor"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = _safe_num(df[col])

    footfall = float(df["count_in"].sum(skipna=True)) if "count_in" in df.columns else float("nan")
    turnover = float(df["turnover"].sum(skipna=True)) if "turnover" in df.columns else float("nan")
    conv = float(df["conversion_rate"].mean(skipna=True)) if "conversion_rate" in df.columns else float("nan")

    spv = float(df["sales_per_visitor"].mean(skipna=True)) if df["sales_per_visitor"].dropna().shape[0] else float("nan")
    if np.isnan(spv) and footfall and footfall > 0 and not np.isnan(turnover):
        spv = turnover / footfall

    return {
        "footfall": footfall,
        "turnover": turnover,
        "conversion_rate": conv,
        "sales_per_visitor": spv,
        "n_stores": int(df["shop_id"].nunique()) if "shop_id" in df.columns else 0,
    }


def compute_opportunity_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Doel: een simpele, robuuste ranking zonder charts.
    Opportunity = (veel traffic + betekenisvolle omzet) + (laag conv/spv = upside).
    """
    work = df.copy()

    for col in ["count_in", "turnover", "conversion_rate", "sales_per_visitor"]:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = _safe_num(work[col])

    # SPV fallback per row indien leeg
    if work["sales_per_visitor"].dropna().empty:
        with np.errstate(divide="ignore", invalid="ignore"):
            work["sales_per_visitor"] = work["turnover"] / work["count_in"]

    agg = work.groupby("shop_id", as_index=False).agg(
        footfall=("count_in", "sum"),
        turnover=("turnover", "sum"),
        conv=("conversion_rate", "mean"),
        spv=("sales_per_visitor", "mean"),
        days=("date", "nunique") if "date" in work.columns else ("shop_id", "size"),
    )

    # z-scores
    for c in ["footfall", "turnover", "conv", "spv"]:
        x = agg[c].astype(float)
        agg[c + "_z"] = (x - x.mean()) / (x.std(ddof=0) + 1e-9)

    agg["opportunity_score"] = (
        0.45 * agg["footfall_z"] +
        0.20 * agg["turnover_z"] -
        0.20 * agg["conv_z"] -
        0.15 * agg["spv_z"]
    )

    return agg.sort_values("opportunity_score", ascending=False)


def llm_brief(
    api_key: str,
    model: str,
    region_kpis: Dict[str, Any],
    opportunities: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> str:
    system = """Je bent een scherpe Retail Performance Analyst voor een regiomanager.
Schrijf in het Nederlands. Wees concreet, actiegericht en kort. Geen managementfluff.

Output EXACT in deze structuur:
1) Samenvatting (max 5 bullets)
2) Belangrijkste drivers (max 5 bullets)
3) Acties (3–7 bullets, elk met: wat / waarom / hoe je succes meet)
4) Vragen om de analyse te verscherpen (max 3 bullets)

Regels:
- Gebruik cijfers uit de input waar mogelijk.
- Verzin geen KPI’s die niet in de input zitten.
- Als data ontbreekt: benoem dat expliciet en stel 1–2 gerichte vragen.
"""

    client = OpenAI(api_key=api_key)

    payload = {
        "region_kpis": region_kpis,
        "top_opportunities": opportunities,
        "meta": meta,
        "tone": "regiomanager-brief, prioriteit, next actions",
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": "Input (JSON):\n" + json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.25,
    )
    return resp.choices[0].message.content.strip()


# ---------- Data fetch wrappers ----------
def fetch_company_mapping(api_url: str, company_id: int) -> Dict[int, str]:
    if get_locations_by_company is None:
        return {}
    try:
        m = get_locations_by_company(api_url, company_id)  # expected {shop_id: name}
        if isinstance(m, dict):
            # ensure int keys
            out = {}
            for k, v in m.items():
                try:
                    out[int(k)] = str(v)
                except Exception:
                    continue
            return out
        return {}
    except Exception:
        return {}


def normalize_payload(payload: Dict[str, Any]) -> pd.DataFrame:
    # Prefer your existing normalizer if present
    if callable(normalize_vemcount_response):
        rows = normalize_vemcount_response(payload)
        return pd.DataFrame(rows)

    # Fallback (defensive) for common nested shapes
    rows: List[Dict[str, Any]] = []
    data = payload.get("data")
    if isinstance(data, dict):
        for date_key, by_shop in data.items():
            if not isinstance(by_shop, dict):
                continue
            for shop_id, metrics in by_shop.items():
                if not isinstance(metrics, dict):
                    continue
                # hour-shape
                if "dates" in metrics and isinstance(metrics["dates"], dict):
                    for ts, ts_payload in metrics["dates"].items():
                        d = ts_payload.get("data") if isinstance(ts_payload, dict) else None
                        if isinstance(d, dict):
                            row = {"date": date_key, "timestamp": ts, "shop_id": int(shop_id)}
                            row.update(d)
                            rows.append(row)
                else:
                    row = {"date": date_key, "shop_id": int(shop_id)}
                    row.update(metrics)
                    rows.append(row)
    return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="Region Agentic Workload", page_icon="🧠", layout="wide")

    st.title("🧠 Region Agentic Workload (no charts)")
    st.caption("Nieuwe test-tool naast je huidige dashboard: **fetch → reason → brief → next questions**. Geen visualisatie, alleen beslisoutput.")

    api_url = st.secrets.get("API_URL", os.getenv("API_URL", "")).strip()
    if not api_url:
        st.error("API_URL ontbreekt in Streamlit secrets.")
        st.stop()

    openai_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()
    openai_model = st.secrets.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini")).strip()

    if build_report_params is None or fetch_report is None:
        st.error("Ik kan helpers_vemcount_api.build_report_params / fetch_report niet importeren. Bestandsnaam of exports wijken af.")
        st.stop()

    col1, col2, col3 = st.columns([1.0, 1.2, 0.8], vertical_alignment="center")
    with col1:
        company_id = st.number_input("Company ID", min_value=1, value=4556, step=1)

    with col2:
        # support both tuple presets or dataclass presets
        if PRESETS and isinstance(PRESETS[0], tuple):
            preset_key = st.selectbox("Period", [p[0] for p in PRESETS], index=2)
            preset_label = dict(PRESETS).get(preset_key, preset_key)
        else:
            # dataclass-like: has .key and .label
            preset_key = st.selectbox("Period", [p.key for p in PRESETS], index=2)  # type: ignore
            preset_label = next((p.label for p in PRESETS if p.key == preset_key), preset_key)  # type: ignore

        period = preset_key

    with col3:
        run = st.button("Run agent")

    # date inputs only when needed
    date_from, date_to = default_date_range(28)
    if period == "date":
        cA, cB = st.columns(2)
        with cA:
            d1 = st.date_input("Date from", value=date_from)
        with cB:
            d2 = st.date_input("Date to", value=date_to)
        date_from_s, date_to_s = d1.isoformat(), d2.isoformat()
    else:
        date_from_s, date_to_s = None, None

    st.divider()

    # Keep a stable cache key
    selection_key = f"{company_id}|{period}|{date_from_s}|{date_to_s}"

    if "agentic_last_key" not in st.session_state:
        st.session_state["agentic_last_key"] = None
    if "agentic_df" not in st.session_state:
        st.session_state["agentic_df"] = None
    if "agentic_mapping" not in st.session_state:
        st.session_state["agentic_mapping"] = {}

    selection_changed = st.session_state["agentic_last_key"] != selection_key
    should_run = run or st.session_state["agentic_df"] is None or selection_changed

    with st.expander("Debug (imports / selection)", expanded=False):
        st.write({
            "api_url": api_url,
            "company_id": company_id,
            "period": period,
            "date_from": date_from_s,
            "date_to": date_to_s,
            "selection_key": selection_key,
            "selection_changed": selection_changed,
            "helpers_imported": {
                "build_report_params": bool(build_report_params),
                "fetch_report": bool(fetch_report),
                "normalize_vemcount_response": bool(normalize_vemcount_response),
                "get_locations_by_company": bool(get_locations_by_company),
            },
        })

    if not should_run:
        df = st.session_state["agentic_df"]
    else:
        # Step 0: mapping
        mapping = fetch_company_mapping(api_url, int(company_id))
        st.session_state["agentic_mapping"] = mapping

        # Determine shop_ids
        shop_ids = list(mapping.keys())
        if not shop_ids:
            st.warning("Geen shop mapping beschikbaar via company endpoint. Vul shop IDs handmatig in (tijdelijk).")
            manual = st.text_input("Shop IDs (comma separated)", value="26304,26305")
            shop_ids = [int(x.strip()) for x in manual.split(",") if x.strip().isdigit()]

        # Step 1: fetch
        data_output = ["count_in", "turnover", "conversion_rate", "sales_per_visitor"]
        params = build_report_params(
            data=shop_ids,
            data_output=data_output,
            period=period,
            period_step="day",     # your standard for this agentic page
            source="shops",
            date_from=date_from_s,
            date_to=date_to_s,
        )

        with st.status("Agent is running…", expanded=True) as status:
            st.write("**Step 1 — Fetch**: KPI’s ophalen via FastAPI → Vemcount")
            payload = fetch_report(api_url, params=params)

            st.write("**Step 2 — Normalize**: response omzetten naar platte tabel")
            df = normalize_payload(payload)

            if df.empty:
                status.update(label="Agent failed (no data)", state="error")
                st.error("Lege dataset. Check shop_ids, periode, of API response.")
                st.stop()

            # Coerce types
            for col in ["count_in", "turnover", "conversion_rate", "sales_per_visitor"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            st.write("**Step 3 — Reason**: opportunities ranken (impact + upside)")
            scores = compute_opportunity_table(df)

            st.write("**Step 4 — Brief**: LLM maakt regiomanager-brief + acties")
            region_kpis = compute_rollup(df)
            region_kpis.update({
                "period": period,
                "date_from": date_from_s,
                "date_to": date_to_s,
            })

            # Build top opportunities payload
            mapping = st.session_state["agentic_mapping"]
            top = scores.head(7).copy()
            opp_list: List[Dict[str, Any]] = []
            for _, r in top.iterrows():
                sid = int(r["shop_id"])
                opp_list.append({
                    "location": name_for(sid, mapping),
                    "shop_id": sid,
                    "footfall": float(r.get("footfall", np.nan)),
                    "turnover": float(r.get("turnover", np.nan)),
                    "conversion_rate": float(r.get("conv", np.nan)),
                    "sales_per_visitor": float(r.get("spv", np.nan)),
                    "opportunity_score": float(r.get("opportunity_score", np.nan)),
                })

            # Save to session (so you can rerun only LLM if needed)
            st.session_state["agentic_df"] = df
            st.session_state["agentic_last_key"] = selection_key

            # LLM call
            if not openai_key:
                status.update(label="Agent ready (no OpenAI key)", state="complete")
                st.warning("OPENAI_API_KEY ontbreekt in secrets. Ik kan wel de opportunities tonen, maar geen brief genereren.")
                brief = None
            else:
                brief = llm_brief(
                    api_key=openai_key,
                    model=openai_model,
                    region_kpis=region_kpis,
                    opportunities=opp_list,
                    meta={"tool": "Region Agentic Workload", "preset_label": preset_label},
                )
                status.update(label="Agent complete ✅", state="complete")

        # Output (no charts)
        st.subheader("📌 Region rollup")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Footfall", f"{int(region_kpis['footfall']):,}".replace(",", ".") if not np.isnan(region_kpis["footfall"]) else "-")
        c2.metric("Turnover", fmt_eur(region_kpis["turnover"]))
        c3.metric("Conversion", fmt_pct(region_kpis["conversion_rate"]))
        c4.metric("SPV", fmt_eur(region_kpis["sales_per_visitor"]))

        st.subheader("🎯 Top opportunities (agent output)")
        show = top.copy()
        show["location"] = show["shop_id"].apply(lambda x: name_for(int(x), st.session_state["agentic_mapping"]))
        show = show[["location", "shop_id", "footfall", "turnover", "conv", "spv", "opportunity_score"]]
        show.rename(columns={
            "location": "Locatie",
            "shop_id": "ShopID",
            "footfall": "Bezoekers",
            "turnover": "Omzet",
            "conv": "Conversie",
            "spv": "SPV",
            "opportunity_score": "OppScore",
        }, inplace=True)

        # format
        show_fmt = show.copy()
        show_fmt["Bezoekers"] = show_fmt["Bezoekers"].map(lambda x: f"{int(x):,}".replace(",", ".") if pd.notna(x) else "-")
        show_fmt["Omzet"] = show_fmt["Omzet"].map(fmt_eur)
        show_fmt["Conversie"] = show_fmt["Conversie"].map(fmt_pct)
        show_fmt["SPV"] = show_fmt["SPV"].map(fmt_eur)
        show_fmt["OppScore"] = show_fmt["OppScore"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")

        st.dataframe(show_fmt, use_container_width=True, hide_index=True)

        st.subheader("🧠 AI brief (agentic)")
        if brief:
            st.markdown(brief)
        else:
            st.info("Geen brief (OPENAI_API_KEY ontbreekt of call overgeslagen).")

        with st.expander("Raw df preview (first 200 rows)", expanded=False):
            st.dataframe(df.head(200), use_container_width=True)

    # If already cached and you didn’t run: show minimal hint
    if st.session_state["agentic_df"] is not None and not should_run:
        st.info("Deze page gebruikt cached data. Klik **Run agent** om opnieuw te draaien met de huidige selectie.")


if __name__ == "__main__":
    main()