# pages/06D_Region_Agentic_Workload_v2.py
from __future__ import annotations

import os
import json
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI


# =========================
# Imports from your repo
# =========================
build_report_params = None
fetch_report = None

try:
    # Expected in your repo
    from helpers_vemcount_api import build_report_params, fetch_report  # type: ignore
except Exception:
    build_report_params = None
    fetch_report = None

# Optional: your existing normalizer (if present anywhere)
normalize_vemcount_response = None
try:
    from helpers_vemcount_api import normalize_vemcount_response  # type: ignore
except Exception:
    normalize_vemcount_response = None


# =========================
# Formatting (EU)
# =========================
def fmt_eur(x: Any) -> str:
    try:
        x = float(x)
    except Exception:
        return "-"
    s = f"{x:,.0f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")  # EU
    return f"€{s}"


def fmt_pct(x: Any) -> str:
    try:
        x = float(x)
    except Exception:
        return "-"
    return f"{x*100:.1f}%".replace(".", ",")


def fmt_int(x: Any) -> str:
    try:
        x = int(float(x))
    except Exception:
        return "-"
    return f"{x:,}".replace(",", ".")


# =========================
# Regions.csv loader (same idea as your existing tool)
# =========================
def _find_regions_csv() -> Optional[str]:
    candidates = [
        "regions.csv",
        "data/regions.csv",
        "assets/regions.csv",
        "config/regions.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_regions_mapping() -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
    """
    Expected structure (flexible):
      - one row per shop, with a region column and a shop_id column
    Tries common column names:
      region: region, region_name, regio
      shop_id: shop_id, location_id, store_id, shop, location
    """
    path = _find_regions_csv()
    if not path:
        raise FileNotFoundError(
            "regions.csv not found. Expected at repo root or data/assets/config folders."
        )

    df = pd.read_csv(path)

    # region column
    region_col = None
    for c in ["region", "region_name", "regio", "Region", "Regio"]:
        if c in df.columns:
            region_col = c
            break
    if not region_col:
        raise ValueError("regions.csv: could not find a region column (region/region_name/regio).")

    # shop column
    shop_col = None
    for c in ["shop_id", "location_id", "store_id", "shop", "location", "ShopID", "LocationID"]:
        if c in df.columns:
            shop_col = c
            break
    if not shop_col:
        raise ValueError("regions.csv: could not find a shop column (shop_id/location_id/store_id).")

    # coerce shop ids to ints
    df = df.copy()
    df[shop_col] = pd.to_numeric(df[shop_col], errors="coerce").astype("Int64")

    region_to_shops: Dict[str, List[int]] = {}
    for region, g in df.groupby(region_col):
        shops = [int(x) for x in g[shop_col].dropna().unique().tolist()]
        region_to_shops[str(region)] = sorted(shops)

    return df, region_to_shops


# =========================
# Period selector (same outputs as your other tool: presets -> real dates)
# =========================
PERIOD_PRESETS = [
    ("last_week", "Last week"),
    ("this_month", "This month"),
    ("last_month", "Last month"),
    ("this_quarter", "This quarter"),
    ("last_quarter", "Last quarter"),
    ("this_year", "This year"),
    ("last_year", "Last year"),
    ("date", "Custom range"),
]


def start_of_month(d: date) -> date:
    return date(d.year, d.month, 1)


def end_of_month(d: date) -> date:
    # next month - 1 day
    if d.month == 12:
        nm = date(d.year + 1, 1, 1)
    else:
        nm = date(d.year, d.month + 1, 1)
    return nm - timedelta(days=1)


def quarter_start(d: date) -> date:
    q = (d.month - 1) // 3 + 1
    m = (q - 1) * 3 + 1
    return date(d.year, m, 1)


def quarter_end(d: date) -> date:
    qs = quarter_start(d)
    # add 3 months then -1 day
    if qs.month == 10:
        nm = date(qs.year + 1, 1, 1)
    else:
        nm = date(qs.year, qs.month + 3, 1)
    return nm - timedelta(days=1)


def resolve_preset_to_dates(preset: str, today: Optional[date] = None) -> Tuple[date, date]:
    today = today or date.today()

    if preset == "last_week":
        # last full 7 days ending yesterday
        end = today - timedelta(days=1)
        start = end - timedelta(days=6)
        return start, end

    if preset == "this_month":
        return start_of_month(today), today

    if preset == "last_month":
        last_month_end = start_of_month(today) - timedelta(days=1)
        return start_of_month(last_month_end), end_of_month(last_month_end)

    if preset == "this_quarter":
        return quarter_start(today), today

    if preset == "last_quarter":
        this_q_start = quarter_start(today)
        last_q_end = this_q_start - timedelta(days=1)
        return quarter_start(last_q_end), quarter_end(last_q_end)

    if preset == "this_year":
        return date(today.year, 1, 1), today

    if preset == "last_year":
        return date(today.year - 1, 1, 1), date(today.year - 1, 12, 31)

    # fallback
    return today - timedelta(days=27), today


# =========================
# Normalization (fallback if your helper isn't importable here)
# =========================
def normalize_payload(payload: Dict[str, Any]) -> pd.DataFrame:
    if callable(normalize_vemcount_response):
        rows = normalize_vemcount_response(payload)  # type: ignore
        return pd.DataFrame(rows)

    rows: List[Dict[str, Any]] = []
    data = payload.get("data")
    if not isinstance(data, dict):
        return pd.DataFrame()

    for date_key, by_shop in data.items():
        if not isinstance(by_shop, dict):
            continue
        for shop_id, metrics in by_shop.items():
            if not isinstance(metrics, dict):
                continue
            # hour-like nested
            if "dates" in metrics and isinstance(metrics["dates"], dict):
                for ts, ts_payload in metrics["dates"].items():
                    if isinstance(ts_payload, dict) and isinstance(ts_payload.get("data"), dict):
                        r = {"date": date_key, "timestamp": ts, "shop_id": int(shop_id)}
                        r.update(ts_payload["data"])
                        rows.append(r)
            else:
                r = {"date": date_key, "shop_id": int(shop_id)}
                r.update(metrics)
                rows.append(r)

    return pd.DataFrame(rows)


# =========================
# Agentic reasoning (no category assumptions, no staffing unless fields exist)
# =========================
def compute_store_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ["count_in", "turnover", "conversion_rate", "sales_per_visitor"]:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")

    # SPV fallback if missing
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

    # opportunity score (traffic + turnover) - (conv + spv)
    for c in ["footfall", "turnover", "conv", "spv"]:
        x = agg[c].astype(float)
        agg[c + "_z"] = (x - x.mean()) / (x.std(ddof=0) + 1e-9)

    agg["opportunity_score"] = (
        0.45 * agg["footfall_z"]
        + 0.20 * agg["turnover_z"]
        - 0.20 * agg["conv_z"]
        - 0.15 * agg["spv_z"]
    )

    return agg.sort_values("opportunity_score", ascending=False)


def classify_lever(row: pd.Series, med: Dict[str, float]) -> Dict[str, Any]:
    footfall = float(row.get("footfall", np.nan))
    conv = float(row.get("conv", np.nan))
    spv = float(row.get("spv", np.nan))

    hi_traffic = (not np.isnan(footfall) and footfall >= med["footfall"])
    lo_traffic = (not np.isnan(footfall) and footfall < med["footfall"])
    lo_conv = (not np.isnan(conv) and conv < med["conv"])
    hi_conv = (not np.isnan(conv) and conv >= med["conv"])
    lo_spv = (not np.isnan(spv) and spv < med["spv"])

    if hi_traffic and lo_conv:
        return {
            "lever": "conversion",
            "why": "Veel bezoekers, conversie onder mediane store performance → grootste upside is 'traffic → kopers'.",
            "test": [
                "Kies 2 piekmomenten en test 1 micro-interventie per moment (1 wijziging tegelijk).",
                "Universele opties (geen categorie-aanname): snellere begroeting, frictie wegnemen bij kassa/queue, duidelijke hulpvraag-punten.",
                "Meet: conversie per dag/uur + SPV. Stop wat niet werkt na 1 week."
            ],
            "needs": [
                "Transactiecount/ATV (dan kunnen we exact sturen op driver).",
                "Store-type/format (mall/high-street/retail park) voor scherpere acties."
            ],
        }

    if hi_traffic and hi_conv and lo_spv:
        return {
            "lever": "spv",
            "why": "Conversie oké, SPV onder median → er wordt gekocht, maar waarde per bezoeker blijft achter.",
            "test": [
                "Test 1 ‘value-driver’ zonder categorie-aanname: 1 vaste adviesprompt + 1 duidelijke 'best value' zone.",
                "Meet: SPV + (ATV/items als beschikbaar) + conversie."
            ],
            "needs": [
                "ATV/items per transactie (mix vs bundling effect).",
                "Promo/event tags (verklaart SPV-schommelingen)."
            ],
        }

    if lo_traffic and hi_conv:
        return {
            "lever": "capture/traffic",
            "why": "Conversie sterk, traffic relatief laag → groei zit waarschijnlijk in meer instroom/capture.",
            "test": [
                "Check openingstijden vs drukte, entree-frictie en 1 duidelijke 'reason to enter'.",
                "Als je passanten/capture hebt: test 1 entree-aanpassing en meet capture rate."
            ],
            "needs": [
                "Passanten/capture rate (hard bewijs i.p.v. aannames).",
                "Locatiecontext (straat/winkelcentrum/retail park)."
            ],
        }

    if lo_traffic and lo_conv:
        return {
            "lever": "diagnose",
            "why": "Traffic én conversie onder median → eerst grootste frictiebron vinden, dan pas optimaliseren.",
            "test": [
                "1 week frictie-log (top 5 redenen afhaken) + daily KPI’s.",
                "Kopieer patronen van dagen die wél goed scoren."
            ],
            "needs": [
                "Event tags / kwalitatieve store-notes.",
                "Transactiecount/ATV voor driver-scherpte."
            ],
        }

    return {
        "lever": "unknown",
        "why": "KPI’s wijzen niet eenduidig één hefboom aan (of er missen velden).",
        "test": [
            "Start met 1 week baseline + 1 simpele notitie per dag (promo/weer/crew/incident).",
            "Daarna: 1 wijziging per week testen."
        ],
        "needs": [
            "Transactiecount, ATV, en (optioneel) passanten/capture.",
            "Store-type/format + primair doel."
        ],
    }


def llm_region_brief(api_key: str, model: str, region_kpis: Dict[str, Any], top_opps: List[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    system = """Je bent een scherpe Retail Performance Analyst voor een regiomanager.
Schrijf Nederlands, actiegericht, kort. Geen fluff.

GUARDRAILS:
- Geen productcategorie/branche aannames (geen parfum-upsell etc.).
- Noem geen staffing/roostering tenzij meta.staffing_fields niet leeg is.
- Verzin geen KPI’s die niet in input zitten.

Output EXACT:
1) Samenvatting (max 5 bullets)
2) Drivers (max 5 bullets)
3) Acties (3–7 bullets: wat / waarom / hoe meten)
4) 3 Vragen om dit scherper te maken
"""
    client = OpenAI(api_key=api_key)
    payload = {"region_kpis": region_kpis, "top_opportunities": top_opps, "meta": meta}
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": "Input (JSON):\n" + json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.25,
    )
    return resp.choices[0].message.content.strip()


# =========================
# Page
# =========================
def main():
    st.set_page_config(page_title="Region Agentic Workload v2", page_icon="🧠", layout="wide")
    st.title("🧠 Region Agentic Workload v2 (same selectors, no charts)")
    st.caption("Zelfde input/selector-flow als de bestaande tool, maar output is 100% agentic (brief + next actions).")

    if build_report_params is None or fetch_report is None:
        st.error("helpers_vemcount_api.build_report_params/fetch_report niet gevonden. Check exports/filenames.")
        st.stop()

    # API_URL: in jouw debug is dit al inclusief /get-report → dus we nemen het zoals het is
    api_url = st.secrets.get("API_URL", os.getenv("API_URL", "")).strip()
    if not api_url:
        st.error("API_URL ontbreekt in Streamlit secrets.")
        st.stop()

    openai_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()
    openai_model = st.secrets.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini")).strip()

    # Load regions.csv mapping
    try:
        regions_df, region_to_shops = load_regions_mapping()
    except Exception as e:
        st.error(f"Kon regions.csv mapping niet laden: {e}")
        st.stop()

    # Header selectors (mirror the standard flow: Region -> Period -> Run)
    h1, h2, h3 = st.columns([1.6, 1.2, 0.7], vertical_alignment="center")
    with h1:
        region = st.selectbox("Region", sorted(region_to_shops.keys()))
    with h2:
        preset = st.selectbox("Period", [p[0] for p in PERIOD_PRESETS], index=2)
    with h3:
        run = st.button("Run agent")

    # Resolve dates (ALWAYS) so debug shows real window and LLM has context
    if preset == "date":
        cA, cB = st.columns(2)
        default_start, default_end = resolve_preset_to_dates("last_month")
        with cA:
            d_from = st.date_input("Date from", value=default_start)
        with cB:
            d_to = st.date_input("Date to", value=default_end)
        date_from_s, date_to_s = d_from.isoformat(), d_to.isoformat()
    else:
        d_from, d_to = resolve_preset_to_dates(preset)
        date_from_s, date_to_s = d_from.isoformat(), d_to.isoformat()

    # Shops from region
    shop_ids = region_to_shops.get(region, [])
    if not shop_ids:
        st.warning("Deze regio heeft geen shop_ids in regions.csv.")
        st.stop()

    # Selection key (same idea: region+period+dates)
    selection_key = f"{region}|{preset}|{date_from_s}|{date_to_s}|nshops={len(shop_ids)}"

    # Cache
    if "agentic_v2_last_key" not in st.session_state:
        st.session_state["agentic_v2_last_key"] = None
    if "agentic_v2_df" not in st.session_state:
        st.session_state["agentic_v2_df"] = None

    selection_changed = st.session_state["agentic_v2_last_key"] != selection_key
    should_fetch = run or st.session_state["agentic_v2_df"] is None or selection_changed

    with st.expander("Debug (selection)", expanded=False):
        st.write({
            "api_url": api_url,
            "region": region,
            "preset": preset,
            "date_from": date_from_s,
            "date_to": date_to_s,
            "shop_ids_count": len(shop_ids),
            "selection_key": selection_key,
            "selection_changed": selection_changed,
            "helpers_imported": {
                "build_report_params": bool(build_report_params),
                "fetch_report": bool(fetch_report),
                "normalize_vemcount_response": bool(normalize_vemcount_response),
            },
        })

    # Fetch (IMPORTANT: use period='date' always, to guarantee date window)
    if should_fetch:
        data_output = ["count_in", "turnover", "conversion_rate", "sales_per_visitor"]
        params = build_report_params(
            data=shop_ids,
            data_output=data_output,
            source="shops",
            period="date",
            period_step="day",
            date_from=date_from_s,
            date_to=date_to_s,
        )

        with st.status("Agent running…", expanded=True) as status:
            st.write("**Step 1 — Fetch** (FastAPI → Vemcount)")
            payload = fetch_report(api_url, params=params)

            st.write("**Step 2 — Normalize**")
            df = normalize_payload(payload)

            if df.empty:
                status.update(label="No data returned", state="error")
                st.error("Lege dataset. Check regions.csv shop_ids, API response, of date window.")
                st.stop()

            # Coerce numerics
            for col in ["count_in", "turnover", "conversion_rate", "sales_per_visitor"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            status.update(label="Complete ✅", state="complete")

        st.session_state["agentic_v2_df"] = df
        st.session_state["agentic_v2_last_key"] = selection_key

    df = st.session_state["agentic_v2_df"]
    if df is None or df.empty:
        st.info("Klik Run agent om te starten.")
        st.stop()

    # Region rollup
    rollup = {
        "footfall": float(df.get("count_in", pd.Series(dtype=float)).sum(skipna=True)),
        "turnover": float(df.get("turnover", pd.Series(dtype=float)).sum(skipna=True)),
        "conversion_rate": float(df.get("conversion_rate", pd.Series(dtype=float)).mean(skipna=True)),
        "sales_per_visitor": float(df.get("sales_per_visitor", pd.Series(dtype=float)).mean(skipna=True)),
        "n_stores": int(df["shop_id"].nunique()) if "shop_id" in df.columns else 0,
        "region": region,
        "date_from": date_from_s,
        "date_to": date_to_s,
    }

    st.subheader("📌 Region rollup")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Footfall", fmt_int(rollup["footfall"]))
    c2.metric("Turnover", fmt_eur(rollup["turnover"]))
    c3.metric("Conversion", fmt_pct(rollup["conversion_rate"]))
    c4.metric("SPV", fmt_eur(rollup["sales_per_visitor"]))

    # Store aggregation + ranking
    stores = compute_store_aggregate(df)

    med = {
        "footfall": float(stores["footfall"].median(skipna=True)),
        "conv": float(stores["conv"].median(skipna=True)),
        "spv": float(stores["spv"].median(skipna=True)),
    }

    # Build a readable store label: if regions.csv has store_name, use it
    # Try to find a name column in regions_df
    name_col = None
    for c in ["store_name", "shop_name", "location_name", "name", "StoreName", "ShopName"]:
        if c in regions_df.columns:
            name_col = c
            break

    shop_name_map: Dict[int, str] = {}
    if name_col:
        # also detect shop_id column used in file
        shop_col = None
        for c in ["shop_id", "location_id", "store_id", "shop", "location", "ShopID", "LocationID"]:
            if c in regions_df.columns:
                shop_col = c
                break
        if shop_col:
            tmp = regions_df[[shop_col, name_col]].copy()
            tmp[shop_col] = pd.to_numeric(tmp[shop_col], errors="coerce")
            for _, r in tmp.dropna().iterrows():
                try:
                    shop_name_map[int(r[shop_col])] = str(r[name_col])
                except Exception:
                    pass

    def shop_label(sid: int) -> str:
        return shop_name_map.get(sid, f"Shop {sid}")

    st.subheader("🎯 Top opportunities (no charts)")
    top = stores.head(7).copy()
    top["Locatie"] = top["shop_id"].astype(int).map(shop_label)

    top_tbl = top[["Locatie", "shop_id", "footfall", "turnover", "conv", "spv", "opportunity_score"]].copy()
    top_tbl.rename(columns={
        "shop_id": "ShopID",
        "footfall": "Bezoekers",
        "turnover": "Omzet",
        "conv": "Conversie",
        "spv": "SPV",
        "opportunity_score": "OppScore",
    }, inplace=True)

    disp = top_tbl.copy()
    disp["Bezoekers"] = disp["Bezoekers"].map(fmt_int)
    disp["Omzet"] = disp["Omzet"].map(fmt_eur)
    disp["Conversie"] = disp["Conversie"].map(fmt_pct)
    disp["SPV"] = disp["SPV"].map(fmt_eur)
    disp["OppScore"] = disp["OppScore"].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else "-")

    st.dataframe(disp, use_container_width=True, hide_index=True)

    # AI Region brief
    st.subheader("🧠 AI brief (region)")
    if not openai_key:
        st.warning("OPENAI_API_KEY ontbreekt in secrets → geen AI-brief.")
    else:
        top_payload: List[Dict[str, Any]] = []
        for _, r in top.iterrows():
            sid = int(r["shop_id"])
            top_payload.append({
                "location": shop_label(sid),
                "shop_id": sid,
                "footfall": float(r.get("footfall", np.nan)),
                "turnover": float(r.get("turnover", np.nan)),
                "conversion_rate": float(r.get("conv", np.nan)),
                "sales_per_visitor": float(r.get("spv", np.nan)),
                "opportunity_score": float(r.get("opportunity_score", np.nan)),
            })

        meta = {
            "tool": "Region Agentic Workload v2",
            "guardrails": [
                "no category assumptions",
                "no staffing unless fields exist",
                "no invented KPIs",
            ],
            "staffing_fields": [],  # we keep empty unless you later add staffing data columns
        }

        with st.spinner("Generating brief…"):
            brief = llm_region_brief(
                api_key=openai_key,
                model=openai_model,
                region_kpis=rollup,
                top_opps=top_payload,
                meta=meta,
            )
        st.markdown(brief)

    # Next best actions per store (rule-based, guarded)
    st.divider()
    st.subheader("✅ Next Best Action per store (guarded, no category assumptions)")

    for _, r in top.iterrows():
        sid = int(r["shop_id"])
        loc = shop_label(sid)
        plan = classify_lever(r, med)

        with st.expander(f"{loc} (Shop {sid}) — lever: {plan['lever']}", expanded=False):
            st.markdown("**Observed (data):**")
            st.write({
                "Footfall": fmt_int(r.get("footfall")),
                "Turnover": fmt_eur(r.get("turnover")),
                "Conversion": fmt_pct(r.get("conv")),
                "SPV": fmt_eur(r.get("spv")),
                "OppScore": f"{float(r.get('opportunity_score')):.2f}" if pd.notna(r.get("opportunity_score")) else "-",
            })

            st.markdown("**Why:**")
            st.write(plan["why"])

            st.markdown("**Test (1 week):**")
            for t in plan["test"]:
                st.write(f"- {t}")

            st.markdown("**To sharpen:**")
            for n in plan["needs"]:
                st.write(f"- {n}")

    with st.expander("Raw df preview (first 200 rows)", expanded=False):
        st.dataframe(df.head(200), use_container_width=True)


if __name__ == "__main__":
    main()