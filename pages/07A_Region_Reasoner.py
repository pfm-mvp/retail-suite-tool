from __future__ import annotations

import os
import json
import inspect
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
    from helpers_vemcount_api import build_report_params, fetch_report  # type: ignore
except Exception:
    build_report_params = None
    fetch_report = None

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
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
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
# Regions.csv loader (sep=';')
# =========================
def _find_regions_csv() -> Optional[str]:
    candidates = ["data/regions.csv", "regions.csv", "assets/regions.csv", "config/regions.csv"]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_regions_mapping() -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
    path = _find_regions_csv()
    if not path:
        raise FileNotFoundError("regions.csv not found (expected at data/regions.csv).")

    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:
        # fallback
        df = pd.read_csv(path, sep=",")

    if "region" not in df.columns:
        raise ValueError(f"regions.csv: missing 'region' column. Found: {list(df.columns)}")
    if "shop_id" not in df.columns:
        raise ValueError(f"regions.csv: missing 'shop_id' column. Found: {list(df.columns)}")

    df = df.copy()
    df["shop_id"] = pd.to_numeric(df["shop_id"], errors="coerce").astype("Int64")

    region_to_shops: Dict[str, List[int]] = {}
    for region, g in df.groupby("region"):
        shops = [int(x) for x in g["shop_id"].dropna().unique().tolist()]
        region_to_shops[str(region)] = sorted(shops)

    return df, region_to_shops


# =========================
# Period -> real dates
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
    if qs.month == 10:
        nm = date(qs.year + 1, 1, 1)
    else:
        nm = date(qs.year, qs.month + 3, 1)
    return nm - timedelta(days=1)


def resolve_preset_to_dates(preset: str, today: Optional[date] = None) -> Tuple[date, date]:
    today = today or date.today()

    if preset == "last_week":
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

    return today - timedelta(days=27), today


# =========================
# build_report_params adapter (THIS FIXES YOUR ERROR)
# =========================
def build_params_adapter(
    shop_ids: List[int],
    data_output: List[str],
    source: str,
    period: str,
    period_step: str,
    date_from: str,
    date_to: str,
) -> Dict[str, Any]:
    """
    Calls your repo's build_report_params regardless of its parameter names.
    We inspect its signature and map accordingly.

    Common variants in repos:
      - data / shop_ids / ids
      - data_output / outputs / kpis
      - date_from, date_to OR from_date, to_date
    """
    if build_report_params is None:
        raise RuntimeError("build_report_params not available")

    sig = inspect.signature(build_report_params)
    params = sig.parameters

    kwargs: Dict[str, Any] = {}

    # map shop ids
    if "data" in params:
        kwargs["data"] = shop_ids
    elif "shop_ids" in params:
        kwargs["shop_ids"] = shop_ids
    elif "ids" in params:
        kwargs["ids"] = shop_ids
    elif "shops" in params:
        kwargs["shops"] = shop_ids
    else:
        # last resort: positional
        # we'll handle below
        pass

    # map outputs
    if "data_output" in params:
        kwargs["data_output"] = data_output
    elif "outputs" in params:
        kwargs["outputs"] = data_output
    elif "kpis" in params:
        kwargs["kpis"] = data_output

    # source/period/step
    if "source" in params:
        kwargs["source"] = source
    if "period" in params:
        kwargs["period"] = period
    if "period_step" in params:
        kwargs["period_step"] = period_step

    # dates
    if "date_from" in params:
        kwargs["date_from"] = date_from
    elif "from_date" in params:
        kwargs["from_date"] = date_from
    if "date_to" in params:
        kwargs["date_to"] = date_to
    elif "to_date" in params:
        kwargs["to_date"] = date_to

    # If signature doesn't accept kwargs for shop_ids, use positional fallback
    try:
        return build_report_params(**kwargs)  # type: ignore
    except TypeError:
        # positional fallback: attempt order [shop_ids, data_output, source, period, period_step, date_from, date_to]
        args = []
        # only include what signature expects
        for name in params.keys():
            if name in ("data", "shop_ids", "ids", "shops"):
                args.append(shop_ids)
            elif name in ("data_output", "outputs", "kpis"):
                args.append(data_output)
            elif name == "source":
                args.append(source)
            elif name == "period":
                args.append(period)
            elif name == "period_step":
                args.append(period_step)
            elif name in ("date_from", "from_date"):
                args.append(date_from)
            elif name in ("date_to", "to_date"):
                args.append(date_to)
            else:
                # ignore unknowns (they may have defaults)
                pass
        return build_report_params(*args)  # type: ignore


# =========================
# Normalization fallback
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
            r = {"date": date_key, "shop_id": int(shop_id)}
            r.update(metrics)
            rows.append(r)

    return pd.DataFrame(rows)


# =========================
# Main
# =========================
def main():
    st.set_page_config(page_title="Region Reasoner", page_icon="🧠", layout="wide")
    st.title("🧠 Region Reasoner (agentic workload, same selectors)")

    if build_report_params is None or fetch_report is None:
        st.error("helpers_vemcount_api.build_report_params/fetch_report niet gevonden.")
        st.stop()

    api_url = st.secrets.get("API_URL", os.getenv("API_URL", "")).strip()
    if not api_url:
        st.error("API_URL ontbreekt in secrets.")
        st.stop()

    # Regions mapping
    try:
        regions_df, region_to_shops = load_regions_mapping()
    except Exception as e:
        st.error(f"Kon regions.csv niet laden: {e}")
        st.stop()

    h1, h2, h3 = st.columns([1.6, 1.2, 0.7], vertical_alignment="center")
    with h1:
        region = st.selectbox("Region", sorted(region_to_shops.keys()))
    with h2:
        preset = st.selectbox("Period", [p[0] for p in PERIOD_PRESETS], index=2)
    with h3:
        run = st.button("Run")

    if preset == "date":
        cA, cB = st.columns(2)
        ds, de = resolve_preset_to_dates("last_month")
        with cA:
            d_from = st.date_input("Date from", value=ds)
        with cB:
            d_to = st.date_input("Date to", value=de)
    else:
        d_from, d_to = resolve_preset_to_dates(preset)

    date_from_s, date_to_s = d_from.isoformat(), d_to.isoformat()
    shop_ids = region_to_shops.get(region, [])

    selection_key = f"{region}|{preset}|{date_from_s}|{date_to_s}|n={len(shop_ids)}"
    if "rr_last_key" not in st.session_state:
        st.session_state["rr_last_key"] = None
    if "rr_df" not in st.session_state:
        st.session_state["rr_df"] = None

    selection_changed = st.session_state["rr_last_key"] != selection_key
    should_fetch = run or st.session_state["rr_df"] is None or selection_changed

    with st.expander("Debug", expanded=False):
        st.write({
            "api_url": api_url,
            "region": region,
            "preset": preset,
            "date_from": date_from_s,
            "date_to": date_to_s,
            "shop_ids_count": len(shop_ids),
            "selection_changed": selection_changed,
            "build_report_params_signature": str(inspect.signature(build_report_params)),
        })

    if should_fetch:
        data_output = ["count_in", "turnover", "conversion_rate", "sales_per_visitor"]

        params = build_params_adapter(
            shop_ids=shop_ids,
            data_output=data_output,
            source="shops",
            period="date",
            period_step="day",
            date_from=date_from_s,
            date_to=date_to_s,
        )

        with st.status("Fetching…", expanded=True) as status:
            st.write("Calling fetch_report…")
            payload = fetch_report(api_url, params=params)

            df = normalize_payload(payload)
            if df.empty:
                status.update(label="No data", state="error")
                st.error("Lege dataset. Check API response / shop ids.")
                st.stop()

            for col in ["count_in", "turnover", "conversion_rate", "sales_per_visitor"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            status.update(label="Done ✅", state="complete")

        st.session_state["rr_df"] = df
        st.session_state["rr_last_key"] = selection_key

    df = st.session_state["rr_df"]
    if df is None or df.empty:
        st.info("Klik Run om te starten.")
        st.stop()

    # Minimal output (no charts)
    st.subheader("Region rollup")
    footfall = float(df.get("count_in", pd.Series(dtype=float)).sum(skipna=True))
    turnover = float(df.get("turnover", pd.Series(dtype=float)).sum(skipna=True))
    conv = float(df.get("conversion_rate", pd.Series(dtype=float)).mean(skipna=True))
    spv = float(df.get("sales_per_visitor", pd.Series(dtype=float)).mean(skipna=True))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Footfall", fmt_int(footfall))
    c2.metric("Turnover", fmt_eur(turnover))
    c3.metric("Conversion", fmt_pct(conv))
    c4.metric("SPV", fmt_eur(spv))

    with st.expander("Raw df preview", expanded=False):
        st.dataframe(df.head(200), use_container_width=True)


if __name__ == "__main__":
    main()