from __future__ import annotations

import os
import inspect
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


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
# EU formatting
# =========================
def fmt_eur(x: Any) -> str:
    try:
        x = float(x)
    except Exception:
        return "-"
    s = f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"€{s}"


def fmt_pct(x: Any) -> str:
    try:
        x = float(x)
    except Exception:
        return "-"
    return f"{x * 100:.1f}%".replace(".", ",")


def fmt_int(x: Any) -> str:
    try:
        x = int(float(x))
    except Exception:
        return "-"
    return f"{x:,}".replace(",", ".")


# =========================
# Regions.csv (your actual format: sep=";")
# =========================
def _find_regions_csv() -> Optional[str]:
    for p in ["data/regions.csv", "regions.csv", "assets/regions.csv", "config/regions.csv"]:
        if os.path.exists(p):
            return p
    return None


def load_regions_mapping() -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
    path = _find_regions_csv()
    if not path:
        raise FileNotFoundError("regions.csv not found (expected at data/regions.csv).")

    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",")

    # Your file has: shop_id;region;sqm_override;store_type
    if "region" not in df.columns:
        raise ValueError(f"regions.csv missing 'region'. Found: {list(df.columns)}")
    if "shop_id" not in df.columns:
        raise ValueError(f"regions.csv missing 'shop_id'. Found: {list(df.columns)}")

    df = df.copy()
    df["shop_id"] = pd.to_numeric(df["shop_id"], errors="coerce").astype("Int64")

    region_to_shops: Dict[str, List[int]] = {}
    for r, g in df.groupby("region"):
        region_to_shops[str(r)] = sorted([int(x) for x in g["shop_id"].dropna().unique().tolist()])

    return df, region_to_shops


# =========================
# Period -> real date window
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
    return date(d.year, (q - 1) * 3 + 1, 1)


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
        last_end = start_of_month(today) - timedelta(days=1)
        return start_of_month(last_end), end_of_month(last_end)
    if preset == "this_quarter":
        return quarter_start(today), today
    if preset == "last_quarter":
        this_q = quarter_start(today)
        last_end = this_q - timedelta(days=1)
        return quarter_start(last_end), quarter_end(last_end)
    if preset == "this_year":
        return date(today.year, 1, 1), today
    if preset == "last_year":
        return date(today.year - 1, 1, 1), date(today.year - 1, 12, 31)
    return today - timedelta(days=27), today


# =========================
# Adapters: build_report_params + fetch_report
# =========================
def call_build_report_params(
    shop_ids: List[int],
    data_output: List[str],
    source: str,
    period: str,
    period_step: str,
    date_from: str,
    date_to: str,
) -> Dict[str, Any]:
    if build_report_params is None:
        raise RuntimeError("build_report_params not imported")

    sig = inspect.signature(build_report_params)
    p = sig.parameters

    kwargs: Dict[str, Any] = {}

    # shops
    if "data" in p:
        kwargs["data"] = shop_ids
    elif "shop_ids" in p:
        kwargs["shop_ids"] = shop_ids
    elif "ids" in p:
        kwargs["ids"] = shop_ids
    elif "shops" in p:
        kwargs["shops"] = shop_ids

    # outputs
    if "data_output" in p:
        kwargs["data_output"] = data_output
    elif "outputs" in p:
        kwargs["outputs"] = data_output
    elif "kpis" in p:
        kwargs["kpis"] = data_output

    # other
    if "source" in p:
        kwargs["source"] = source
    if "period" in p:
        kwargs["period"] = period
    if "period_step" in p:
        kwargs["period_step"] = period_step

    if "date_from" in p:
        kwargs["date_from"] = date_from
    elif "from_date" in p:
        kwargs["from_date"] = date_from

    if "date_to" in p:
        kwargs["date_to"] = date_to
    elif "to_date" in p:
        kwargs["to_date"] = date_to

    try:
        return build_report_params(**kwargs)  # type: ignore
    except TypeError:
        # positional fallback
        args: List[Any] = []
        for name in p.keys():
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
        return build_report_params(*args)  # type: ignore


def call_fetch_report(api_url: str, params_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Your fetch_report differs: some accept (api_url, params),
    some accept (api_url, query), some accept only (params).
    We inspect signature and call correctly.
    """
    if fetch_report is None:
        raise RuntimeError("fetch_report not imported")

    sig = inspect.signature(fetch_report)
    p = list(sig.parameters.keys())

    # If it accepts **kwargs, pass as kwargs
    try:
        if any(sig.parameters[k].kind == inspect.Parameter.VAR_KEYWORD for k in sig.parameters):
            # Could be fetch_report(api_url=..., **params)
            # Prefer passing api_url if it exists in signature.
            if "api_url" in sig.parameters or "base_url" in sig.parameters or "url" in sig.parameters:
                return fetch_report(api_url=api_url, **params_dict)  # type: ignore
            return fetch_report(**params_dict)  # type: ignore
    except TypeError:
        pass

    # Common patterns:
    # 1) fetch_report(api_url, params_dict) positional
    # 2) fetch_report(api_url, **params?) already handled above
    # 3) fetch_report(params_dict) positional only

    if len(p) >= 2:
        # Try (api_url, params_dict)
        try:
            return fetch_report(api_url, params_dict)  # type: ignore
        except TypeError:
            pass
        # Try (api_url, query=params_dict) if second arg named query
        if "query" in sig.parameters:
            try:
                return fetch_report(api_url, query=params_dict)  # type: ignore
            except TypeError:
                pass

    if len(p) == 1:
        # Try (params_dict)
        return fetch_report(params_dict)  # type: ignore

    # Last resort
    return fetch_report(api_url, params_dict)  # type: ignore


# =========================
# Normalization fallback
# =========================
def normalize_payload(payload: Dict[str, Any]) -> pd.DataFrame:
    if callable(normalize_vemcount_response):
        rows = normalize_vemcount_response(payload)  # type: ignore
        return pd.DataFrame(rows)

    # basic fallback
    data = payload.get("data")
    if not isinstance(data, dict):
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for date_key, by_shop in data.items():
        if not isinstance(by_shop, dict):
            continue
        for shop_id, metrics in by_shop.items():
            if isinstance(metrics, dict):
                r = {"date": date_key, "shop_id": int(shop_id)}
                r.update(metrics)
                rows.append(r)
    return pd.DataFrame(rows)


# =========================
# Main
# =========================
def main():
    st.set_page_config(page_title="Region Reasoner", page_icon="🧠", layout="wide")
    st.title("🧠 Region Reasoner (agentic workload, selectors aligned)")

    if build_report_params is None or fetch_report is None:
        st.error("helpers_vemcount_api.build_report_params/fetch_report niet gevonden.")
        st.stop()

    api_url = st.secrets.get("API_URL", os.getenv("API_URL", "")).strip()
    if not api_url:
        st.error("API_URL ontbreekt.")
        st.stop()

    regions_df, region_to_shops = load_regions_mapping()

    c1, c2, c3 = st.columns([1.6, 1.2, 0.7], vertical_alignment="center")
    with c1:
        region = st.selectbox("Region", sorted(region_to_shops.keys()))
    with c2:
        preset = st.selectbox("Period", [p[0] for p in PERIOD_PRESETS], index=2)
    with c3:
        run = st.button("Run")

    if preset == "date":
        a, b = st.columns(2)
        ds, de = resolve_preset_to_dates("last_month")
        with a:
            d_from = st.date_input("Date from", value=ds)
        with b:
            d_to = st.date_input("Date to", value=de)
    else:
        d_from, d_to = resolve_preset_to_dates(preset)

    date_from_s, date_to_s = d_from.isoformat(), d_to.isoformat()
    shop_ids = region_to_shops.get(region, [])

    selection_key = f"{region}|{preset}|{date_from_s}|{date_to_s}|n={len(shop_ids)}"
    st.session_state.setdefault("rr_last_key", None)
    st.session_state.setdefault("rr_df", None)

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
            "fetch_report_signature": str(inspect.signature(fetch_report)),
        })

    if should_fetch:
        data_output = ["count_in", "turnover", "conversion_rate", "sales_per_visitor"]
        params_dict = call_build_report_params(
            shop_ids=shop_ids,
            data_output=data_output,
            source="shops",
            period="date",
            period_step="day",
            date_from=date_from_s,
            date_to=date_to_s,
        )

        with st.status("Fetching…", expanded=True) as status:
            status.write("Calling fetch_report…")
            payload = call_fetch_report(api_url, params_dict)

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

    # Minimal output
    st.subheader("Region rollup")
    footfall = float(df.get("count_in", pd.Series(dtype=float)).sum(skipna=True))
    turnover = float(df.get("turnover", pd.Series(dtype=float)).sum(skipna=True))
    conv = float(df.get("conversion_rate", pd.Series(dtype=float)).mean(skipna=True))
    spv = float(df.get("sales_per_visitor", pd.Series(dtype=float)).mean(skipna=True))

    a, b, c, d = st.columns(4)
    a.metric("Footfall", fmt_int(footfall))
    b.metric("Turnover", fmt_eur(turnover))
    c.metric("Conversion", fmt_pct(conv))
    d.metric("SPV", fmt_eur(spv))

    with st.expander("Raw df preview", expanded=False):
        st.dataframe(df.head(200), use_container_width=True)


if __name__ == "__main__":
    main()