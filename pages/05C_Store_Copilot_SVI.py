# pages/07_Store_Manager_Copilot.py
# ------------------------------------------------------------
# PFM Store Manager Copilot (MVP)
# - Minimal choices: Client + Store + Week (2024 only)
# - Shows: Store SVI + driver bars + actions + weekly leaderboard
# - Adds: Month-to-date progress donut vs manual target (demo)
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from datetime import date, timedelta

from helpers_clients import load_clients
from helpers_normalize import normalize_vemcount_response
from helpers_vemcount_api import VemcountApiConfig, fetch_report, build_report_params

from stylesheet import inject_css


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

inject_css(
    PFM_PURPLE=PFM_PURPLE,
    PFM_RED=PFM_RED,
    PFM_DARK=PFM_DARK,
    PFM_GRAY=PFM_GRAY,
    PFM_LIGHT=PFM_LIGHT,
    PFM_LINE=PFM_LINE,
)

st.set_page_config(page_title="PFM Store Manager Copilot", layout="wide")


# ======================
# Helpers (small + safe)
# ======================
def safe_div(a, b):
    try:
        if a is None or b is None:
            return np.nan
        a = float(a)
        b = float(b)
        return (a / b) if b != 0 else np.nan
    except Exception:
        return np.nan


def fmt_int(x):
    try:
        if pd.isna(x):
            return "-"
        return f"{int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return "-"


def fmt_pct(x):
    try:
        if pd.isna(x):
            return "-"
        return f"{float(x):.1f}%".replace(".", ",")
    except Exception:
        return "-"


def fmt_eur(x):
    try:
        if pd.isna(x):
            return "-"
        v = float(x)
        s = f"{v:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"€ {s}"
    except Exception:
        return "-"


def status_from_score(score_0_100: float):
    try:
        s = float(score_0_100)
    except Exception:
        s = 0.0

    if s >= 75:
        return "Excellent", PFM_GREEN
    if s >= 60:
        return "Good / stable", PFM_PURPLE
    if s >= 45:
        return "Needs focus", PFM_AMBER
    return "Critical", PFM_RED


def kpi_card(label, value, help_txt=""):
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-help">{help_txt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compute_driver_values_from_period(footfall, turnover, transactions, sqm_sum=None, capture_pct=None):
    footfall = float(footfall) if pd.notna(footfall) else np.nan
    turnover = float(turnover) if pd.notna(turnover) else np.nan
    transactions = float(transactions) if pd.notna(transactions) else np.nan
    sqm_sum = float(sqm_sum) if pd.notna(sqm_sum) else np.nan

    cr = safe_div(transactions, footfall) * 100.0 if pd.notna(footfall) else np.nan
    spv = safe_div(turnover, footfall) if pd.notna(footfall) else np.nan
    atv = safe_div(turnover, transactions) if pd.notna(transactions) else np.nan
    spm2 = safe_div(turnover, sqm_sum) if (pd.notna(sqm_sum) and sqm_sum > 0) else np.nan
    cap = float(capture_pct) if pd.notna(capture_pct) else np.nan

    return {
        "conversion_rate": cr,
        "sales_per_visitor": spv,
        "sales_per_transaction": atv,
        "sales_per_sqm": spm2,
        "capture_rate": cap,
    }


def get_svi_weights_for_store_type(store_type: str):
    stype = (store_type or "").strip().lower()
    base = {
        "sales_per_visitor": 0.35,
        "conversion_rate": 0.25,
        "sales_per_transaction": 0.20,
        "sales_per_sqm": 0.15,
        "capture_rate": 0.05,
    }
    if "mall" in stype or "shopping" in stype:
        base["capture_rate"] = 0.10
        base["sales_per_sqm"] = 0.10
    if "city" in stype:
        base["conversion_rate"] = 0.30
        base["capture_rate"] = 0.03

    s = sum(base.values())
    return {k: v / s for k, v in base.items()}


def ratio_to_score(ratio_pct, floor=80.0, cap=120.0):
    if pd.isna(ratio_pct):
        return np.nan
    r = float(ratio_pct)
    r = max(float(floor), min(float(cap), r))
    return (r - float(floor)) / (float(cap) - float(floor)) * 100.0


def compute_svi_explainable(vals_a: dict, vals_b: dict, floor=80.0, cap=120.0, weights=None):
    weights = weights or get_svi_weights_for_store_type("")
    drivers = ["sales_per_visitor", "conversion_rate", "sales_per_transaction", "sales_per_sqm", "capture_rate"]

    rows = []
    scores = []
    wts = []
    ratios = []

    for d in drivers:
        a = vals_a.get(d, np.nan)
        b = vals_b.get(d, np.nan)

        ratio = safe_div(a, b) * 100.0 if (pd.notna(a) and pd.notna(b) and float(b) != 0.0) else np.nan
        score = ratio_to_score(ratio, floor=floor, cap=cap)
        w = float(weights.get(d, 0.0))

        rows.append({"driver": d, "a": a, "b": b, "ratio_pct": ratio, "score_0_100": score, "weight": w})
        if pd.notna(score) and w > 0:
            scores.append(score)
            wts.append(w)
        if pd.notna(ratio) and w > 0:
            ratios.append((ratio, w))

    if len(scores) == 0:
        return np.nan, np.nan, pd.DataFrame(rows)

    wsum = sum(wts)
    svi = float(sum(s * w for s, w in zip(scores, wts)) / wsum) if wsum > 0 else np.nan

    if len(ratios) > 0:
        rsum = sum(w for _, w in ratios)
        avg_ratio = float(sum(r * w for r, w in ratios) / rsum) if rsum > 0 else np.nan
    else:
        avg_ratio = np.nan

    return svi, avg_ratio, pd.DataFrame(rows)


def iso_week_range(year: int, week: int):
    d = date.fromisocalendar(year, week, 1)
    return d, d + timedelta(days=6)


def make_week_options_2024():
    opts = []
    for w in range(1, 53):
        s, e = iso_week_range(2024, w)
        label = f"2024-W{w:02d} ({s.strftime('%b %d')}–{e.strftime('%b %d')})"
        opts.append((label, s, e, w))
    return opts


def driver_label(d):
    return {
        "sales_per_visitor": "SPV",
        "conversion_rate": "Conversion",
        "sales_per_transaction": "ATV",
        "sales_per_sqm": "Sales / m²",
        "capture_rate": "Capture",
    }.get(d, d)


def make_actions(store_vals, bench_vals, weights):
    drivers = [
        ("conversion_rate", "Focus on converting visitors into customers (service, queue, closing routines)."),
        ("sales_per_transaction", "Increase ATV (bundles, add-ons, premium options, guided selling)."),
        ("sales_per_visitor", "Lift SPV by improving conversion *and* ATV (best lever for total revenue)."),
        ("sales_per_sqm", "Optimize space productivity (hot zones, product placement, promotions per m²)."),
        ("capture_rate", "Increase capture (window/entrance, local activation, signage, peak-hour staffing)."),
    ]

    rows = []
    for d, action in drivers:
        a = store_vals.get(d, np.nan)
        b = bench_vals.get(d, np.nan)
        ratio = safe_div(a, b) * 100.0 if (pd.notna(a) and pd.notna(b) and float(b) != 0.0) else np.nan
        rows.append((d, ratio, float(weights.get(d, 0.0)), action))

    df = pd.DataFrame(rows, columns=["driver", "ratio_pct", "weight", "action"])
    df = df.dropna(subset=["ratio_pct"])
    if df.empty:
        return []

    df["priority"] = (100 - df["ratio_pct"]).clip(lower=0) * df["weight"]
    df = df.sort_values("priority", ascending=False)
    return df.head(3).to_dict("records")


def donut_progress(progress_pct: float):
    p = 0.0 if pd.isna(progress_pct) else float(progress_pct)
    p = max(0.0, min(120.0, p))

    df = pd.DataFrame({"label": ["Progress", "Remaining"], "value": [p, max(0.0, 100.0 - p)]})

    c = (
        alt.Chart(df)
        .mark_arc(innerRadius=55, outerRadius=75)
        .encode(
            theta="value:Q",
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=["Progress", "Remaining"], range=[PFM_PURPLE, PFM_LINE]),
                legend=None,
            ),
            tooltip=[alt.Tooltip("label:N"), alt.Tooltip("value:Q", format=".0f")],
        )
        .properties(height=190)
    )

    txt = alt.Chart(pd.DataFrame({"t": [f"{p:.0f}%"]})).mark_text(
        fontSize=26, fontWeight=900, color=PFM_DARK
    ).encode(text="t:N")

    st.altair_chart((c + txt).configure_view(strokeWidth=0), use_container_width=True)


# ======================
# API Config
# ======================
API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "")).strip()
if not API_URL:
    st.error("Missing API_URL (set in Streamlit secrets or env).")
    st.stop()

cfg = VemcountApiConfig(api_url=API_URL)


# ======================
# Header (same structure)
# ======================
clients_df = load_clients()
if clients_df is None or clients_df.empty:
    st.error("No clients available (helpers_clients.load_clients).")
    st.stop()

client_labels = (clients_df["client_name"].astype(str) + " — " + "company_id " + clients_df["company_id"].astype(str)).tolist()

col_left, col_right = st.columns([3.8, 2.2], vertical_alignment="center")

with col_left:
    st.markdown(
        """
        <div class="pfm-header pfm-header--fixed">
          <div>
            <div class="pfm-title">PFM Store Manager Copilot <span class="pill">MVP</span></div>
            <div class="pfm-sub">Store-level: SVI + driver focus + actions + weekly leaderboard + month progress</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_right:
    st.markdown('<div class="pfm-header-right">', unsafe_allow_html=True)
    client_choice = st.selectbox("Client", client_labels, index=0, key="sm_client", label_visibility="collapsed")
    run = st.button("Run analysis", key="sm_run")
    st.markdown("</div>", unsafe_allow_html=True)

company_id = int(str(client_choice).split("company_id")[-1].strip())


# ======================
# Week selector (2024)
# ======================
weeks = make_week_options_2024()
if "sm_week_label" not in st.session_state:
    st.session_state.sm_week_label = weeks[26][0]  # mid-year default
if "sm_store_id" not in st.session_state:
    st.session_state.sm_store_id = None

sel_a, sel_b, sel_c = st.columns([2.2, 2.2, 1.6], vertical_alignment="top")

with sel_b:
    st.markdown('<div class="panel"><div class="panel-title">Week (2024)</div>', unsafe_allow_html=True)
    week_label = st.selectbox(
        "Week",
        [w[0] for w in weeks],
        index=[w[0] for w in weeks].index(st.session_state.sm_week_label) if st.session_state.sm_week_label in [w[0] for w in weeks] else 0,
        key="sm_week_select",
        label_visibility="collapsed",
    )
    st.session_state.sm_week_label = week_label
    week_meta = [w for w in weeks if w[0] == week_label][0]
    wk_start, wk_end, wk_num = week_meta[1], week_meta[2], week_meta[3]
    st.markdown("</div>", unsafe_allow_html=True)

with sel_c:
    st.markdown('<div class="panel"><div class="panel-title">Demo target</div>', unsafe_allow_html=True)
    month_target = st.number_input(
        "Monthly revenue target (€)",
        min_value=0,
        value=250000,
        step=5000,
        key="sm_month_target",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with sel_a:
    st.markdown('<div class="panel"><div class="panel-title">Store</div>', unsafe_allow_html=True)
    st.markdown("<div class='hint'>Click <b>Run analysis</b> first to load stores for the selected week.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ======================
# Data fetch (cached)
# ======================
@st.cache_data(show_spinner=False, ttl=1800)
def fetch_company_week(company_id: int, date_from: str, date_to: str):
    params = build_report_params(
        data_output=["count_in", "turnover", "transactions"],
        source="shops",
        period="date",
        period_step="day",
        date_from=date_from,
        date_to=date_to,
        company=company_id,
    )
    js = fetch_report(cfg, params=params)
    return normalize_vemcount_response(js)


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_store_mtd(store_id: int, date_from: str, date_to: str):
    params = build_report_params(
        data=[int(store_id)],
        data_output=["turnover"],
        source="shops",
        period="date",
        period_step="day",
        date_from=date_from,
        date_to=date_to,
    )
    js = fetch_report(cfg, params=params)
    return normalize_vemcount_response(js)


if "sm_should_fetch" not in st.session_state:
    st.session_state.sm_should_fetch = False

if run:
    st.session_state.sm_should_fetch = True

if st.session_state.sm_should_fetch:
    st.session_state.sm_should_fetch = False

    with st.spinner("Fetching week data..."):
        df_company_week = fetch_company_week(company_id, wk_start.isoformat(), wk_end.isoformat())

    if df_company_week is None or df_company_week.empty:
        st.error("No data returned for this week/company.")
        st.stop()

    df = df_company_week.copy()

    # Normalize expected columns
    rename_map = {
        "count_in": "footfall",
        "turnover": "turnover",
        "transactions": "transactions",
        "shop_id": "id",
        "store_id": "id",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    if "id" not in df.columns:
        id_candidates = [c for c in df.columns if c.lower() in ["id", "shop", "shop_id", "store_id"]]
        if id_candidates:
            df = df.rename(columns={id_candidates[0]: "id"})

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype(int)

    # Best-effort meta
    df["store_display"] = df.get("store_display", df["id"].astype(str))
    df["store_type"] = df.get("store_type", "Unknown")

    # Store list (from data)
    stores = df[["id", "store_display", "store_type"]].drop_duplicates().sort_values("id").copy()
    stores["label"] = stores["store_display"].astype(str) + " · " + stores["id"].astype(str)

    if st.session_state.sm_store_id is None or st.session_state.sm_store_id not in stores["id"].tolist():
        st.session_state.sm_store_id = int(stores["id"].iloc[0])

    # Re-render selection row with store dropdown filled
    sel_a, sel_b, sel_c = st.columns([2.2, 2.2, 1.6], vertical_alignment="top")
    with sel_a:
        st.markdown('<div class="panel"><div class="panel-title">Store</div>', unsafe_allow_html=True)
        chosen_label = st.selectbox(
            "Store",
            stores["label"].tolist(),
            index=int(np.where(stores["id"].values == st.session_state.sm_store_id)[0][0]) if st.session_state.sm_store_id in stores["id"].values else 0,
            key="sm_store_select",
            label_visibility="collapsed",
        )
        chosen_id = int(chosen_label.split("·")[-1].strip())
        st.session_state.sm_store_id = chosen_id
        st.markdown("</div>", unsafe_allow_html=True)

    with sel_b:
        st.markdown(f'<div class="panel"><div class="panel-title">Week</div><div class="hint">{week_label}</div></div>', unsafe_allow_html=True)

    with sel_c:
        st.markdown(f'<div class="panel"><div class="panel-title">Monthly target</div><div class="hint">{fmt_eur(month_target)}</div></div>', unsafe_allow_html=True)

    chosen_id = int(st.session_state.sm_store_id)
    store_row = stores[stores["id"] == chosen_id].iloc[0]
    store_name = str(store_row["store_display"])
    store_type = str(store_row["store_type"] or "Unknown")

    # Weekly aggregation
    agg = df.groupby(["id", "store_display", "store_type"], as_index=False).agg(
        footfall=("footfall", "sum"),
        turnover=("turnover", "sum"),
        transactions=("transactions", "sum"),
    )
    agg["conversion_rate"] = np.where(agg["footfall"] > 0, agg["transactions"] / agg["footfall"] * 100.0, np.nan)
    agg["sales_per_visitor"] = np.where(agg["footfall"] > 0, agg["turnover"] / agg["footfall"], np.nan)
    agg["sales_per_transaction"] = np.where(agg["transactions"] > 0, agg["turnover"] / agg["transactions"], np.nan)

    s_df = agg[agg["id"] == chosen_id].copy()
    foot_s = float(pd.to_numeric(s_df["footfall"], errors="coerce").fillna(0).sum())
    turn_s = float(pd.to_numeric(s_df["turnover"], errors="coerce").fillna(0).sum())
    trans_s = float(pd.to_numeric(s_df["transactions"], errors="coerce").fillna(0).sum())

    # Company same-type benchmark (week)
    same_type = store_type if store_type else "Unknown"
    com_type = agg[agg["store_type"] == same_type].copy()

    com_bench = {
        "sales_per_visitor": com_type["turnover"].sum() / com_type["footfall"].sum() if com_type["footfall"].sum() > 0 else np.nan,
        "conversion_rate": (com_type["transactions"].sum() / com_type["footfall"].sum() * 100.0) if com_type["footfall"].sum() > 0 else np.nan,
        "sales_per_transaction": (com_type["turnover"].sum() / com_type["transactions"].sum()) if com_type["transactions"].sum() > 0 else np.nan,
        "sales_per_sqm": np.nan,
        "capture_rate": np.nan,
    }

    # Region benchmark placeholder (until you wire region mapping)
    reg_bench = com_bench.copy()

    # Store values and SVI
    store_vals = compute_driver_values_from_period(foot_s, turn_s, trans_s, sqm_sum=np.nan, capture_pct=np.nan)
    weights = get_svi_weights_for_store_type(same_type)

    lever_floor = 80.0
    lever_cap = 120.0

    store_svi, store_avg_ratio, store_bd = compute_svi_explainable(
        vals_a=store_vals,
        vals_b=reg_bench,
        floor=lever_floor,
        cap=lever_cap,
        weights=weights,
    )

    status_txt, status_color = status_from_score(store_svi if pd.notna(store_svi) else 0)

    # ======================
    # Store header + KPIs
    # ======================
    st.markdown(
        f"## {store_name} · storeID {chosen_id} <span class='pill'>{same_type}</span>",
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        kpi_card("Footfall", fmt_int(foot_s), f"Week {wk_num} · 2024")
    with k2:
        kpi_card("Revenue", fmt_eur(turn_s), f"Week {wk_num} · 2024")
    with k3:
        conv_s = safe_div(trans_s, foot_s) * 100.0 if foot_s > 0 else np.nan
        kpi_card("Conversion", fmt_pct(conv_s), "Transactions / Visitors")
    with k4:
        spv_s = safe_div(turn_s, foot_s) if foot_s > 0 else np.nan
        kpi_card("SPV", fmt_eur(spv_s), "Revenue / Visitor")
    with k5:
        kpi_card("Store SVI", "-" if pd.isna(store_svi) else f"{store_svi:.0f} / 100", "vs same-type benchmark")

    st.markdown(
        f"<div class='muted'>Status: <span style='font-weight:900;color:{status_color}'>{status_txt}</span> · "
        f"Weighted ratio vs benchmark ≈ <b>{'' if pd.isna(store_avg_ratio) else f'{store_avg_ratio:.0f}%'}</b></div>",
        unsafe_allow_html=True,
    )

    # ======================
    # Row: Driver bars + Actions + Month progress
    # ======================
    col_a, col_b, col_c = st.columns([2.3, 2.1, 1.6], vertical_alignment="top")

    with col_a:
        st.markdown('<div class="panel"><div class="panel-title">SVI drivers — index vs benchmark</div>', unsafe_allow_html=True)
        bd = store_bd.copy()
        bd["driver_label"] = bd["driver"].apply(driver_label)
        bd["ratio_pct"] = pd.to_numeric(bd["ratio_pct"], errors="coerce")
        bd = bd.dropna(subset=["ratio_pct"]).copy()
        bd["ratio_clip"] = bd["ratio_pct"].clip(lower=60, upper=140)

        if bd.empty:
            st.info("No driver breakdown available (missing data).")
        else:
            chart = (
                alt.Chart(bd)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    y=alt.Y("driver_label:N", sort=["SPV", "Conversion", "ATV", "Sales / m²", "Capture"], title=None),
                    x=alt.X("ratio_clip:Q", title="Index (100 = benchmark)", scale=alt.Scale(domain=[60, 140])),
                    color=alt.condition(alt.datum.ratio_pct >= 100, alt.value(PFM_PURPLE), alt.value(PFM_RED)),
                    tooltip=[
                        alt.Tooltip("driver_label:N", title="Driver"),
                        alt.Tooltip("ratio_pct:Q", title="Index", format=".0f"),
                        alt.Tooltip("weight:Q", title="Weight", format=".2f"),
                    ],
                )
                .properties(height=210)
            )
            st.altair_chart(chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="panel"><div class="panel-title">Focus actions (this week)</div>', unsafe_allow_html=True)
        actions = make_actions(store_vals, reg_bench, weights)
        if not actions:
            st.info("Not enough data to generate actions.")
        else:
            for i, a in enumerate(actions, start=1):
                d = driver_label(a["driver"])
                idx = a["ratio_pct"]
                badge = f"{idx:.0f}%" if pd.notna(idx) else "-"
                color = PFM_RED if pd.notna(idx) and idx < 100 else PFM_PURPLE
                st.markdown(
                    f"""
                    <div class="callout" style="margin-bottom:0.6rem;">
                      <div class="callout-title">#{i} — {d} <span class="pill" style="margin-left:0.4rem;border-color:{color};">{badge}</span></div>
                      <div class="callout-sub">{a["action"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_c:
        st.markdown('<div class="panel"><div class="panel-title">Month-to-date progress (demo)</div>', unsafe_allow_html=True)

        month_start = date(wk_start.year, wk_start.month, 1)
        mtd_df = fetch_store_mtd(chosen_id, month_start.isoformat(), wk_end.isoformat())

        mtd = 0.0
        if mtd_df is not None and not mtd_df.empty:
            if "turnover" in mtd_df.columns:
                mtd = float(pd.to_numeric(mtd_df["turnover"], errors="coerce").fillna(0).sum())
            elif "data" in mtd_df.columns:
                mtd = float(pd.to_numeric(mtd_df["data"], errors="coerce").fillna(0).sum())

        progress_pct = (mtd / float(month_target) * 100.0) if float(month_target) > 0 else 0.0
        donut_progress(progress_pct)

        st.markdown(
            f"<div class='muted'>MTD revenue: <b>{fmt_eur(mtd)}</b><br/>Target: <b>{fmt_eur(month_target)}</b></div>",
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ======================
    # Weekly leaderboard
    # ======================
    st.markdown("## Weekly leaderboard (company-wide)")

    bench = com_bench.copy()
    w = weights

    def _svi_for_row(r):
        vals = compute_driver_values_from_period(
            footfall=r["footfall"],
            turnover=r["turnover"],
            transactions=r["transactions"],
            sqm_sum=np.nan,
            capture_pct=np.nan,
        )
        svi, avg_ratio, _ = compute_svi_explainable(vals, bench, floor=lever_floor, cap=lever_cap, weights=w)
        return svi, avg_ratio

    tmp = agg.copy()
    tmp[["svi", "avg_ratio"]] = tmp.apply(lambda r: pd.Series(_svi_for_row(r)), axis=1)

    tmp = tmp.sort_values("svi", ascending=False).copy()
    tmp["Rank"] = np.arange(1, len(tmp) + 1)

    show = tmp[["Rank", "store_display", "store_type", "svi", "turnover", "conversion_rate", "sales_per_visitor", "sales_per_transaction"]].copy()
    show = show.rename(
        columns={
            "store_display": "Store",
            "store_type": "Store type",
            "turnover": "Revenue",
            "conversion_rate": "CR",
            "sales_per_visitor": "SPV",
            "sales_per_transaction": "ATV",
        }
    )

    def _row_style(row):
        try:
            if str(row["Store"]).strip() == store_name.strip():
                return ["background-color:#F3F4F6; font-weight:900;"] * len(row)
        except Exception:
            pass
        return [""] * len(row)

    st.dataframe(
        show.style.apply(_row_style, axis=1).format(
            {
                "svi": lambda x: "-" if pd.isna(x) else f"{float(x):.0f}",
                "Revenue": lambda x: fmt_eur(x),
                "CR": lambda x: fmt_pct(x),
                "SPV": lambda x: fmt_eur(x),
                "ATV": lambda x: fmt_eur(x),
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info("Select client + week and click **Run analysis**.")
