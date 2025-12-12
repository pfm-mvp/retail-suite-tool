# pages/06B_Region_Copilot_OneScreen.py

import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt

from datetime import datetime, timedelta

from helpers_clients import load_clients
from helpers_normalize import normalize_vemcount_response

from services.cbs_service import (
    get_cci_series,
    get_retail_index,
)
from services.svi_service import build_store_vitality

# ----------------------
# Page config
# ----------------------
st.set_page_config(
    page_title="PFM Region Copilot (One Screen)",
    layout="wide"
)

# ----------------------
# PFM brand-ish colors (keep simple & consistent)
# ----------------------
PFM_PURPLE = "#762181"
PFM_RED = "#F04438"
PFM_DARK = "#111827"
PFM_GRAY = "#6B7280"
PFM_LIGHT = "#F3F4F6"
PFM_LINE = "#E5E7EB"
PFM_GREEN = "#22C55E"
PFM_AMBER = "#F59E0B"

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
# Minimal CSS for “dashboard cards”
# ----------------------
st.markdown(
    f"""
    <style>
      .block-container {{
        padding-top: 1.0rem;
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
      div.stButton > button {{
        background: {PFM_RED} !important;
        color: white !important;
        border: 0px !important;
        border-radius: 12px !important;
        padding: 0.65rem 1rem !important;
        font-weight: 800 !important;
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
def get_report(shop_ids, data_outputs, period: str, step: str = "day", source: str = "shops"):
    params: list[tuple[str, str]] = []
    for sid in shop_ids:
        params.append(("data", str(sid)))
    for dout in data_outputs:
        params.append(("data_output", dout))
    params.append(("period", period))
    params.append(("step", step))
    params.append(("source", source))

    resp = requests.post(REPORT_URL, params=params, timeout=90)
    resp.raise_for_status()
    return resp.json()

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
# KPI helpers
# ----------------------
def compute_daily_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "turnover" in df.columns and "footfall" in df.columns:
        df["sales_per_visitor"] = np.where(df["footfall"] > 0, df["turnover"] / df["footfall"], np.nan)
    if "transactions" in df.columns and "footfall" in df.columns:
        df["conversion_rate"] = np.where(df["footfall"] > 0, df["transactions"] / df["footfall"] * 100, np.nan)
    return df

def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["week_start"] = df["date"].dt.to_period("W-SAT").dt.start_time

    agg_dict: dict[str, str] = {}
    if "footfall" in df.columns:
        agg_dict["footfall"] = "sum"
    if "turnover" in df.columns:
        agg_dict["turnover"] = "sum"
    if "sales_per_visitor" in df.columns:
        agg_dict["sales_per_visitor"] = "mean"
    if "conversion_rate" in df.columns:
        agg_dict["conversion_rate"] = "mean"

    if not agg_dict:
        return df[["week_start"]].drop_duplicates().reset_index(drop=True)
    return df.groupby("week_start", as_index=False).agg(agg_dict)

# ----------------------
# Small UI helpers
# ----------------------
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
    # 0–100 scale
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
        .mark_arc(innerRadius=62, outerRadius=82)
        .encode(
            theta="value:Q",
            color=alt.Color(
                "segment:N",
                scale=alt.Scale(domain=["filled", "empty"], range=[fill_color, PFM_LINE]),
                legend=None,
            ),
        )
        .properties(width=240, height=240)
    )
    text = (
        alt.Chart(pd.DataFrame({"label": [f"{score_0_100:.0f}"]}))
        .mark_text(size=34, fontWeight="bold", color=PFM_DARK)
        .encode(text="label:N")
    )
    return arc + text

# ----------------------
# MAIN
# ----------------------
def main():
    # Header + controls in one row (dashboard feel)
    header_left, header_right = st.columns([2.2, 1.8])

    with header_left:
        st.markdown(
            f"""
            <div class="pfm-header">
              <div>
                <div class="pfm-title">PFM Region Performance Copilot</div>
                <div class="pfm-sub">One-screen layout – snel lezen, weinig scrollen (macro optioneel onderaan)</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Data needed for selectors
    clients = load_clients("clients.json")
    clients_df = pd.DataFrame(clients)
    clients_df["label"] = clients_df.apply(
        lambda r: f"{r['brand']} – {r['name']} (company_id {r['company_id']})",
        axis=1,
    )

    with header_right:
        c1, c2 = st.columns(2)
        with c1:
            client_label = st.selectbox("Retailer", clients_df["label"].tolist(), label_visibility="collapsed")
        with c2:
            period_choice = st.selectbox(
                "Periode",
                ["Kalenderjaar 2024", "Laatste 26 weken"],
                index=0,
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

    # sqm_effective + store_display
    if "sqm" in merged.columns:
        merged["sqm_effective"] = np.where(
            merged["sqm_override"].notna(),
            merged["sqm_override"],
            pd.to_numeric(merged["sqm"], errors="coerce"),
        )
    else:
        merged["sqm_effective"] = merged["sqm_override"]

    if "store_label" in merged.columns and merged["store_label"].notna().any():
        merged["store_display"] = merged["store_label"]
    else:
        merged["store_display"] = merged["name"] if "name" in merged.columns else merged["id"].astype(str)

    # Region selector + analyse button
    available_regions = sorted(merged["region"].dropna().unique().tolist())
    top_controls = st.columns([1.2, 1.2, 1.2, 1.4])
    with top_controls[0]:
        region_choice = st.selectbox("Regio", available_regions)
    with top_controls[1]:
        compare_all_regions = st.toggle("Vergelijk met andere regio’s", value=True)
    with top_controls[2]:
        show_macro = st.toggle("Toon macro (CBS/CCI)", value=False)
    with top_controls[3]:
        run_btn = st.button("Analyseer", type="primary")

    if not run_btn:
        st.info("Selecteer retailer/regio/periode en klik op **Analyseer**.")
        return

    # Period range (we filter locally)
    today = datetime.today().date()
    if period_choice == "Kalenderjaar 2024":
        start_period = datetime(2024, 1, 1).date()
        end_period = datetime(2024, 12, 31).date()
    else:
        end_period = today
        start_period = today - timedelta(weeks=26)

    start_ts = pd.Timestamp(start_period)
    end_ts = pd.Timestamp(end_period)

    # Shop IDs
    region_shops = merged[merged["region"] == region_choice].copy()
    region_shop_ids = region_shops["id"].dropna().astype(int).unique().tolist()
    if not region_shop_ids:
        st.warning(f"Geen winkels gevonden voor regio '{region_choice}'.")
        return

    # For region compare, we need data for ALL shops (so we can compute region_scores across all regions)
    all_shop_ids = merged["id"].dropna().astype(int).unique().tolist()
    fetch_ids = all_shop_ids if compare_all_regions else region_shop_ids

    # Fetch report
    metric_map = {"count_in": "footfall", "turnover": "turnover"}
    with st.spinner("Data ophalen via FastAPI..."):
        resp = get_report(
            fetch_ids,
            list(metric_map.keys()),
            period="this_year",   # we filter by date range below
            step="day",
            source="shops",
        )
        df_raw = normalize_vemcount_response(resp, kpi_keys=metric_map.keys()).rename(columns=metric_map)

    if df_raw.empty:
        st.warning("Geen data ontvangen voor de gekozen selectie.")
        return

    # Identify store id column
    df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
    df_raw = df_raw.dropna(subset=["date"])

    store_key_col = None
    for cand in ["id", "shop_id", "location_id"]:
        if cand in df_raw.columns:
            store_key_col = cand
            break

    if store_key_col is None:
        st.error("Geen store-id kolom gevonden in de response (id/shop_id/location_id).")
        return

    # Filter to selected period
    df_period_all = df_raw[(df_raw["date"] >= start_ts) & (df_raw["date"] <= end_ts)].copy()
    if df_period_all.empty:
        st.warning("Geen data in de geselecteerde periode.")
        return

    df_period_all = compute_daily_kpis(df_period_all)

    # Join store metadata (all)
    join_cols = ["id", "store_display", "region", "sqm_effective"]
    df_period_all = df_period_all.merge(
        merged[join_cols].drop_duplicates(),
        left_on=store_key_col,
        right_on="id",
        how="left",
    )

    # Region-only view
    df_region = df_period_all[df_period_all["region"] == region_choice].copy()
    if df_region.empty:
        st.warning("Geen data voor geselecteerde regio binnen de periode.")
        return

    # Weekly region aggregation + capture
    region_weekly = aggregate_weekly(df_region)
    pathzz_weekly = fetch_region_street_traffic(region=region_choice, start_date=start_period, end_date=end_period)

    capture_weekly = pd.DataFrame()
    avg_capture = np.nan
    if not region_weekly.empty and not pathzz_weekly.empty:
        region_weekly["week_start"] = pd.to_datetime(region_weekly["week_start"])
        pathzz_weekly["week_start"] = pd.to_datetime(pathzz_weekly["week_start"])
        capture_weekly = pd.merge(region_weekly, pathzz_weekly, on="week_start", how="inner")
        if not capture_weekly.empty:
            capture_weekly["capture_rate"] = np.where(
                capture_weekly["street_footfall"] > 0,
                capture_weekly["footfall"] / capture_weekly["street_footfall"] * 100.0,
                np.nan,
            )
            avg_capture = float(capture_weekly["capture_rate"].mean())

    # KPI cards
    foot_total = float(df_region["footfall"].sum()) if "footfall" in df_region.columns else 0.0
    turn_total = float(df_region["turnover"].sum()) if "turnover" in df_region.columns else 0.0
    spv_avg = float(df_region["sales_per_visitor"].mean()) if "sales_per_visitor" in df_region.columns else np.nan

    # SVI calculations
    svi_all = build_store_vitality(
        df_period=df_period_all,
        region_shops=merged,
        store_key_col=store_key_col,
    )

    region_scores = pd.DataFrame()
    region_svi = np.nan
    region_status, region_color = ("-", PFM_LINE)

    if not svi_all.empty:
        # Ensure region column exists
        if "region" not in svi_all.columns:
            svi_all = svi_all.merge(
                merged[["id", "region"]].drop_duplicates(),
                left_on=store_key_col,
                right_on="id",
                how="left",
            )

        region_scores = (
            svi_all.groupby("region", as_index=False)["svi_score"]
            .mean()
            .rename(columns={"svi_score": "region_svi"})
            .dropna(subset=["region"])
        )

        cur = region_scores[region_scores["region"] == region_choice]
        if not cur.empty:
            region_svi = float(np.clip(cur["region_svi"].iloc[0], 0, 100))
            region_status, region_color = status_from_score(region_svi)

    # One-screen layout blocks
    st.markdown(f"### {selected_client['brand']} — Regio **{region_choice}**  ·  {start_period} → {end_period}")

    k1, k2, k3, k4, k5 = st.columns([1, 1, 1, 1, 1.2])
    with k1:
        kpi_card("Footfall", fmt_int(foot_total), "Regio · periode")
    with k2:
        kpi_card("Omzet", fmt_eur(turn_total), "Regio · periode")
    with k3:
        kpi_card("SPV", (f"€ {spv_avg:.2f}".replace(".", ",") if not pd.isna(spv_avg) else "-"), "Gemiddelde")
    with k4:
        kpi_card("Capture", (fmt_pct(avg_capture) if not pd.isna(avg_capture) else "-"), "Gemiddeld (Pathzz)")
    with k5:
        # Region vitality mini-panel
        st.markdown('<div class="panel"><div class="panel-title">Regio Vitality</div>', unsafe_allow_html=True)
        if not pd.isna(region_svi):
            st.altair_chart(gauge_chart(region_svi, region_color), use_container_width=True)
            st.markdown(f"**{region_svi:.0f}** · {region_status}")
        else:
            st.info("Nog geen regio-score.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 2: weekly trend + region compare
    r2_left, r2_right = st.columns([1.6, 1.0])

    with r2_left:
        st.markdown('<div class="panel"><div class="panel-title">Weekly trend — Store vs Street + Capture</div>', unsafe_allow_html=True)

        if capture_weekly.empty:
            st.info("Geen matchende Pathzz-weekdata gevonden voor deze regio/periode.")
        else:
            chart_df = capture_weekly[["week_start", "footfall", "street_footfall", "turnover", "capture_rate"]].copy()
            iso = chart_df["week_start"].dt.isocalendar()
            chart_df["week_label"] = iso.week.apply(lambda w: f"W{int(w):02d}")
            week_order = chart_df.sort_values("week_start")["week_label"].unique().tolist()

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

            st.altair_chart(alt.layer(bar, line).resolve_scale(y="independent").properties(height=280), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with r2_right:
        st.markdown('<div class="panel"><div class="panel-title">Regio vergelijking — RVI (SVI gemiddeld)</div>', unsafe_allow_html=True)

        if not compare_all_regions:
            st.info("Zet ‘Vergelijk met andere regio’s’ aan om alle regio’s te tonen.")
        elif region_scores.empty or region_scores["region"].nunique() <= 1:
            st.info("Nog onvoldoende regio’s of data om te vergelijken.")
        else:
            chart_regions = region_scores.copy()
            chart_regions["is_selected"] = chart_regions["region"] == region_choice

            region_chart = (
                alt.Chart(chart_regions.sort_values("region_svi", ascending=False))
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("region_svi:Q", title="RVI (0–100)", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("region:N", sort="-x", title=None),
                    color=alt.Color(
                        "is_selected:N",
                        scale=alt.Scale(domain=[True, False], range=[PFM_PURPLE, PFM_LINE]),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("region:N", title="Regio"),
                        alt.Tooltip("region_svi:Q", title="RVI", format=".0f"),
                    ],
                )
                .properties(height=260)
            )

            st.altair_chart(region_chart, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Row 3: store ranking + opportunities
    r3_left, r3_right = st.columns([1.6, 1.0])

    with r3_left:
        st.markdown('<div class="panel"><div class="panel-title">Store Vitality ranking — geselecteerde regio</div>', unsafe_allow_html=True)

        if svi_all.empty:
            st.info("Geen SVI beschikbaar.")
        else:
            # --- Ensure 'region' exists on svi_region (defensive) ---
if "region" not in svi_region.columns:
    region_lookup = merged[["id", "region"]].drop_duplicates()

    # store_key_col is the column in svi_region that identifies the store (id/shop_id/location_id)
    # In most setups build_store_vitality keeps store_key_col as the store identifier column
    if store_key_col in svi_region.columns:
        svi_region = svi_region.merge(
            region_lookup,
            left_on=store_key_col,
            right_on="id",
            how="left",
        )
    elif "id" in svi_region.columns:
        svi_region = svi_region.merge(
            region_lookup,
            on="id",
            how="left",
        )

# Handle possible suffixes if region ended up as region_x/region_y
if "region" not in svi_region.columns:
    for cand in ["region_x", "region_y"]:
        if cand in svi_region.columns:
            svi_region["region"] = svi_region[cand]
            break
            svi_region = svi_all.merge(
                merged[["id", "region"]].drop_duplicates(),
                left_on=store_key_col,
                right_on="id",
                how="left",
            )
            svi_region = svi_region[svi_region["region"] == region_choice].copy()

            if svi_region.empty:
                st.info("Geen stores in deze regio met SVI.")
            else:
                svi_region["svi_score"] = pd.to_numeric(svi_region["svi_score"], errors="coerce")
                svi_region = svi_region.dropna(subset=["svi_score"])

                chart_rank = (
                    alt.Chart(svi_region.sort_values("svi_score", ascending=False).head(12))
                    .mark_bar(cornerRadiusEnd=4)
                    .encode(
                        x=alt.X("svi_score:Q", title="SVI (0–100)", scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y("store_name:N", sort="-x", title=None),
                        color=alt.Color(
                            "svi_status:N",
                            scale=alt.Scale(
                                domain=["High performance", "Good / stable", "Attention required", "Under pressure"],
                                range=[PFM_GREEN, PFM_PURPLE, PFM_AMBER, PFM_RED],
                            ),
                            legend=alt.Legend(title=""),
                        ),
                        tooltip=[
                            alt.Tooltip("store_name:N", title="Winkel"),
                            alt.Tooltip("svi_score:Q", title="SVI", format=".0f"),
                            alt.Tooltip("svi_status:N", title="Status"),
                            alt.Tooltip("reason_short:N", title="Waarom"),
                        ],
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart_rank, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with r3_right:
        st.markdown('<div class="panel"><div class="panel-title">Biggest opportunities</div>', unsafe_allow_html=True)

        if svi_all.empty:
            st.info("Geen opportunity data.")
        else:
            svi_region = svi_all.merge(
                merged[["id", "region"]].drop_duplicates(),
                left_on=store_key_col,
                right_on="id",
                how="left",
            )
            svi_region = svi_region[svi_region["region"] == region_choice].copy()

            if svi_region.empty or "profit_potential_period" not in svi_region.columns:
                st.info("Geen profit potential gevonden.")
            else:
                period_days = (end_ts - start_ts).days + 1
                year_factor = 365.0 / period_days if period_days > 0 else 1.0
                svi_region["profit_potential_year"] = svi_region["profit_potential_period"] * year_factor

                opp = (
                    svi_region[["store_name", "profit_potential_year"]]
                    .dropna()
                    .sort_values("profit_potential_year", ascending=False)
                    .head(6)
                )

                # show as simple chart (fast, readable)
                opp_chart = (
                    alt.Chart(opp)
                    .mark_bar(cornerRadiusEnd=4, color=PFM_RED)
                    .encode(
                        x=alt.X("profit_potential_year:Q", title="€ / jaar", axis=alt.Axis(format=",.0f")),
                        y=alt.Y("store_name:N", sort="-x", title=None),
                        tooltip=[
                            alt.Tooltip("store_name:N", title="Winkel"),
                            alt.Tooltip("profit_potential_year:Q", title="€ / jaar", format=",.0f"),
                        ],
                    )
                    .properties(height=300)
                )
                st.altair_chart(opp_chart, use_container_width=True)

                # quick “in your face” line
                total_top5 = float(opp["profit_potential_year"].head(5).sum())
                st.markdown(f"**Top 5 samen:** {fmt_eur(total_top5)} / jaar")

        st.markdown("</div>", unsafe_allow_html=True)

    # Macro section (toggle)
    if show_macro:
        st.markdown("### Macro-context (optioneel)")
        st.caption("Onderstaand mag scrollen — maar staat bewust *onder* je 1-screen dashboard.")

        macro_col1, macro_col2 = st.columns(2)

        # Build regional monthly indices
        region_month = df_region.copy()
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
            base_idx = nonzero.index[0]
            base_val = nonzero.iloc[0]
            idx = s / base_val * 100.0
            idx.loc[s.index < base_idx] = np.nan
            return idx

        if not region_month.empty:
            region_month["region_turnover_index"] = index_from_first_nonzero(region_month["region_turnover"])
            region_month["region_footfall_index"] = index_from_first_nonzero(region_month["region_footfall"])

        # CBS retail index
        cbs_retail_month = pd.DataFrame()
        cci_df = pd.DataFrame()

        with macro_col1:
            st.markdown('<div class="panel"><div class="panel-title">CBS detailhandelindex vs Regio</div>', unsafe_allow_html=True)
            try:
                retail_series = get_retail_index(months_back=24)
            except Exception:
                retail_series = []

            if retail_series:
                cbs_retail_df = pd.DataFrame(retail_series)
                cbs_retail_df["date"] = pd.to_datetime(
                    cbs_retail_df["period"].str[:4] + "-" + cbs_retail_df["period"].str[-2:] + "-15",
                    errors="coerce",
                )
                cbs_retail_df = cbs_retail_df.dropna(subset=["date"])
                cbs_retail_month = cbs_retail_df.groupby("date", as_index=False)["retail_value"].mean()
                if not cbs_retail_month.empty and cbs_retail_month["retail_value"].notna().any():
                    base = cbs_retail_month["retail_value"].dropna().iloc[0]
                    cbs_retail_month["cbs_retail_index"] = np.where(base != 0, cbs_retail_month["retail_value"] / base * 100.0, np.nan)

            lines = []
            if not region_month.empty:
                a = region_month.rename(columns={"month": "date"})[["date", "region_footfall_index"]].copy()
                a["series"] = "Regio footfall-index"
                a = a.rename(columns={"region_footfall_index": "value"})
                lines.append(a)

                b = region_month.rename(columns={"month": "date"})[["date", "region_turnover_index"]].copy()
                b["series"] = "Regio omzet-index"
                b = b.rename(columns={"region_turnover_index": "value"})
                lines.append(b)

            if not cbs_retail_month.empty and "cbs_retail_index" in cbs_retail_month.columns:
                c = cbs_retail_month[["date", "cbs_retail_index"]].copy()
                c["series"] = "CBS detailhandelindex"
                c = c.rename(columns={"cbs_retail_index": "value"})
                lines.append(c)

            if lines:
                macro = (
                    alt.Chart(pd.concat(lines, ignore_index=True))
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("date:T", title="Maand"),
                        y=alt.Y("value:Q", title="Index (100 = eerste maand)"),
                        color=alt.Color("series:N", title=""),
                        tooltip=[
                            alt.Tooltip("date:T", title="Maand"),
                            alt.Tooltip("series:N", title="Reeks"),
                            alt.Tooltip("value:Q", title="Index", format=".1f"),
                        ],
                    )
                    .properties(height=260)
                )
                st.altair_chart(macro, use_container_width=True)
            else:
                st.info("Geen macro-lijnen beschikbaar.")

            st.markdown("</div>", unsafe_allow_html=True)

        with macro_col2:
            st.markdown('<div class="panel"><div class="panel-title">Consumentenvertrouwen (CCI) vs Regio</div>', unsafe_allow_html=True)
            try:
                cci_series = get_cci_series(months_back=24)
            except Exception:
                cci_series = []

            if cci_series:
                cci_df = pd.DataFrame(cci_series)
                cci_df["date"] = pd.to_datetime(
                    cci_df["period"].str[:4] + "-" + cci_df["period"].str[-2:] + "-15",
                    errors="coerce",
                )
                cci_df = cci_df.dropna(subset=["date"])
                if not cci_df.empty and cci_df["cci"].notna().any():
                    base = cci_df["cci"].dropna().iloc[0]
                    cci_df["cci_index"] = np.where(base != 0, cci_df["cci"] / base * 100.0, np.nan)

            lines = []
            if not cci_df.empty and "cci_index" in cci_df.columns:
                c = cci_df[["date", "cci_index"]].copy()
                c["series"] = "Consumentenvertrouwen-index"
                c = c.rename(columns={"cci_index": "value"})
                lines.append(c)

            if not region_month.empty:
                a = region_month.rename(columns={"month": "date"})[["date", "region_footfall_index"]].copy()
                a["series"] = "Regio footfall-index"
                a = a.rename(columns={"region_footfall_index": "value"})
                lines.append(a)

                b = region_month.rename(columns={"month": "date"})[["date", "region_turnover_index"]].copy()
                b["series"] = "Regio omzet-index"
                b = b.rename(columns={"region_turnover_index": "value"})
                lines.append(b)

            if lines:
                cc = (
                    alt.Chart(pd.concat(lines, ignore_index=True))
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("date:T", title="Maand"),
                        y=alt.Y("value:Q", title="Index (100 = eerste maand)"),
                        color=alt.Color("series:N", title=""),
                        tooltip=[
                            alt.Tooltip("date:T", title="Maand"),
                            alt.Tooltip("series:N", title="Reeks"),
                            alt.Tooltip("value:Q", title="Index", format=".1f"),
                        ],
                    )
                    .properties(height=260)
                )
                st.altair_chart(cc, use_container_width=True)
            else:
                st.info("Geen CCI-data beschikbaar.")

            st.markdown("</div>", unsafe_allow_html=True)

    # Optional debug expander (kept lightweight)
    with st.expander("🔧 Debug"):
        st.write("Retailer:", selected_client)
        st.write("Regio:", region_choice)
        st.write("Periode:", start_period, "→", end_period)
        st.write("Compare all regions:", compare_all_regions)
        st.write("Store key col:", store_key_col)
        st.write("All shops:", len(all_shop_ids), "Region shops:", len(region_shop_ids))
        st.write("df_period_all head:", df_period_all.head())
        st.write("df_region head:", df_region.head())
        st.write("region_scores:", region_scores)


if __name__ == "__main__":
    main()