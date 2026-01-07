# pages/06C_Region_Copilot_V2.py

import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt

from datetime import datetime, timedelta

from helpers_clients import load_clients
from helpers_normalize import normalize_vemcount_response
from helpers_periods import period_catalog
from helpers_vemcount_api import VemcountApiConfig, fetch_report, build_report_params

from services.cbs_service import (
    get_cci_series,
    get_retail_index,
)

# ----------------------
# Page config
# ----------------------
st.set_page_config(
    page_title="PFM Region Copilot v2 (Regio)",
    layout="wide"
)

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

# ✅ beter zichtbaar dan "bijna-wit"
OTHER_REGION_PURPLE = "#C084FC"   # licht-paars (duidelijk zichtbaar)
OTHER_REGION_OPACITY = 0.65

BLACK = "#111111"

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
# Minimal CSS
# ----------------------
st.markdown(
    f"""
    <style>
      .block-container {{
        padding-top: 2.25rem;
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
      .pill {{
        display:inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 800;
        border: 1px solid {PFM_LINE};
        background: {PFM_LIGHT};
        color: {PFM_DARK};
      }}
      .muted {{
        color: {PFM_GRAY};
        font-size: 0.86rem;
      }}
      .hint {{
        color: {PFM_GRAY};
        font-size: 0.82rem;
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

def fmt_eur_2(x: float) -> str:
    if pd.isna(x):
        return "-"
    s = f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"€ {s}"

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

    if "store_type" in df.columns:
        df["store_type"] = df["store_type"].astype(str)
    else:
        df["store_type"] = np.nan

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
# UI helpers
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
        .mark_arc(innerRadius=54, outerRadius=70)
        .encode(
            theta="value:Q",
            color=alt.Color(
                "segment:N",
                scale=alt.Scale(domain=["filled", "empty"], range=[fill_color, PFM_LINE]),
                legend=None,
            ),
        )
        .properties(width=180, height=180)
    )
    text = (
        alt.Chart(pd.DataFrame({"label": [f"{score_0_100:.0f}"]}))
        .mark_text(size=28, fontWeight="bold", color=PFM_DARK)
        .encode(text="label:N")
        .properties(width=180, height=180)
    )
    return arc + text

# ----------------------
# Data helpers
# ----------------------
def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def collapse_to_daily_store(df: pd.DataFrame, store_key_col: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])

    numeric_cols = [
        "footfall", "turnover", "transactions",
        "sales_per_visitor", "conversion_rate",
        "avg_basket_size", "sales_per_sqm", "sales_per_transaction",
    ]
    out = _coerce_numeric(out, [c for c in numeric_cols if c in out.columns])

    group_cols = [store_key_col, "date"]

    agg = {}
    for c in numeric_cols:
        if c not in out.columns:
            continue
        if c in ("footfall", "turnover", "transactions"):
            agg[c] = "sum"
        else:
            agg[c] = "mean"

    out = out.groupby(group_cols, as_index=False).agg(agg)

    if "turnover" in out.columns and "footfall" in out.columns:
        out["sales_per_visitor"] = np.where(out["footfall"] > 0, out["turnover"] / out["footfall"], np.nan)
    if "transactions" in out.columns and "footfall" in out.columns:
        out["conversion_rate"] = np.where(out["footfall"] > 0, out["transactions"] / out["footfall"] * 100.0, np.nan)
    if "turnover" in out.columns and "transactions" in out.columns:
        out["sales_per_transaction"] = np.where(out["transactions"] > 0, out["turnover"] / out["transactions"], np.nan)

    return out

def aggregate_weekly_region(df_region_daily: pd.DataFrame) -> pd.DataFrame:
    if df_region_daily is None or df_region_daily.empty:
        return pd.DataFrame()

    df = df_region_daily.copy()
    df["week_start"] = df["date"].dt.to_period("W-SAT").dt.start_time

    agg = {}
    if "footfall" in df.columns:
        agg["footfall"] = "sum"
    if "turnover" in df.columns:
        agg["turnover"] = "sum"
    if "transactions" in df.columns:
        agg["transactions"] = "sum"

    out = df.groupby("week_start", as_index=False).agg(agg)

    if "turnover" in out.columns and "footfall" in out.columns:
        out["sales_per_visitor"] = np.where(out["footfall"] > 0, out["turnover"] / out["footfall"], np.nan)
    if "transactions" in out.columns and "footfall" in out.columns:
        out["conversion_rate"] = np.where(out["footfall"] > 0, out["transactions"] / out["footfall"] * 100.0, np.nan)

    return out

def enrich_merged_with_sqm_from_df_norm(merged: pd.DataFrame, df_norm: pd.DataFrame, store_key_col: str) -> pd.DataFrame:
    if merged is None or merged.empty or df_norm is None or df_norm.empty:
        return merged

    sqm_col_norm = None
    for cand in ["sq_meter", "sqm", "sq_meters", "square_meters"]:
        if cand in df_norm.columns:
            sqm_col_norm = cand
            break
    if sqm_col_norm is None:
        return merged

    tmp = df_norm[[store_key_col, sqm_col_norm]].copy()
    tmp[store_key_col] = pd.to_numeric(tmp[store_key_col], errors="coerce")
    tmp[sqm_col_norm] = pd.to_numeric(tmp[sqm_col_norm], errors="coerce")
    tmp = tmp.dropna(subset=[store_key_col, sqm_col_norm])

    if tmp.empty:
        return merged

    sqm_by_shop = (
        tmp.groupby(store_key_col, as_index=False)[sqm_col_norm]
        .first()
        .rename(columns={sqm_col_norm: "sqm_api"})
    )

    out = merged.copy()
    out["id"] = pd.to_numeric(out["id"], errors="coerce")
    out = out.merge(sqm_by_shop, left_on="id", right_on=store_key_col, how="left")
    if store_key_col in out.columns:
        out = out.drop(columns=[store_key_col])
    return out

def mark_closed_days_as_nan(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in ["footfall", "turnover", "transactions"]:
        if c not in out.columns:
            return out

    base = (
        pd.to_numeric(out["footfall"], errors="coerce").fillna(0).eq(0)
        & pd.to_numeric(out["turnover"], errors="coerce").fillna(0).eq(0)
        & pd.to_numeric(out["transactions"], errors="coerce").fillna(0).eq(0)
    )
    cols_to_nan = ["footfall", "turnover", "transactions", "conversion_rate", "sales_per_visitor", "sales_per_transaction", "sales_per_sqm"]
    for c in cols_to_nan:
        if c in out.columns:
            out.loc[base, c] = np.nan
    return out

# ----------------------
# Scoring helpers
# ----------------------
def ratio_to_score_0_100(ratio_pct: float, floor: float, cap: float) -> float:
    if pd.isna(ratio_pct):
        return np.nan
    r = float(np.clip(ratio_pct, floor, cap))
    return (r - floor) / (cap - floor) * 100.0

def nice_component_name(key: str) -> str:
    mapping = {
        "sales_per_sqm": "Sales / m² (€)",
        "capture_rate": "Capture rate (%)",
        "conversion_rate": "Conversie (%)",
        "sales_per_visitor": "SPV (€ / visitor)",
        "sales_per_transaction": "ATV (€)",
    }
    return mapping.get(key, key)

def compute_svi_score(vals_a: dict, vals_b: dict, floor: float, cap: float) -> tuple[float, float]:
    """
    Returns (svi_score_0_100, avg_ratio_pct)
    SVI = avg ratio(conv, spv, atv, sales/m²) mapped to 0–100 via floor/cap clipping.
    """
    ratio_list = []
    for m in ["conversion_rate", "sales_per_visitor", "sales_per_transaction", "sales_per_sqm"]:
        va = vals_a.get(m, np.nan)
        vb = vals_b.get(m, np.nan)
        if pd.notna(va) and pd.notna(vb) and float(vb) != 0.0:
            ratio_list.append((float(va) / float(vb)) * 100.0)

    if not ratio_list:
        return np.nan, np.nan

    avg_ratio = float(np.nanmean(ratio_list))
    svi = ratio_to_score_0_100(avg_ratio, floor=float(floor), cap=float(cap))
    return float(svi), avg_ratio

def agg_period(df_: pd.DataFrame) -> dict:
    foot = float(pd.to_numeric(df_.get("footfall", 0), errors="coerce").dropna().sum())
    turn = float(pd.to_numeric(df_.get("turnover", 0), errors="coerce").dropna().sum())
    trans = float(pd.to_numeric(df_.get("transactions", 0), errors="coerce").dropna().sum())

    out = {"footfall": foot, "turnover": turn, "transactions": trans}
    out["conversion_rate"] = (trans / foot * 100.0) if foot > 0 else np.nan
    out["sales_per_visitor"] = (turn / foot) if foot > 0 else np.nan
    out["sales_per_transaction"] = (turn / trans) if trans > 0 else np.nan

    sqm = pd.to_numeric(df_.get("sqm_effective", np.nan), errors="coerce")
    sqm_sum = float(sqm.dropna().drop_duplicates().sum()) if sqm.notna().any() else np.nan
    out["sales_per_sqm"] = (turn / sqm_sum) if (pd.notna(sqm_sum) and sqm_sum > 0) else np.nan
    return out

# ----------------------
# MAIN
# ----------------------
def main():
    # ---------- Session state to prevent "Analyseer" wipe on drilldown ----------
    if "rcp_last_key" not in st.session_state:
        st.session_state.rcp_last_key = None
    if "rcp_payload" not in st.session_state:
        st.session_state.rcp_payload = None
    if "rcp_ran" not in st.session_state:
        st.session_state.rcp_ran = False

    st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)

    header_left, header_right = st.columns([2.2, 1.8])

    with header_left:
        st.markdown(
            f"""
            <div class="pfm-header">
              <div>
                <div class="pfm-title">PFM Region Performance Copilot <span class="pill">v2</span></div>
                <div class="pfm-sub">Regio-level: SVI + opportunities + store drilldown + macro context</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    clients = load_clients("clients.json")
    clients_df = pd.DataFrame(clients)
    clients_df["label"] = clients_df.apply(
        lambda r: f"{r['brand']} – {r['name']} (company_id {r['company_id']})",
        axis=1,
    )

    today = datetime.today().date()
    periods = period_catalog(today)
    period_labels = list(periods.keys())

    with header_right:
        c1, c2 = st.columns(2)
        with c1:
            client_label = st.selectbox("Retailer", clients_df["label"].tolist(), label_visibility="collapsed")
        with c2:
            period_choice = st.selectbox(
                "Periode",
                period_labels,
                index=period_labels.index("Q3 2024") if "Q3 2024" in period_labels else 0,
                label_visibility="collapsed",
            )

    selected_client = clients_df[clients_df["label"] == client_label].iloc[0].to_dict()
    company_id = int(selected_client["company_id"])

    start_period = periods[period_choice].start
    end_period = periods[period_choice].end

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

    if "store_label" in merged.columns and merged["store_label"].notna().any():
        merged["store_display"] = merged["store_label"]
    else:
        merged["store_display"] = merged["name"] if "name" in merged.columns else merged["id"].astype(str)

    available_regions = sorted(merged["region"].dropna().unique().tolist())

    top_controls = st.columns([1.2, 1.2, 1.2, 1.2, 1.2])
    with top_controls[0]:
        region_choice = st.selectbox("Regio", available_regions)
    with top_controls[1]:
        show_macro = st.toggle("Toon macro (CBS/CCI)", value=True)
    with top_controls[2]:
        show_quadrant = st.toggle("Toon quadrant", value=True)
    with top_controls[3]:
        lever_floor = st.selectbox("Lever scan gevoeligheid (vloer)", [70, 75, 80, 85], index=2)
    with top_controls[4]:
        run_btn = st.button("Analyseer", type="primary")

    lever_cap = 200 - lever_floor

    run_key = (company_id, region_choice, str(start_period), str(end_period), int(lever_floor), int(lever_cap))
    should_fetch = run_btn or (st.session_state.rcp_last_key != run_key) or (not st.session_state.rcp_ran)

    if not should_fetch and st.session_state.rcp_payload is None:
        st.info("Selecteer retailer/regio/periode en klik op **Analyseer**.")
        return

    if should_fetch:
        region_shops = merged[merged["region"] == region_choice].copy()
        region_shop_ids = region_shops["id"].dropna().astype(int).unique().tolist()
        if not region_shop_ids:
            st.warning(f"Geen winkels gevonden voor regio '{region_choice}'.")
            return

        all_shop_ids = merged["id"].dropna().astype(int).unique().tolist()

        metric_map = {
            "count_in": "footfall",
            "turnover": "turnover",
            "transactions": "transactions",
            "conversion_rate": "conversion_rate",
            "sales_per_visitor": "sales_per_visitor",
            "avg_basket_size": "avg_basket_size",
            "sales_per_sqm": "sales_per_sqm",
            "sales_per_transaction": "sales_per_transaction",
        }

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

        with st.spinner("Data ophalen via FastAPI..."):
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
                st.error(f"❌ HTTPError bij /get-report: {e}")
                try:
                    st.code(e.response.text)
                except Exception:
                    pass
                with st.expander("🔧 Debug request (params)"):
                    st.write("REPORT_URL:", REPORT_URL)
                    st.write("Params:", params_preview)
                return
            except requests.exceptions.RequestException as e:
                st.error(f"❌ RequestException bij /get-report: {e}")
                with st.expander("🔧 Debug request (params)"):
                    st.write("REPORT_URL:", REPORT_URL)
                    st.write("Params:", params_preview)
                return

        df_norm = normalize_vemcount_response(resp, kpi_keys=metric_map.keys()).rename(columns=metric_map)

        if df_norm is None or df_norm.empty:
            st.warning("Geen data ontvangen voor de gekozen selectie.")
            return

        store_key_col = None
        for cand in ["shop_id", "id", "location_id"]:
            if cand in df_norm.columns:
                store_key_col = cand
                break
        if store_key_col is None:
            st.error("Geen store-id kolom gevonden in de response (shop_id/id/location_id).")
            return

        merged2 = enrich_merged_with_sqm_from_df_norm(merged, df_norm, store_key_col=store_key_col)

        sqm_col = None
        for cand in ["sqm", "sq_meter", "sq_meters", "square_meters"]:
            if cand in merged2.columns:
                sqm_col = cand
                break

        base_sqm = pd.to_numeric(merged2[sqm_col], errors="coerce") if sqm_col is not None else np.nan
        sqm_api = pd.to_numeric(merged2["sqm_api"], errors="coerce") if "sqm_api" in merged2.columns else np.nan

        merged2["sqm_effective"] = np.where(
            merged2["sqm_override"].notna(),
            pd.to_numeric(merged2["sqm_override"], errors="coerce"),
            np.where(pd.notna(base_sqm), base_sqm, sqm_api)
        )

        df_daily_store = collapse_to_daily_store(df_norm, store_key_col=store_key_col)
        if df_daily_store is None or df_daily_store.empty:
            st.warning("Geen data na opschonen (daily/store collapse).")
            return

        join_cols = ["id", "store_display", "region", "sqm_effective"]
        if "store_type" in merged2.columns:
            join_cols.append("store_type")

        df_daily_store = df_daily_store.merge(
            merged2[join_cols].drop_duplicates(),
            left_on=store_key_col,
            right_on="id",
            how="left",
        )

        if "sales_per_sqm" not in df_daily_store.columns:
            df_daily_store["sales_per_sqm"] = np.nan
        sqm_eff = pd.to_numeric(df_daily_store.get("sqm_effective", np.nan), errors="coerce")
        turn = pd.to_numeric(df_daily_store.get("turnover", np.nan), errors="coerce")
        calc_spm2 = np.where((pd.notna(sqm_eff) & (sqm_eff > 0)), (turn / sqm_eff), np.nan)
        df_daily_store["sales_per_sqm"] = pd.to_numeric(df_daily_store["sales_per_sqm"], errors="coerce")
        df_daily_store["sales_per_sqm"] = df_daily_store["sales_per_sqm"].combine_first(pd.Series(calc_spm2, index=df_daily_store.index))

        df_daily_store = mark_closed_days_as_nan(df_daily_store)

        df_region_daily = df_daily_store[df_daily_store["region"] == region_choice].copy()

        st.session_state.rcp_last_key = run_key
        st.session_state.rcp_payload = {
            "df_daily_store": df_daily_store,
            "df_region_daily": df_region_daily,
            "merged": merged2,
            "store_key_col": store_key_col,
            "start_period": start_period,
            "end_period": end_period,
            "selected_client": selected_client,
            "region_choice": region_choice,
            "lever_floor": lever_floor,
            "lever_cap": lever_cap,
        }
        st.session_state.rcp_ran = True

    payload = st.session_state.rcp_payload
    df_daily_store = payload["df_daily_store"]
    df_region_daily = payload["df_region_daily"]
    merged = payload["merged"]
    start_period = payload["start_period"]
    end_period = payload["end_period"]
    selected_client = payload["selected_client"]
    region_choice = payload["region_choice"]
    lever_floor = payload["lever_floor"]
    lever_cap = payload["lever_cap"]

    if df_region_daily is None or df_region_daily.empty:
        st.warning("Geen data voor geselecteerde regio binnen de periode.")
        return

    # ----------------------
    # Region KPI headline
    # ----------------------
    foot_total = float(pd.to_numeric(df_region_daily["footfall"], errors="coerce").dropna().sum()) if "footfall" in df_region_daily.columns else 0.0
    turn_total = float(pd.to_numeric(df_region_daily["turnover"], errors="coerce").dropna().sum()) if "turnover" in df_region_daily.columns else 0.0
    trans_total = float(pd.to_numeric(df_region_daily["transactions"], errors="coerce").dropna().sum()) if "transactions" in df_region_daily.columns else 0.0

    conv = (trans_total / foot_total * 100.0) if foot_total > 0 else np.nan
    spv = (turn_total / foot_total) if foot_total > 0 else np.nan
    atv = (turn_total / trans_total) if trans_total > 0 else np.nan

    # ----------------------
    # Weekly capture inputs
    # ----------------------
    region_weekly = aggregate_weekly_region(df_region_daily)
    pathzz_weekly = fetch_region_street_traffic(region=region_choice, start_date=start_period, end_date=end_period)

    capture_weekly = pd.DataFrame()
    avg_capture = np.nan

    if not region_weekly.empty and not pathzz_weekly.empty:
        region_weekly = region_weekly.copy()
        region_weekly["week_start"] = pd.to_datetime(region_weekly["week_start"], errors="coerce")
        region_weekly = region_weekly.dropna(subset=["week_start"])
        region_weekly = region_weekly.groupby("week_start", as_index=False).agg(
            footfall=("footfall", "sum"),
            turnover=("turnover", "sum") if "turnover" in region_weekly.columns else ("footfall", "sum"),
            transactions=("transactions", "sum") if "transactions" in region_weekly.columns else ("footfall", "sum"),
        )

        pathzz_weekly = pathzz_weekly.copy()
        pathzz_weekly["week_start"] = pd.to_datetime(pathzz_weekly["week_start"], errors="coerce")
        pathzz_weekly = pathzz_weekly.dropna(subset=["week_start"])
        pathzz_weekly = pathzz_weekly.groupby("week_start", as_index=False).agg(street_footfall=("street_footfall", "mean"))

        capture_weekly = pd.merge(region_weekly, pathzz_weekly, on="week_start", how="inner")

    if not capture_weekly.empty:
        capture_weekly["capture_rate"] = np.where(
            capture_weekly["street_footfall"] > 0,
            capture_weekly["footfall"] / capture_weekly["street_footfall"] * 100.0,
            np.nan,
        )
        avg_capture = float(pd.to_numeric(capture_weekly["capture_rate"], errors="coerce").dropna().mean())

    # ----------------------
    # Header
    # ----------------------
    st.markdown(f"## {selected_client['brand']} — Regio **{region_choice}** · {start_period} → {end_period}")

    k1, k2, k3, k4, k5 = st.columns([1, 1, 1, 1, 1])
    with k1:
        kpi_card("Footfall", fmt_int(foot_total), "Regio · periode")
    with k2:
        kpi_card("Omzet", fmt_eur(turn_total), "Regio · periode")
    with k3:
        kpi_card("Conversion", fmt_pct(conv), "Transacties / bezoekers")
    with k4:
        kpi_card("ATV", fmt_eur_2(atv), "Omzet / transactie")
    with k5:
        kpi_card("Capture", fmt_pct(avg_capture), "Regio totaal (Pathzz)")

    st.markdown(
        "<div class='muted'>Company baseline = alle shops in company (zelfde periode). Regio baseline = shops binnen de gekozen regio.</div>",
        unsafe_allow_html=True,
    )

    # ✅ FIX: reg_vals/comp_vals hier berekenen (voor je SVI-gauge gebruikt wordt)
    reg_vals = agg_period(df_region_daily)
    comp_vals = agg_period(df_daily_store)

    # ----------------------
    # Regio Vitality Index (SVI) — prominent
    # ----------------------
    region_svi, region_avg_ratio = compute_svi_score(reg_vals, comp_vals, lever_floor, lever_cap)
    status_txt, status_color = status_from_score(region_svi if pd.notna(region_svi) else 0)

    c_svi_1, c_svi_2 = st.columns([1.1, 2.9])
    with c_svi_1:
        st.altair_chart(gauge_chart(region_svi if pd.notna(region_svi) else 0, status_color), use_container_width=False)
    with c_svi_2:
        st.markdown(
            f"""
            <div class="panel">
              <div class="panel-title">Regio Vitality Index (SVI)</div>
              <div style="font-size:2rem;font-weight:900;color:{PFM_DARK};line-height:1.1">
                {"" if pd.isna(region_svi) else f"{region_svi:.0f}"} <span class="pill">/ 100</span>
              </div>
              <div class="muted" style="margin-top:0.35rem">
                Status: <span style="font-weight:900;color:{status_color}">{status_txt}</span><br/>
                Benchmark: Company · ratio-gemiddelde (conv / SPV / ATV / sales m²) ≈ <b>{"" if pd.isna(region_avg_ratio) else f"{region_avg_ratio:.0f}%"} </b>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ----------------------
    # Weekly trend — Store vs Street + Capture (weeknumbers)
    # ----------------------
    st.markdown('<div class="panel"><div class="panel-title">Weekly trend — Store vs Street + Capture</div>', unsafe_allow_html=True)

    if capture_weekly.empty:
        st.info("Geen matchende Pathzz-weekdata gevonden voor deze regio/periode.")
    else:
        chart_df = capture_weekly[["week_start", "footfall", "street_footfall", "turnover", "capture_rate"]].copy()
        chart_df = chart_df.sort_values("week_start")

        iso = chart_df["week_start"].dt.isocalendar()
        chart_df["week_label"] = iso.week.apply(lambda w: f"W{int(w):02d}")
        week_order = chart_df["week_label"].tolist()

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

        st.altair_chart(
            alt.layer(bar, line).resolve_scale(y="independent").properties(height=260),
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------
    # Regio quadrant (v2) — moved ABOVE drilldown + better color visibility
    # ----------------------
    if show_quadrant:
        st.markdown("## Regio quadrant (v2)")

        regs = sorted(merged["region"].dropna().unique().tolist())
        rows = []

        for r in regs:
            drr = df_daily_store[df_daily_store["region"] == r].copy()
            if drr.empty:
                continue

            reg_vals_r = agg_period(drr)
            comp_vals_all = agg_period(df_daily_store)

            reg_spv = reg_vals_r.get("sales_per_visitor", np.nan)
            cmp_spv = comp_vals_all.get("sales_per_visitor", np.nan)
            y = (reg_spv / cmp_spv * 100.0) if (pd.notna(reg_spv) and pd.notna(cmp_spv) and cmp_spv != 0) else 100.0

            ratios = []
            for m in ["conversion_rate", "sales_per_visitor", "sales_per_transaction", "sales_per_sqm"]:
                v_reg = reg_vals_r.get(m, np.nan)
                v_cmp = comp_vals_all.get(m, np.nan)
                if pd.notna(v_reg) and pd.notna(v_cmp) and v_cmp != 0:
                    ratios.append((v_reg / v_cmp) * 100.0)

            x = np.nan
            if ratios:
                avg_ratio = float(np.nanmean(ratios))
                x = ratio_to_score_0_100(avg_ratio, floor=float(lever_floor), cap=float(lever_cap))

            rows.append({
                "region": r,
                "x_svi_proxy": x,
                "y_spv_index": y,
                "is_selected": (r == region_choice),
            })

        quad = pd.DataFrame(rows)
        quad["x_svi_proxy"] = pd.to_numeric(quad["x_svi_proxy"], errors="coerce")
        quad["y_spv_index"] = pd.to_numeric(quad["y_spv_index"], errors="coerce")

        if quad.empty or quad["x_svi_proxy"].dropna().empty:
            st.info("Nog onvoldoende data voor quadrant.")
        else:
            st.caption("X-as: SVI-proxy (0–100) op basis van regio vs company (conv/SPV/ATV/sales/m²). Y-as: SPV-index vs company (100 = gelijk).")

            chart = (
                alt.Chart(quad.dropna(subset=["x_svi_proxy", "y_spv_index"]))
                .mark_circle(size=240, opacity=0.95)
                .encode(
                    x=alt.X("x_svi_proxy:Q", title="SVI proxy (0–100)", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("y_spv_index:Q", title="SPV index vs company", axis=alt.Axis(format=".0f")),
                    color=alt.Color(
                        "is_selected:N",
                        scale=alt.Scale(domain=[True, False], range=[PFM_PURPLE, OTHER_REGION_PURPLE]),
                        legend=None,
                    ),
                    opacity=alt.condition("datum.is_selected", alt.value(1.0), alt.value(OTHER_REGION_OPACITY)),
                    tooltip=[
                        alt.Tooltip("region:N", title="Regio"),
                        alt.Tooltip("x_svi_proxy:Q", title="SVI proxy", format=".0f"),
                        alt.Tooltip("y_spv_index:Q", title="SPV index", format=".0f"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)

    # ----------------------
    # Store ranking (top → bottom) binnen regio — met SVI per store
    # ----------------------
    st.markdown("## Stores in regio — Top → Bottom")

    reg_store_daily = df_daily_store[df_daily_store["region"] == region_choice].copy()

    agg = reg_store_daily.groupby(["id", "store_display"], as_index=False).agg(
        turnover=("turnover", "sum"),
        footfall=("footfall", "sum"),
        transactions=("transactions", "sum"),
        sqm_effective=("sqm_effective", "first"),
    )

    agg["conversion_rate"] = np.where(agg["footfall"] > 0, agg["transactions"] / agg["footfall"] * 100.0, np.nan)
    agg["sales_per_visitor"] = np.where(agg["footfall"] > 0, agg["turnover"] / agg["footfall"], np.nan)
    agg["sales_per_transaction"] = np.where(agg["transactions"] > 0, agg["turnover"] / agg["transactions"], np.nan)
    agg["sales_per_sqm"] = np.where((agg["sqm_effective"] > 0) & pd.notna(agg["sqm_effective"]), agg["turnover"] / agg["sqm_effective"], np.nan)

    reg_baseline_vals = reg_vals

    def compute_store_svi_row(row):
        store_vals = {
            "conversion_rate": row["conversion_rate"],
            "sales_per_visitor": row["sales_per_visitor"],
            "sales_per_transaction": row["sales_per_transaction"],
            "sales_per_sqm": row["sales_per_sqm"],
        }
        svi, avg_ratio = compute_svi_score(store_vals, reg_baseline_vals, lever_floor, lever_cap)
        return pd.Series({"SVI": svi, "SVI_ratio": avg_ratio})

    agg[["SVI", "SVI_ratio"]] = agg.apply(compute_store_svi_row, axis=1)

    agg = agg.sort_values(["SVI", "turnover"], ascending=[False, False])

    show_cols = ["store_display", "SVI", "turnover", "conversion_rate", "sales_per_visitor", "sales_per_transaction", "sales_per_sqm"]
    display_df = agg[show_cols].copy()
    display_df["SVI"] = display_df["SVI"].map(lambda x: "-" if pd.isna(x) else f"{x:.0f}")
    display_df["turnover"] = display_df["turnover"].map(fmt_eur)
    display_df["conversion_rate"] = display_df["conversion_rate"].map(fmt_pct)
    display_df["sales_per_visitor"] = display_df["sales_per_visitor"].map(fmt_eur_2)
    display_df["sales_per_transaction"] = display_df["sales_per_transaction"].map(fmt_eur_2)
    display_df["sales_per_sqm"] = display_df["sales_per_sqm"].map(fmt_eur_2)

    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.caption("Tip: gebruik dit overzicht om meteen te kiezen welke store je wil drillen (SVI is vs regio-baseline).")

    # ----------------------
    # Store drilldown
    # ----------------------
    st.markdown("## Store drilldown")

    region_stores = merged[merged["region"] == region_choice].copy()
    region_stores = region_stores.dropna(subset=["id"]).copy()
    region_stores["id_int"] = region_stores["id"].astype(int)

    region_stores["dd_label"] = region_stores["store_display"].fillna(region_stores["id"].astype(str)) + " · " + region_stores["id"].astype(str)

    if "rcp_store_choice" not in st.session_state:
        st.session_state.rcp_store_choice = int(region_stores["id_int"].iloc[0])

    store_choice_label = st.selectbox(
        "Winkel",
        region_stores["dd_label"].tolist(),
        index=int(np.where(region_stores["id_int"].values == st.session_state.rcp_store_choice)[0][0]) if (st.session_state.rcp_store_choice in region_stores["id_int"].values) else 0,
    )

    chosen_id = int(store_choice_label.split("·")[-1].strip())
    st.session_state.rcp_store_choice = chosen_id

    df_store = df_daily_store[pd.to_numeric(df_daily_store["id"], errors="coerce").astype("Int64") == chosen_id].copy()
    store_name = region_stores.loc[region_stores["id_int"] == chosen_id, "store_display"].iloc[0] if (region_stores["id_int"] == chosen_id).any() else str(chosen_id)

    st.markdown(f"### **{store_name}** · storeID {chosen_id}")

    foot_s = float(pd.to_numeric(df_store["footfall"], errors="coerce").dropna().sum()) if "footfall" in df_store.columns else 0.0
    turn_s = float(pd.to_numeric(df_store["turnover"], errors="coerce").dropna().sum()) if "turnover" in df_store.columns else 0.0
    trans_s = float(pd.to_numeric(df_store["transactions"], errors="coerce").dropna().sum()) if "transactions" in df_store.columns else 0.0

    conv_s = (trans_s / foot_s * 100.0) if foot_s > 0 else np.nan
    atv_s = (turn_s / trans_s) if trans_s > 0 else np.nan

    sqm_eff_store = pd.to_numeric(region_stores.loc[region_stores["id_int"] == chosen_id, "sqm_effective"], errors="coerce")
    sqm_eff_store = float(sqm_eff_store.iloc[0]) if (sqm_eff_store is not None and not sqm_eff_store.empty and pd.notna(sqm_eff_store.iloc[0])) else np.nan
    spm2_s = (turn_s / sqm_eff_store) if (pd.notna(sqm_eff_store) and sqm_eff_store > 0) else np.nan

    # ✅ store SVI card
    store_vals_for_svi = {
        "conversion_rate": conv_s,
        "sales_per_visitor": (turn_s / foot_s) if foot_s > 0 else np.nan,
        "sales_per_transaction": atv_s,
        "sales_per_sqm": spm2_s,
    }
    store_svi, store_avg_ratio = compute_svi_score(store_vals_for_svi, reg_vals, lever_floor, lever_cap)
    store_status, store_status_color = status_from_score(store_svi if pd.notna(store_svi) else 0)

    sk1, sk2, sk3, sk4, sk5 = st.columns([1, 1, 1, 1, 1])
    with sk1:
        kpi_card("Footfall", fmt_int(foot_s), "Store · periode")
    with sk2:
        kpi_card("Omzet", fmt_eur(turn_s), "Store · periode")
    with sk3:
        kpi_card("Conversion", fmt_pct(conv_s), "Store · periode")
    with sk4:
        kpi_card("Sales / m²", fmt_eur(spm2_s), "Store · periode")
    with sk5:
        kpi_card("Store SVI (vs Regio)", "-" if pd.isna(store_svi) else f"{store_svi:.0f} / 100", store_status)

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    # ----------------------
    # Lever scan — boven Top/Bottom days (zoals gevraagd)
    # ----------------------
    st.markdown('<div class="panel"><div class="panel-title">Lever scan — waar zit het gat?</div>', unsafe_allow_html=True)

    compare_mode = st.radio("Vergelijking", ["vs Regio", "vs Company"], horizontal=True)

    bench = reg_vals if compare_mode == "vs Regio" else comp_vals
    store_vals = agg_period(df_store)

    lever_metrics = ["conversion_rate", "sales_per_visitor", "sales_per_transaction", "sales_per_sqm"]
    rows = []
    for m in lever_metrics:
        v_store = store_vals.get(m, np.nan)
        v_bench = bench.get(m, np.nan)

        ratio = np.nan
        if pd.notna(v_store) and pd.notna(v_bench) and float(v_bench) != 0.0:
            ratio = (float(v_store) / float(v_bench)) * 100.0

        score = ratio_to_score_0_100(ratio, floor=float(lever_floor), cap=float(lever_cap))
        gap = (100.0 - float(ratio)) if pd.notna(ratio) else np.nan

        rows.append({
            "metric": nice_component_name(m),
            "score": score,
            "ratio_pct": ratio,
            "gap_pct": gap,
            "store_value": v_store,
            "bench_value": v_bench,
        })

    lever_df = pd.DataFrame(rows)
    lever_df["score"] = pd.to_numeric(lever_df["score"], errors="coerce")
    lever_df = lever_df.sort_values("score", ascending=True)

    st.caption(
        f"Score is geclipt ({lever_floor}–{lever_cap}% → 0–100). "
        f"Zet vloer hoger (80/85) voor meer onderscheid. 100 betekent: (ver) boven benchmark."
    )

    lever_chart = (
        alt.Chart(lever_df.dropna(subset=["score"]))
        .mark_bar(cornerRadiusEnd=4, color=PFM_AMBER)
        .encode(
            x=alt.X("score:Q", title=f"Score {compare_mode} (0–100)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("metric:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("score:Q", title="Score", format=".0f"),
                alt.Tooltip("ratio_pct:Q", title="Ratio vs benchmark (%)", format=".1f"),
                alt.Tooltip("gap_pct:Q", title="Gap vs benchmark (%; + = onder)", format=".1f"),
                alt.Tooltip("store_value:Q", title="Store value", format=",.2f"),
                alt.Tooltip("bench_value:Q", title="Benchmark value", format=",.2f"),
            ],
        )
        .properties(height=240)
    )
    st.altair_chart(lever_chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------
    # Best / worst days (store)
    # ----------------------
    st.markdown('<div class="panel"><div class="panel-title">Best / worst days (store)</div>', unsafe_allow_html=True)

    rank_opts = {
        "Omzet": "turnover",
        "Conversie": "conversion_rate",
        "SPV": "sales_per_visitor",
        "ATV": "sales_per_transaction",
        "Sales / m²": "sales_per_sqm",
        "Footfall": "footfall",
    }
    rank_label = st.selectbox("Rank op", list(rank_opts.keys()), index=0)
    rank_col = rank_opts[rank_label]

    ds = df_store.copy()
    ds["date"] = pd.to_datetime(ds["date"], errors="coerce")
    ds = ds.dropna(subset=["date"])
    closed = (
        pd.to_numeric(ds.get("footfall", 0), errors="coerce").fillna(0).eq(0)
        & pd.to_numeric(ds.get("turnover", 0), errors="coerce").fillna(0).eq(0)
        & pd.to_numeric(ds.get("transactions", 0), errors="coerce").fillna(0).eq(0)
    )
    ds = ds[~closed].copy()

    left, right = st.columns(2)
    with left:
        st.subheader("Top 5 days")
        if ds.empty:
            st.info("Geen (open) dagen in deze periode.")
        else:
            st.dataframe(
                ds.sort_values(rank_col, ascending=False).head(5)[
                    ["date", "turnover", "footfall", "transactions", "conversion_rate", "sales_per_visitor", "sales_per_transaction", "sales_per_sqm"]
                ],
                use_container_width=True
            )
    with right:
        st.subheader("Bottom 5 days")
        if ds.empty:
            st.info("Geen (open) dagen in deze periode.")
        else:
            st.dataframe(
                ds.sort_values(rank_col, ascending=True).head(5)[
                    ["date", "turnover", "footfall", "transactions", "conversion_rate", "sales_per_visitor", "sales_per_transaction", "sales_per_sqm"]
                ],
                use_container_width=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------
    # (Macro context blijft hieronder ongewijzigd uit je huidige versie)
    # ----------------------

    # ----------------------
    # Debug
    # ----------------------
    with st.expander("🔧 Debug (v2)"):
        st.write("REPORT_URL:", REPORT_URL)
        st.write("Company:", company_id)
        st.write("Region:", region_choice)
        st.write("Period:", start_period, "→", end_period)
        st.write("Lever floor/cap:", lever_floor, lever_cap)
        st.write("reg_vals:", reg_vals)
        st.write("comp_vals:", comp_vals)
        st.write("df_daily_store cols:", df_daily_store.columns.tolist())
        st.write("Example store rows:", df_store.head(10))

if __name__ == "__main__":
    main()
