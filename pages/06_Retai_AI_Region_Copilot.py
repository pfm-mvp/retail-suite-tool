# pages/06_Retai_AI_Region_Copilot.py

import numpy as np
import pandas as pd
import requests
import streamlit as st

from datetime import datetime, timedelta

from helpers_clients import load_clients
from helpers_normalize import normalize_vemcount_response

st.set_page_config(
    page_title="PFM Region Performance Copilot",
    layout="wide"
)

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


# -------------
# Format helpers
# -------------

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


# -------------
# Region & API helpers
# -------------

@st.cache_data(ttl=600)
def load_region_mapping(path: str = "data/regions.csv") -> pd.DataFrame:
    """
    Verwacht een CSV met minimaal:
    shop_id;region

    shop_id → int
    region  → str
    """
    try:
        df = pd.read_csv(path, sep=";")
    except Exception:
        return pd.DataFrame()

    if "shop_id" not in df.columns or "region" not in df.columns:
        return pd.DataFrame()

    df["shop_id"] = pd.to_numeric(df["shop_id"], errors="coerce").astype("Int64")
    df["region"] = df["region"].astype(str)
    df = df.dropna(subset=["shop_id"])
    return df


@st.cache_data(ttl=600)
def get_locations_by_company(company_id: int) -> pd.DataFrame:
    """
    Wrapper rond /company/{company_id}/location van de vemcount-agent.
    Extra ruime timeout, fouten worden in main() afgehandeld.
    """
    url = f"{FASTAPI_BASE_URL.rstrip('/')}/company/{company_id}/location"

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict) and "locations" in data:
        df = pd.DataFrame(data["locations"])
    else:
        df = pd.DataFrame(data)

    return df


@st.cache_data(ttl=600)
def get_report(
    shop_ids,
    data_outputs,
    period: str,
    step: str = "day",
    source: str = "shops",
):
    """
    Wrapper rond /get-report (POST) van de vemcount-agent, met querystring zonder [].
    """
    params: list[tuple[str, str]] = []

    for sid in shop_ids:
        params.append(("data", str(sid)))

    for dout in data_outputs:
        params.append(("data_output", dout))

    params.append(("period", period))
    params.append(("step", step))
    params.append(("source", source))

    resp = requests.post(REPORT_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


# --- Pathzz loader per regio: simpele, bewezen werkwijze -------------------

@st.cache_data(ttl=600)
def load_pathzz_weekly_for_region(region_name: str, start_date, end_date) -> pd.DataFrame:
    """
    Leest demo-Pathzz data uit data/pathzz_sample_weekly.csv en filtert op regio + periode.

    CSV-structuur (jouw bestand):
    Region;Week;Visits;...

    - Region  : 'Noord', 'Oost', 'Zuid', 'West'
    - Week    : '2023-12-31 To 2024-01-06'
    - Visits  : '54.085' (duizendtallen met punt → 54085)
    """
    csv_path = "data/pathzz_sample_weekly.csv"
    try:
        # Zelfde aanpak als in je winkel-script: alleen Visits als string afdwingen
        df = pd.read_csv(csv_path, sep=";", dtype={"Visits": "string"})
    except Exception:
        return pd.DataFrame()

    # Kolommen netjes hernoemen
    df = df.rename(
        columns={
            "Region": "region",
            "Week": "week",
            "Visits": "street_footfall",
        }
    )

    if not {"region", "week", "street_footfall"}.issubset(df.columns):
        return pd.DataFrame()

    # Regio opschonen + exact matchen
    df["region"] = (
        df["region"]
        .astype(str)
        .str.replace("\ufeff", "")
        .str.strip()
    )
    region_clean = str(region_name).replace("\ufeff", "").strip()

    df = df[df["region"] == region_clean].copy()
    if df.empty:
        return pd.DataFrame()

    # Visits: "54.085" → "54085" → float
    df["street_footfall"] = (
        df["street_footfall"]
        .astype(str)
        .str.replace("\ufeff", "")
        .str.strip()
        .str.replace(".", "", regex=False)   # punt = duizendscheiding
        .str.replace(",", ".", regex=False)  # safety
    )
    df = df[df["street_footfall"] != ""]
    df["street_footfall"] = pd.to_numeric(df["street_footfall"], errors="coerce")
    df = df.dropna(subset=["street_footfall"])

    # "2023-12-31 To 2024-01-06" → 2023-12-31
    def _parse_week_start(s: str):
        if isinstance(s, str) and "To" in s:
            return pd.to_datetime(s.split("To")[0].strip(), errors="coerce")
        return pd.NaT

    df["week_start"] = df["week"].apply(_parse_week_start)
    df = df.dropna(subset=["week_start"])

    # Filter op periode
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    df = df[(df["week_start"] >= start) & (df["week_start"] <= end)]

    return df[["week_start", "street_footfall"]].reset_index(drop=True)


# -------------
# KPI helpers
# -------------

def compute_daily_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "turnover" in df.columns and "footfall" in df.columns:
        df["sales_per_visitor"] = np.where(
            df["footfall"] > 0,
            df["turnover"] / df["footfall"],
            np.nan,
        )
    if "transactions" in df.columns and "footfall" in df.columns:
        df["conversion_rate"] = np.where(
            df["footfall"] > 0,
            df["transactions"] / df["footfall"] * 100,
            np.nan,
        )
    return df


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregatie naar weekniveau (week eindigt op zaterdag, start = zondag).
    Past bij Pathzz-weekranges '2025-10-26 To 2025-11-01' e.d.
    """
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


# -------------
# MAIN UI (Region view)
# -------------

def main():
    st.title("PFM Region Performance Copilot – Regio-overzicht")

    # --- Retailer selectie via clients.json ---
    clients = load_clients("clients.json")
    clients_df = pd.DataFrame(clients)
    clients_df["label"] = clients_df.apply(
        lambda r: f"{r['brand']} – {r['name']} (company_id {r['company_id']})",
        axis=1,
    )

    st.sidebar.header("Selecteer retailer & regio")

    client_label = st.sidebar.selectbox("Retailer", clients_df["label"].tolist())
    selected_client = clients_df[clients_df["label"] == client_label].iloc[0].to_dict()
    company_id = int(selected_client["company_id"])

    # --- Locations & regions inladen ---
    try:
        locations_df = get_locations_by_company(company_id)
    except requests.exceptions.ReadTimeout:
        st.error(
            "De verbinding met de FastAPI-server duurde te lang bij het ophalen van de winkels "
            "(timeout). Probeer het nog eens of kies voorlopig een andere retailer."
        )
        return
    except requests.exceptions.RequestException as e:
        st.error(f"Fout bij ophalen van winkels uit FastAPI: {e}")
        return

    if locations_df.empty:
        st.error("Geen winkels gevonden voor deze retailer.")
        return

    # Region mapping inladen
    region_map = load_region_mapping()
    if region_map.empty:
        st.error("Geen geldige regions.csv gevonden (verwacht kolommen: shop_id;region).")
        return

    # Koppel regions.csv aan locations op shop_id = id
    locations_df["id"] = pd.to_numeric(locations_df["id"], errors="coerce").astype("Int64")
    merged = locations_df.merge(
        region_map,
        left_on="id",
        right_on="shop_id",
        how="inner",
    )

    if merged.empty:
        st.warning("Er zijn geen winkels met een regio-mapping voor deze retailer.")
        return

    available_regions = sorted(merged["region"].unique().tolist())
    region_choice = st.sidebar.selectbox("Regio", available_regions)

    region_shops = merged[merged["region"] == region_choice].copy()
    shop_ids = region_shops["id"].dropna().astype(int).unique().tolist()

    if not shop_ids:
        st.warning(f"Geen winkels gevonden voor regio '{region_choice}'.")
        return

    # --- Periode keuze ---
    period_choice = st.sidebar.selectbox(
        "Periode",
        [
            "Kalenderjaar 2024",
            "Laatste 26 weken",
        ],
        index=0,
    )

    today = datetime.today().date()

    if period_choice == "Kalenderjaar 2024":
        start_period = datetime(2024, 1, 1).date()
        end_period = datetime(2024, 12, 31).date()
    else:  # "Laatste 26 weken"
        end_period = today
        start_period = today - timedelta(weeks=26)

    run_btn = st.sidebar.button("Analyseer regio", type="primary")

    if not run_btn:
        st.info("Kies een retailer, regio en periode en klik op **Analyseer regio**.")
        return

    # --- Data ophalen uit FastAPI ---
    with st.spinner("Regionale data ophalen uit Storescan / FastAPI..."):
        metric_map = {
            "count_in": "footfall",
            "turnover": "turnover",
        }

        resp_all = get_report(
            shop_ids,
            list(metric_map.keys()),
            period="this_year",
            step="day",
            source="shops",
        )
        df_all_raw = normalize_vemcount_response(
            resp_all,
            kpi_keys=metric_map.keys(),
        )
        df_all_raw = df_all_raw.rename(columns=metric_map)

    if df_all_raw.empty:
        st.warning("Geen data gevonden voor deze regio.")
        return

    df_all_raw["date"] = pd.to_datetime(df_all_raw["date"], errors="coerce")
    df_all_raw = df_all_raw.dropna(subset=["date"])

    # Filter op periode
    start_ts = pd.Timestamp(start_period)
    end_ts = pd.Timestamp(end_period)

    df_period = df_all_raw[
        (df_all_raw["date"] >= start_ts) & (df_all_raw["date"] <= end_ts)
    ].copy()

    if df_period.empty:
        st.warning("Geen data in de geselecteerde periode voor deze regio.")
        return

    df_period = compute_daily_kpis(df_period)

    # --- Wekelijkse aggregatie voor de regio ---
    region_weekly = aggregate_weekly(df_period)

    # --- Pathzz street traffic per regio ---
    pathzz_weekly = load_pathzz_weekly_for_region(
        region_name=region_choice,
        start_date=start_period,
        end_date=end_period,
    )

    capture_weekly = pd.DataFrame()
    avg_capture = None

    if not pathzz_weekly.empty and not region_weekly.empty:
        pathzz_weekly["week_start"] = pd.to_datetime(pathzz_weekly["week_start"])
        region_weekly["week_start"] = pd.to_datetime(region_weekly["week_start"])

        capture_weekly = pd.merge(
            region_weekly,
            pathzz_weekly[["week_start", "street_footfall"]],
            on="week_start",
            how="inner",
        )

        if not capture_weekly.empty:
            capture_weekly["capture_rate"] = np.where(
                capture_weekly["street_footfall"] > 0,
                capture_weekly["footfall"] / capture_weekly["street_footfall"] * 100,
                np.nan,
            )

            capture_weekly = capture_weekly.sort_values("week_start")
            avg_capture = capture_weekly["capture_rate"].mean()

    # -----------------------
    # KPI cards op regioniveau
    # -----------------------

    st.subheader(f"{selected_client['brand']} – Regio {region_choice}")

    foot_total = df_period["footfall"].sum() if "footfall" in df_period.columns else 0
    turn_total = df_period["turnover"].sum() if "turnover" in df_period.columns else 0
    spv_avg = df_period["sales_per_visitor"].mean() if "sales_per_visitor" in df_period.columns else np.nan

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Footfall (regio, periode)", fmt_int(foot_total))
    with col2:
        st.metric("Omzet (regio, periode)", fmt_eur(turn_total))
    with col3:
        if not pd.isna(spv_avg):
            val = f"€ {spv_avg:.2f}".replace(".", ",")
        else:
            val = "-"
        st.metric("Gem. besteding/visitor", val)
    with col4:
        if avg_capture is not None and not pd.isna(avg_capture):
            st.metric("Gem. capture rate (regio)", fmt_pct(avg_capture))
        else:
            st.metric("Gem. capture rate (regio)", "-")

    # -----------------------
    # Grafiek: store vs street + capture-index
    # -----------------------

    st.markdown("### Regioweekbeeld – winkeltraffic vs straattraffic (Pathzz)")

    if not capture_weekly.empty:
        chart_df = capture_weekly[
            ["week_start", "footfall", "street_footfall", "capture_rate"]
        ].copy()

        max_foot = chart_df[["footfall", "street_footfall"]].max().max()
        if max_foot and max_foot > 0:
            max_cap = chart_df["capture_rate"].abs().max()
            if pd.notna(max_cap) and max_cap > 0:
                chart_df["capture_rate_index"] = (
                    chart_df["capture_rate"] / max_cap * max_foot
                )

        chart_df = chart_df.set_index("week_start")
        cols_for_chart = ["footfall", "street_footfall"]
        if "capture_rate_index" in chart_df.columns:
            cols_for_chart.append("capture_rate_index")

        st.line_chart(chart_df[cols_for_chart])

        # Tabel met ruwe waarden
        st.markdown("### Weekly tabel – regio-footfall, straattraffic & capture rate")
        table_df = capture_weekly[
            ["week_start", "footfall", "street_footfall", "capture_rate"]
        ].copy()
        table_df["week_start"] = table_df["week_start"].dt.strftime("%Y-%m-%d")
        table_df["capture_rate"] = table_df["capture_rate"].map(
            lambda x: fmt_pct(x) if not pd.isna(x) else "-"
        )

        table_df = table_df.rename(
            columns={
                "week_start": "Week start",
                "footfall": "Store footfall (regio)",
                "street_footfall": "Street footfall (Pathzz)",
                "capture_rate": "Capture rate",
            }
        )
        st.dataframe(table_df, use_container_width=True)
    else:
        st.info("Geen matchende Pathzz-weekdata gevonden voor deze regio/periode.")

    # -----------------------
    # Debug-sectie
    # -----------------------
    with st.expander("🔧 Debug regio"):
        st.write("Geselecteerde retailer:", selected_client)
        st.write("Regio mapping (subset):", region_shops[["id", "name", "region"]].head())
        st.write("Shop IDs regio:", shop_ids)
        st.write("Periode:", start_period, "→", end_period)
        st.write("df_all_raw (head):", df_all_raw.head())
        st.write("df_period (head):", df_period.head())
        st.write("Region weekly:", region_weekly.head())
        st.write("Pathzz weekly:", pathzz_weekly.head())
        st.write("Capture weekly:", capture_weekly.head())

        # Extra: ruwe Pathzz-head zodat je kunt zien wat er binnenkomt
        try:
            raw_pathzz = pd.read_csv(
                "data/pathzz_sample_weekly.csv",
                sep=";",
                dtype={"Visits": "string"},
            )
            st.write("Pathzz raw columns:", list(raw_pathzz.columns))
            st.write("Pathzz raw (head):", raw_pathzz.head())
        except Exception as e:
            st.write("Pathzz raw load error:", str(e))


if __name__ == "__main__":
    main()
