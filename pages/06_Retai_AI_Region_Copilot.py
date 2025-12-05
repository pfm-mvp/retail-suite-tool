# pages/06_Retai_AI_Region_Copilot.py

import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt

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

    Optioneel:
    - sqm_region  (float, m² per winkel)

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

    # Optionele sqm_region naar float
    if "sqm_region" in df.columns:
        df["sqm_region"] = pd.to_numeric(df["sqm_region"], errors="coerce")

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


@st.cache_data(ttl=600)
def fetch_region_street_traffic(region: str, start_date, end_date) -> pd.DataFrame:
    """
    Leest demo-straattraffic per regio uit data/pathzz_sample_weekly.csv

    Verwachte CSV-structuur (opgeschoond):
    Region;Week;Visits

    Voorbeeld:
    Noord;2024-01-14 To 2024-01-20;43.931

    Visits zijn waardes als 43.931 (→ 43931 bezoekers).
    Return:
    - week_start (datetime)
    - street_footfall (float)
    """
    csv_path = "data/pathzz_sample_weekly.csv"
    try:
        # Alleen de eerste 3 kolommen lezen, ongeacht eventuele rommel erachter
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

    # Kolommen forceren naar vaste namen
    df.columns = ["region", "week", "street_footfall"]

    # Regio case-insensitive matchen
    df["region"] = df["region"].astype(str).str.strip()
    region_norm = str(region).strip().lower()
    df = df[df["region"].str.lower() == region_norm].copy()
    if df.empty:
        return pd.DataFrame()

    # Visits: "43.931" → "43931" → 43931.0
    df["street_footfall"] = (
        df["street_footfall"]
        .astype(str)
        .str.strip()
        .replace("", np.nan)
    )
    df = df.dropna(subset=["street_footfall"])

    df["street_footfall"] = (
        df["street_footfall"]
        .str.replace(".", "", regex=False)   # punt = duizendscheiding
        .str.replace(",", ".", regex=False)  # safety
        .astype(float)
    )

    # "2024-01-14 To 2024-01-20" → 2024-01-14
    def _parse_week_start(s: str):
        if isinstance(s, str) and "To" in s:
            return pd.to_datetime(s.split("To")[0].strip(), errors="coerce")
        return pd.NaT

    df["week_start"] = df["week"].apply(_parse_week_start)
    df = df.dropna(subset=["week_start"])

    # Filter op aangevraagde periode
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

    # 👉 hier vullen we sqm vanuit regions.csv als fallback
    if "sqm_region" in region_shops.columns:
        if "sqm" in region_shops.columns:
            region_shops["sqm"] = region_shops["sqm"].fillna(region_shops["sqm_region"])
        else:
            region_shops["sqm"] = region_shops["sqm_region"]

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
    pathzz_weekly = fetch_region_street_traffic(
        region=region_choice,
        start_date=start_period,
        end_date=end_period,
    )

    capture_weekly = pd.DataFrame()
    avg_capture = None
    corr_store_street = None
    corr_turn_street = None

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

            # Correlaties over tijd: street vs store-footfall & omzet
            try:
                x_street = capture_weekly["street_footfall"].astype(float).values
                y_store = capture_weekly["footfall"].astype(float).values
                y_turn = capture_weekly["turnover"].astype(float).values

                if len(x_street) >= 3:
                    if np.std(x_street) > 0 and np.std(y_store) > 0:
                        corr_store_street = float(
                            np.corrcoef(x_street, y_store)[0, 1]
                        )
                    if np.std(x_street) > 0 and np.std(y_turn) > 0:
                        corr_turn_street = float(
                            np.corrcoef(x_street, y_turn)[0, 1]
                        )
            except Exception:
                corr_store_street = None
                corr_turn_street = None

    # -----------------------
    # Store-level performance incl. m²-index
    # -----------------------

    # Koppel shop-meta (sqm, name, city)
    region_shops_short = region_shops[["id", "name", "city", "sqm"]].copy()
    region_shops_short["sqm"] = pd.to_numeric(region_shops_short["sqm"], errors="coerce")

    store_perf = (
        df_period
        .groupby("id", as_index=False)
        .agg(
            footfall=("footfall", "sum"),
            turnover=("turnover", "sum"),
            sales_per_visitor=("sales_per_visitor", "mean"),
            conversion_rate=("conversion_rate", "mean"),
        )
    )

    store_perf = store_perf.merge(
        region_shops_short,
        left_on="id",
        right_on="id",
        how="left",
    )

    # Totale regio-oppervlakte (alle winkels met geldige sqm)
    region_total_sqm = store_perf["sqm"].where(store_perf["sqm"] > 0).sum()

    # Omzet per m² en footfall per m²
    store_perf["turnover_per_sqm"] = np.where(
        store_perf["sqm"] > 0,
        store_perf["turnover"] / store_perf["sqm"],
        np.nan,
    )
    store_perf["footfall_per_sqm"] = np.where(
        store_perf["sqm"] > 0,
        store_perf["footfall"] / store_perf["sqm"],
        np.nan,
    )

    # Regio-gemiddelde omzet per m² (gebaseerd op som omzet / som m²)
    region_total_turnover = store_perf["turnover"].sum()
    region_total_footfall = store_perf["footfall"].sum()

    if region_total_sqm and region_total_sqm > 0:
        region_avg_turnover_per_sqm = region_total_turnover / region_total_sqm
    else:
        region_avg_turnover_per_sqm = np.nan

    # CSm²I: omzet per m² t.o.v. regiogemiddelde (=100 is gemiddeld)
    store_perf["csm2i"] = np.where(
        ~pd.isna(store_perf["turnover_per_sqm"]) & (region_avg_turnover_per_sqm > 0),
        (store_perf["turnover_per_sqm"] / region_avg_turnover_per_sqm) * 100.0,
        np.nan,
    )

    # Aandeel in regionale omzet & footfall
    store_perf["share_turnover_pct"] = np.where(
        region_total_turnover > 0,
        store_perf["turnover"] / region_total_turnover * 100.0,
        np.nan,
    )
    store_perf["share_footfall_pct"] = np.where(
        region_total_footfall > 0,
        store_perf["footfall"] / region_total_footfall * 100.0,
        np.nan,
    )

    # Performance segment (voor kleur in scatter)
    def classify_segment(row):
        if pd.isna(row["csm2i"]):
            return "Onbekend"
        if row["csm2i"] >= 110:
            return "Overperformer (≥110)"
        if row["csm2i"] <= 90:
            return "Underperformer (≤90)"
        return "Middenveld (90-110)"

    store_perf["segment"] = store_perf.apply(classify_segment, axis=1)

    # -----------------------
    # KPI cards op regioniveau
    # -----------------------

    st.subheader(f"{selected_client['brand']} – Regio {region_choice}")

    foot_total = region_total_footfall if not pd.isna(region_total_footfall) else 0
    turn_total = region_total_turnover if not pd.isna(region_total_turnover) else 0
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
        st.metric("Gem. besteding/visitor (regio)", val)
    with col4:
        if avg_capture is not None and not pd.isna(avg_capture):
            st.metric("Gem. capture rate (regio)", fmt_pct(avg_capture))
        else:
            st.metric("Gem. capture rate (regio)", "-")

    # Kleine extra context rond m²
    if region_total_sqm and region_total_sqm > 0:
        st.caption(
            f"🔎 Regio-oppervlak: {fmt_int(region_total_sqm)} m² – "
            f"gemiddelde omzet/m²: {fmt_eur(region_avg_turnover_per_sqm)} per geselecteerde periode."
        )

    # -----------------------
    # Regio m²-index (CSm²I) – store ranking
    # -----------------------

    st.markdown("### 📊 Regio performance per m² – CSm²I")

    st.markdown(
        """
        <small>
        <strong>CSm²I</strong> = omzet per m² t.o.v. het regiogemiddelde (=100).<br>
        > 110 ⇒ winkel benut zijn m² bovengemiddeld goed. <br>
        < 90 ⇒ winkel laat omzet liggen per m², relatief t.o.v. de regio.
        </small>
        """,
        unsafe_allow_html=True,
    )

    # Tabel voor regiomanager (gesorteerd op CSm²I dalend)
    table_cols = [
        "name",
        "city",
        "sqm",
        "footfall",
        "turnover",
        "turnover_per_sqm",
        "footfall_per_sqm",
        "sales_per_visitor",
        "conversion_rate",
        "csm2i",
        "share_turnover_pct",
        "share_footfall_pct",
        "segment",
    ]

    tbl = store_perf[table_cols].copy()
    tbl = tbl.rename(
        columns={
            "name": "Winkel",
            "city": "Plaats",
            "sqm": "m²",
            "footfall": "Footfall",
            "turnover": "Omzet",
            "turnover_per_sqm": "Omzet/m²",
            "footfall_per_sqm": "Footfall/m²",
            "sales_per_visitor": "SPV",
            "conversion_rate": "Conversie (%)",
            "csm2i": "CSm²I (index)",
            "share_turnover_pct": "Aandeel omzet (%)",
            "share_footfall_pct": "Aandeel footfall (%)",
            "segment": "Segment",
        }
    )

    # EU-formattering
    for col in ["Footfall", "Omzet", "m²", "Footfall/m²"]:
        if col in tbl.columns:
            tbl[col] = tbl[col].map(lambda x: fmt_int(x) if not pd.isna(x) else "-")

    if "Omzet/m²" in tbl.columns:
        tbl["Omzet/m²"] = tbl["Omzet/m²"].map(
            lambda x: fmt_eur(x).replace("€ ", "") if x != "-" else "-"
        )

    if "SPV" in tbl.columns:
        tbl["SPV"] = tbl["SPV"].map(
            lambda x: f"€ {x:.2f}".replace(".", ",") if not pd.isna(x) else "-"
        )

    if "Conversie (%)" in tbl.columns:
        tbl["Conversie (%)"] = tbl["Conversie (%)"].map(
            lambda x: fmt_pct(x) if not pd.isna(x) else "-"
        )

    if "CSm²I (index)" in tbl.columns:
        tbl["CSm²I (index)"] = tbl["CSm²I (index)"].map(
            lambda x: f"{x:.0f}".replace(".", ",") if not pd.isna(x) else "-"
        )

    if "Aandeel omzet (%)" in tbl.columns:
        tbl["Aandeel omzet (%)"] = tbl["Aandeel omzet (%)"].map(
            lambda x: fmt_pct(x) if not pd.isna(x) else "-"
        )

    if "Aandeel footfall (%)" in tbl.columns:
        tbl["Aandeel footfall (%)"] = tbl["Aandeel footfall (%)"].map(
            lambda x: fmt_pct(x) if not pd.isna(x) else "-"
        )

    # Sorteren op CSm²I (hoog → laag)
    if "CSm²I (index)" in tbl.columns:
        # Voor sortering: maak een helperkolom
        store_perf_sorted = store_perf.copy()
        store_perf_sorted = store_perf_sorted.sort_values("csm2i", ascending=False)
        ordered_ids = store_perf_sorted["id"].tolist()
        tbl["__id"] = store_perf["id"]
        tbl["__order"] = tbl["__id"].apply(lambda i: ordered_ids.index(i) if i in ordered_ids else 9999)
        tbl = tbl.sort_values("__order").drop(columns=["__id", "__order"])

    st.dataframe(tbl, use_container_width=True)

    # Top/bottom lijstjes voor snelle actie
    if "CSm²I (index)" in tbl.columns:
        with st.expander("🔍 Snel overzicht: top & bottom op CSm²I"):
            # Terug naar ruwe csm2i voor logica
            sp = store_perf.copy()
            sp_valid = sp[~pd.isna(sp["csm2i"])].copy()

            top_n = sp_valid.sort_values("csm2i", ascending=False).head(5)
            bottom_n = sp_valid.sort_values("csm2i", ascending=True).head(5)

            col_top, col_bottom = st.columns(2)
            with col_top:
                st.markdown("**Top 5 – sterkste omzet/m² (CSm²I)**")
                for _, r in top_n.iterrows():
                    st.write(
                        f"- {r['name']} ({r.get('city', '')}) – "
                        f"CSm²I: {r['csm2i']:.0f}, SPV: {fmt_eur(r['sales_per_visitor'])}"
                    )

            with col_bottom:
                st.markdown("**Top 5 – meeste ruimte voor verbetering (CSm²I)**")
                for _, r in bottom_n.iterrows():
                    st.write(
                        f"- {r['name']} ({r.get('city', '')}) – "
                        f"CSm²I: {r['csm2i']:.0f}, SPV: {fmt_eur(r['sales_per_visitor'])}"
                    )

    # -----------------------
    # Scatter: CSm²I vs SPV (bubble = footfall)
    # -----------------------

    st.markdown("### 🔺 Quadrant – benutting m² vs klantwaarde")

    scatter_df = store_perf.copy()
    scatter_df = scatter_df[~pd.isna(scatter_df["csm2i"])].copy()

    if not scatter_df.empty:
        scatter_df["label"] = scatter_df.apply(
            lambda r: f"{r.get('name', 'Store')} ({r.get('city', '')})", axis=1
        )

        chart = (
            alt.Chart(scatter_df)
            .mark_circle(opacity=0.8)
            .encode(
                x=alt.X(
                    "csm2i:Q",
                    title="CSm²I – omzet per m² (regio=100)",
                    scale=alt.Scale(zero=False),
                ),
                y=alt.Y(
                    "sales_per_visitor:Q",
                    title="Gem. besteding per bezoeker (SPV, €)",
                ),
                size=alt.Size(
                    "footfall:Q",
                    title="Footfall (periode)",
                    legend=alt.Legend(orient="right"),
                ),
                color=alt.Color(
                    "segment:N",
                    title="Segment",
                    scale=alt.Scale(
                        domain=[
                            "Overperformer (≥110)",
                            "Middenveld (90-110)",
                            "Underperformer (≤90)",
                            "Onbekend",
                        ],
                        range=["#16a34a", "#0ea5e9", "#f97316", "#6b7280"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("label:N", title="Winkel"),
                    alt.Tooltip("csm2i:Q", title="CSm²I", format=".0f"),
                    alt.Tooltip("sales_per_visitor:Q", title="SPV (€)", format=".2f"),
                    alt.Tooltip("footfall:Q", title="Footfall", format=",.0f"),
                    alt.Tooltip("turnover:Q", title="Omzet", format=",.0f"),
                    alt.Tooltip("segment:N", title="Segment"),
                ],
            )
            .properties(height=350)
        )

        st.altair_chart(chart, use_container_width=True)

        st.caption(
            "💡 Linksboven: lage CSm²I maar hoge SPV – mogelijk te weinig footfall of te veel m². "
            "Rechtsonder: hoge CSm²I maar lage SPV – vloer presteert goed, maar klantwaarde per bezoeker kan omhoog."
        )
    else:
        st.info("Onvoldoende m²-data om een CSm²I-scatter te tonen voor deze regio.")

    # -----------------------
    # Grafiek: store vs street + capture-index
    # -----------------------

    st.markdown("### Regioweekbeeld – winkeltraffic vs straattraffic (Pathzz)")

    if not capture_weekly.empty:
        chart_df = capture_weekly[
            ["week_start", "footfall", "street_footfall", "capture_rate"]
        ].copy()

        # Weeklabel als weeknummer: W01, W02, ...
        iso_calendar = chart_df["week_start"].dt.isocalendar()
        chart_df["week_label"] = "W" + iso_calendar.week.astype(str)

        # Data voor de bars (footfall vs street_footfall)
        counts_long = chart_df.melt(
            id_vars=["week_label"],
            value_vars=["footfall", "street_footfall"],
            var_name="metric",
            value_name="value",
        )

        bar_chart = (
            alt.Chart(counts_long)
            .mark_bar(width=20, opacity=0.8)
            .encode(
                x=alt.X("week_label:N", title="Week", sort=None),
                xOffset=alt.XOffset("metric:N"),
                y=alt.Y(
                    "value:Q",
                    axis=alt.Axis(title="Footfall / streettraffic (regio)"),
                ),
                color=alt.Color(
                    "metric:N",
                    title="",
                    scale=alt.Scale(
                        domain=["footfall", "street_footfall"],
                        range=["#1f77b4", "#ff7f0e"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("week_label:N", title="Week"),
                    alt.Tooltip("metric:N", title="Type"),
                    alt.Tooltip("value:Q", title="Aantal", format=",.0f"),
                ],
            )
        )

        # Lijn met capture rate
        line_chart = (
            alt.Chart(chart_df)
            .mark_line(point=True, strokeWidth=2, color="#F04438")
            .encode(
                x=alt.X("week_label:N", title="Week", sort=None),
                y=alt.Y(
                    "capture_rate:Q",
                    axis=alt.Axis(title="Capture rate regio (%)"),
                    scale=alt.Scale(zero=True),
                ),
                tooltip=[
                    alt.Tooltip("week_label:N", title="Week"),
                    alt.Tooltip(
                        "capture_rate:Q",
                        title="Capture rate",
                        format=".1f",
                    ),
                ],
            )
        )

        combined = (
            alt.layer(bar_chart, line_chart)
            .resolve_scale(y="independent")
            .properties(height=350)
        )

        st.altair_chart(combined, use_container_width=True)

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

        # Correlatie-duiding
        corr_txt_parts = []
        if corr_store_street is not None:
            corr_txt_parts.append(
                f"correlatie streettraffic ↔ regio-footfall ≈ {corr_store_street:.2f}"
            )
        if corr_turn_street is not None:
            corr_txt_parts.append(
                f"correlatie streettraffic ↔ regio-omzet ≈ {corr_turn_street:.2f}"
            )

        if corr_txt_parts:
            st.caption(
                "📈 " + " | ".join(corr_txt_parts) +
                " – hoe dichter bij 1.00, hoe sterker de relatie tussen drukte op straat en performance in de winkels."
            )

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
        st.write("Store performance (head):", store_perf.head())
        st.write("Region total sqm:", region_total_sqm)
        st.write("Region avg turnover/m²:", region_avg_turnover_per_sqm)
        st.write("corr_store_street:", corr_store_street)
        st.write("corr_turn_street:", corr_turn_street)


if __name__ == "__main__":
    main()
