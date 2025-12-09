# pages/06_Retai_AI_Region_Copilot.py

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
    - sqm_override  (float)
    - store_label   (mooie naam per winkel)
    """
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
        st.error(
            "Geen geldige regions.csv gevonden (verwacht minimaal kolommen: shop_id;region)."
        )
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

    # Effectieve sqm per winkel bepalen: override > API-sqm > NaN
    if "sqm" in merged.columns:
        merged["sqm_effective"] = np.where(
            merged["sqm_override"].notna(),
            merged["sqm_override"],
            pd.to_numeric(merged["sqm"], errors="coerce"),
        )
    else:
        merged["sqm_effective"] = merged["sqm_override"]

    # Label per winkel (mooie naam)
    if "store_label" in merged.columns and merged["store_label"].notna().any():
        merged["store_display"] = merged["store_label"]
    else:
        if "name" in merged.columns:
            merged["store_display"] = merged["name"]
        else:
            merged["store_display"] = merged["id"].astype(str)

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

    # Probeer een store-id kolom te vinden voor per-winkel-analyses
    store_key_col = None
    for cand in ["id", "shop_id", "location_id"]:
        if cand in df_all_raw.columns:
            store_key_col = cand
            break

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

    # Als we een store-id kolom hebben: join met region_shops voor sqm_effective + namen
    if store_key_col is not None:
        join_cols = ["id", "store_display", "region", "sqm_effective"]
        join_cols_existing = [c for c in join_cols if c in region_shops.columns]
        if "id" in join_cols_existing:
            df_period = df_period.merge(
                region_shops[join_cols_existing],
                left_on=store_key_col,
                right_on="id",
                how="left",
            )

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
    # Store-level sqm & index analyse (binnen regio)
    # -----------------------

    st.markdown("### Store performance in regio (per winkel)")

    store_table = pd.DataFrame()
    if store_key_col is not None and "turnover" in df_period.columns:
        # Per winkel aggregatie
        group_cols = [store_key_col]
        agg_dict = {
            "footfall": "sum",
            "turnover": "sum",
        }
        if "sales_per_visitor" in df_period.columns:
            agg_dict["sales_per_visitor"] = "mean"

        if "sqm_effective" in df_period.columns:
            agg_dict["sqm_effective"] = "max"

        store_agg = (
            df_period.groupby(group_cols, as_index=False)
            .agg(agg_dict)
        )

        # Turnover per sqm
        if "sqm_effective" in store_agg.columns:
            store_agg["turnover_per_sqm"] = np.where(
                store_agg["sqm_effective"] > 0,
                store_agg["turnover"] / store_agg["sqm_effective"],
                np.nan,
            )

            median_tps = store_agg["turnover_per_sqm"].median(skipna=True)
            store_agg["sqm_index"] = np.where(
                (store_agg["turnover_per_sqm"].notna()) & (median_tps > 0),
                store_agg["turnover_per_sqm"] / median_tps * 100,
                np.nan,
            )
        else:
            store_agg["turnover_per_sqm"] = np.nan
            store_agg["sqm_index"] = np.nan

        # Namen erbij hangen
        name_map = {}
        for _, r in region_shops.iterrows():
            name_map[int(r["id"])] = r.get("store_display", str(r["id"]))

        store_agg["store_name"] = store_agg[store_key_col].map(name_map)

        store_table = store_agg.copy()
        store_table = store_table.sort_values("sqm_index", ascending=True)

        # Mooie tabel
        tbl = store_table.copy()
        tbl["footfall"] = tbl["footfall"].map(fmt_int)
        tbl["turnover"] = tbl["turnover"].map(fmt_eur)
        tbl["sales_per_visitor"] = tbl["sales_per_visitor"].map(
            lambda x: f"€ {x:.2f}".replace(".", ",") if not pd.isna(x) else "-"
        )
        tbl["sqm_effective"] = tbl["sqm_effective"].map(fmt_int)
        tbl["turnover_per_sqm"] = tbl["turnover_per_sqm"].map(
            lambda x: fmt_eur(x) if not pd.isna(x) else "-"
        )
        tbl["sqm_index"] = tbl["sqm_index"].map(
            lambda x: fmt_pct(x - 100) if not pd.isna(x) else "-"
        )

        tbl = tbl.rename(
            columns={
                "store_name": "Winkel",
                "footfall": "Footfall",
                "turnover": "Omzet",
                "sales_per_visitor": "Gem. besteding/visitor",
                "sqm_effective": "m² (effectief)",
                "turnover_per_sqm": "Omzet per m²",
                "sqm_index": "m²-index t.o.v. regio",
            }
        )

        st.dataframe(tbl[[
            "Winkel",
            "Footfall",
            "Omzet",
            "Gem. besteding/visitor",
            "m² (effectief)",
            "Omzet per m²",
            "m²-index t.o.v. regio",
        ]], use_container_width=True)

        # Korte uitleg
        st.caption(
            "m²-index t.o.v. regio: 100 = gelijk aan regiomedian. "
            "Onder 100 → onderbenut potentieel per m², boven 100 → outperformer."
        )
    else:
        st.info(
            "Geen store-level ID of omzet beschikbaar in de dagdata – "
            "m²-indexanalyse wordt daarom overgeslagen."
        )

    # -----------------------
    # Grafiek: store vs street + capture-index (regio)
    # -----------------------

    st.markdown("### Regioweekbeeld – winkeltraffic vs straattraffic (Pathzz)")

    if not capture_weekly.empty:
        # We nemen nu ook de regio-omzet mee
        chart_df = capture_weekly[
            ["week_start", "footfall", "street_footfall", "turnover", "capture_rate"]
        ].copy()

        # Weeklabel als nette weeknummers, bv. W01, W02, ...
        iso_calendar = chart_df["week_start"].dt.isocalendar()
        chart_df["week_label"] = iso_calendar.week.apply(lambda w: f"W{int(w):02d}")

        week_order = (
            chart_df.sort_values("week_start")["week_label"]
            .unique()
            .tolist()
        )

        # Bars: footfall, street_footfall én omzet
        counts_long = chart_df.melt(
            id_vars=["week_label"],
            value_vars=["footfall", "street_footfall", "turnover"],
            var_name="metric",
            value_name="value",
        )

        bar_chart = (
            alt.Chart(counts_long)
            .mark_bar(width=20, opacity=0.8)
            .encode(
                x=alt.X("week_label:N", title="Week", sort=week_order),
                xOffset=alt.XOffset("metric:N"),
                y=alt.Y(
                    "value:Q",
                    axis=alt.Axis(
                        title="Footfall / streettraffic / omzet (regio)"
                    ),
                ),
                color=alt.Color(
                    "metric:N",
                    title="",
                    scale=alt.Scale(
                        domain=["footfall", "street_footfall", "turnover"],
                        range=["#1f77b4", "#ff7f0e", "#4ade80"],  # desnoods later PFM-kleuren
                    ),
                ),
                tooltip=[
                    alt.Tooltip("week_label:N", title="Week"),
                    alt.Tooltip("metric:N", title="Type"),
                    alt.Tooltip("value:Q", title="Waarde", format=",.0f"),
                ],
            )
        )

        # Lijn met capture rate (%)
        line_chart = (
            alt.Chart(chart_df)
            .mark_line(point=True, strokeWidth=2, color="#F04438")
            .encode(
                x=alt.X("week_label:N", title="Week", sort=week_order),
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

        # -----------------------
        # Weekly tabel – nu óók met omzet
        # -----------------------
        st.markdown("### Weekly tabel – regio-footfall, straattraffic, omzet & capture rate")

        table_df = capture_weekly[
            ["week_start", "footfall", "street_footfall", "turnover", "capture_rate"]
        ].copy()

        table_df["week_start"] = table_df["week_start"].dt.strftime("%Y-%m-%d")
        table_df["footfall"] = table_df["footfall"].map(fmt_int)
        table_df["street_footfall"] = table_df["street_footfall"].map(fmt_int)
        table_df["turnover"] = table_df["turnover"].map(fmt_eur)
        table_df["capture_rate"] = table_df["capture_rate"].map(
            lambda x: fmt_pct(x) if not pd.isna(x) else "-"
        )

        table_df = table_df.rename(
            columns={
                "week_start": "Week start",
                "footfall": "Store footfall (regio)",
                "street_footfall": "Street footfall (Pathzz)",
                "turnover": "Omzet (regio)",
                "capture_rate": "Capture rate",
            }
        )

        st.dataframe(table_df, use_container_width=True)

    else:
        st.info("Geen matchende Pathzz-weekdata gevonden voor deze regio/periode.")

    # ... vlak vóór de macro-context sectie
    cbs_retail_df = pd.DataFrame()
    cci_df_dbg = pd.DataFrame()

    # -----------------------
    # Macro-context: CBS detailhandel & consumentenvertrouwen
    # -----------------------

    st.markdown("### Macro-context: CBS detailhandel & consumentenvertrouwen")
    st.caption(
        "Regio-footfall- en omzet worden genormaliseerd op 100 = eerste maand in de geselecteerde periode. "
        "CBS-detailhandelindex en consumentenvertrouwen worden toegevoegd zodra de CBS-API geldige data teruggeeft."
    )

    macro_chart_shown = False
    cbs_retail_df = pd.DataFrame()
    cci_df = pd.DataFrame()

    # --- Helper om periodecodes 'YYYYMMxx' → datum te maken (15e van de maand) ---
    def _period_to_date(period_code: str) -> pd.Timestamp:
        if not isinstance(period_code, str):
            return pd.NaT
        s = period_code.strip()
        # bv. '2024MM06'
        if len(s) >= 8 and s[4:6].upper() == "MM":
            try:
                year = int(s[0:4])
                month = int(s[6:8])
                return pd.Timestamp(year=year, month=month, day=15)
            except Exception:
                return pd.NaT
        # fallback: probeer direct te parsen
        return pd.to_datetime(s, errors="coerce")

    # --- Regio-maanddata opbouwen (footfall + omzet) ---
    tmp = df_period.copy()
    tmp["month"] = tmp["date"].dt.to_period("M").dt.to_timestamp()

    agg_dict = {}
    if "turnover" in tmp.columns:
        agg_dict["turnover"] = "sum"
    if "footfall" in tmp.columns:
        agg_dict["footfall"] = "sum"

    region_month = tmp.groupby("month", as_index=False).agg(agg_dict)

    def _index_series(df: pd.DataFrame, value_col: str, label: str) -> pd.DataFrame:
        """
        Bouw een indexreeks (100 = eerste maand met >0 waarde).
        Output: kolommen [date, series, value]
        """
        if df.empty or value_col not in df.columns:
            return pd.DataFrame(columns=["date", "series", "value"])

        df = df.sort_values("month").copy()
        # eerste maand met echte waarde
        base_series = df.loc[df[value_col] > 0, value_col]
        if base_series.empty:
            return pd.DataFrame(columns=["date", "series", "value"])
        base = float(base_series.iloc[0])
        if base <= 0:
            return pd.DataFrame(columns=["date", "series", "value"])

        out = df[["month", value_col]].rename(columns={"month": "date"})
        out["value"] = out[value_col] / base * 100.0
        out["series"] = label
        return out[["date", "series", "value"]]

    # Regio-footfall-index & omzet-index
    series_list = []
    foot_idx = _index_series(region_month, "footfall", "Regio footfall-index")
    if not foot_idx.empty:
        series_list.append(foot_idx)

    turn_idx = _index_series(region_month, "turnover", "Regio omzet-index")
    if not turn_idx.empty:
        series_list.append(turn_idx)

    # --- CBS detailhandelindex proberen toe te voegen ---
    try:
        retail_raw = get_retail_index(
            series="Omzetontwikkeling_1",
            branch_code_or_title="DH_TOTAAL",
            months_back=24,
        )
    except Exception:
        retail_raw = []

    if isinstance(retail_raw, list) and retail_raw:
        cbs_retail_df = pd.DataFrame(retail_raw)
        if not cbs_retail_df.empty:
            if "date" not in cbs_retail_df.columns:
                cbs_retail_df["date"] = cbs_retail_df["period"].apply(_period_to_date)
            cbs_retail_df["date"] = pd.to_datetime(cbs_retail_df["date"], errors="coerce")
            cbs_retail_df = cbs_retail_df.dropna(subset=["date"])

            # Normaliseer CBS-index ook op 100 = eerste maand in deze reeks
            base_series = cbs_retail_df["retail_value"]
            if not base_series.empty:
                base = float(base_series.iloc[0]) if base_series.iloc[0] != 0 else None
            else:
                base = None

            if base is not None and base > 0:
                nat_line = cbs_retail_df[["date", "retail_value"]].copy()
                nat_line["value"] = nat_line["retail_value"] / base * 100.0
                nat_line["series"] = "CBS detailhandelindex"
                series_list.append(nat_line[["date", "series", "value"]])

    # --- Hoofdgrafiek: regio-indexen + CBS-detailhandel (indien beschikbaar) ---
    if series_list:
        chart_all = pd.concat(series_list, ignore_index=True).sort_values("date")

        macro_chart = (
            alt.Chart(chart_all)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title="Maand"),
                y=alt.Y("value:Q", title="Index (100 = startperiode)"),
                color=alt.Color("series:N", title="Reeks"),
                tooltip=[
                    alt.Tooltip("date:T", title="Maand"),
                    alt.Tooltip("series:N", title="Reeks"),
                    alt.Tooltip("value:Q", title="Index", format=".1f"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(macro_chart, use_container_width=True)
        macro_chart_shown = True
    else:
        st.info(
            "Kon geen regio-index opbouwen (footfall/omzet). Controleer of er omzet- en footfalldata beschikbaar is."
        )

    # --- Maandentabel: regio vs CBS ---
    if not region_month.empty:
        table_df = region_month.rename(columns={"month": "Maand"}).copy()
        table_df["Maand"] = table_df["Maand"].dt.strftime("%Y-%m")
        if "turnover" in table_df.columns:
            table_df["Omzet (regio)"] = table_df["turnover"].map(fmt_eur)
        if "footfall" in table_df.columns:
            table_df["Footfall (regio)"] = table_df["footfall"].map(fmt_int)

        # Indexen t.o.v. start (footfall & omzet)
        if "footfall" in region_month.columns:
            fi = _index_series(region_month, "footfall", "fi")
            if not fi.empty:
                idx_map = dict(zip(fi["date"], fi["value"]))
                table_df["Footfall-index vs start"] = table_df["Maand"].map(
                    lambda m: fmt_pct(idx_map.get(pd.to_datetime(m), np.nan) - 100)
                    if idx_map.get(pd.to_datetime(m), np.nan) is not np.nan
                    else "-"
                )
        if "turnover" in region_month.columns:
            ti = _index_series(region_month, "turnover", "ti")
            if not ti.empty:
                idx_map_t = dict(zip(ti["date"], ti["value"]))
                table_df["Omzet-index vs start"] = table_df["Maand"].map(
                    lambda m: fmt_pct(idx_map_t.get(pd.to_datetime(m), np.nan) - 100)
                    if idx_map_t.get(pd.to_datetime(m), np.nan) is not np.nan
                    else "-"
                )

        # CBS-kolom (optioneel)
        if not cbs_retail_df.empty:
            cbs_map = dict(
                zip(
                    cbs_retail_df["date"].dt.strftime("%Y-%m"),
                    cbs_retail_df["retail_value"],
                )
            )
            table_df["CBS detailhandelindex"] = table_df["Maand"].map(
                lambda m: f"{cbs_map.get(m):.1f}" if m in cbs_map else "-"
            )
        else:
            table_df["CBS detailhandelindex"] = "-"

        # Alleen de zichtbare kolommen tonen
        cols_show = ["Maand"]
        if "Omzet (regio)" in table_df.columns:
            cols_show.append("Omzet (regio)")
        if "Footfall (regio)" in table_df.columns:
            cols_show.append("Footfall (regio)")
        if "Omzet-index vs start" in table_df.columns:
            cols_show.append("Omzet-index vs start")
        if "Footfall-index vs start" in table_df.columns:
            cols_show.append("Footfall-index vs start")
        cols_show.append("CBS detailhandelindex")

        st.markdown("#### Maandoverzicht – Regio vs CBS")
        st.dataframe(table_df[cols_show], use_container_width=True)

    # --- Tweede grafiek: Consumentenvertrouwen vs regionale performance ---
    st.markdown("#### Consumentenvertrouwen vs regionale performance")

    try:
        cci_raw = get_cci_series(months_back=24)
    except Exception:
        cci_raw = []

    if isinstance(cci_raw, list) and cci_raw:
        cci_df = pd.DataFrame(cci_raw)
        if not cci_df.empty:
            cci_df["date"] = cci_df["period"].apply(_period_to_date)
            cci_df["date"] = pd.to_datetime(cci_df["date"], errors="coerce")
            cci_df = cci_df.dropna(subset=["date"]).sort_values("date")

            # Consumentenvertrouwen zelf normaliseren op 100 = eerste waarde in deze reeks
            base_cci = float(cci_df["cci"].iloc[0]) if cci_df["cci"].iloc[0] != 0 else None
            if base_cci is not None:
                cci_df["cci_index"] = (cci_df["cci"] / base_cci) * 100.0

                # Bouw indexreeksen voor footfall & omzet over dezelfde maanden
                cci_months = cci_df["date"].dt.to_period("M").dt.to_timestamp().unique()
                region_match = region_month.copy()
                region_match = region_match[region_match["month"].isin(cci_months)]

                series_cci = []

                if not region_match.empty:
                    fi2 = _index_series(region_match, "footfall", "Regio footfall-index")
                    if not fi2.empty:
                        series_cci.append(fi2)
                    ti2 = _index_series(region_match, "turnover", "Regio omzet-index")
                    if not ti2.empty:
                        series_cci.append(ti2)

                # CCI-lijn als index
                cci_line = cci_df[["date", "cci_index"]].copy()
                cci_line = cci_line.rename(columns={"cci_index": "value"})
                cci_line["series"] = "Consumentenvertrouwen (index)"
                series_cci.append(cci_line)

                chart_cci = (
                    alt.Chart(pd.concat(series_cci, ignore_index=True))
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("date:T", title="Maand"),
                        y=alt.Y("value:Q", title="Index (100 = startperiode)"),
                        color=alt.Color("series:N", title="Reeks"),
                        tooltip=[
                            alt.Tooltip("date:T", title="Maand"),
                            alt.Tooltip("series:N", title="Reeks"),
                            alt.Tooltip("value:Q", title="Index", format=".1f"),
                        ],
                    )
                    .properties(height=260)
                )
                st.altair_chart(chart_cci, use_container_width=True)
                macro_chart_shown = True
    else:
        st.caption("Geen geldige consumentenvertrouwensreeks gevonden in CBS-API.")

    if not macro_chart_shown:
        st.caption(
            "Macro-index (CBS detailhandel / consumentenvertrouwen) kon niet volledig worden opgebouwd. "
            "Controleer eventueel de output van get_retail_index() en get_cci_series() in cbs_service."
        )
    # -----------------------
    # Debug-sectie
    # -----------------------
    with st.expander("🔧 Debug regio"):
        st.write("Geselecteerde retailer:", selected_client)
        st.write("Region mapping (subset):", region_shops[[
            "id", "store_display", "region", "sqm_effective"
        ]].head())
        st.write("Shop IDs regio:", shop_ids)
        st.write("Periode:", start_period, "→", end_period)
        st.write("Store key column in df_all_raw:", store_key_col)
        st.write("df_all_raw (head):", df_all_raw.head())
        st.write("df_period (head):", df_period.head())
        st.write("Region monthly:", region_month.head() if 'region_month' in locals() else "n.v.t.")
        st.write("CBS retail index (sample):", cbs_retail_df.head() if not cbs_retail_df.empty else "empty")
        st.write("CBS CCI (sample):", cci_df_dbg.head() if not cci_df_dbg.empty else "empty")
        st.write("Region weekly:", region_weekly.head())
        st.write("Pathzz weekly:", pathzz_weekly.head())
        st.write("Capture weekly:", capture_weekly.head())
        st.write("Store table (raw):", store_table.head() if not store_table.empty else "n.v.t.")

if __name__ == "__main__":
    main()
