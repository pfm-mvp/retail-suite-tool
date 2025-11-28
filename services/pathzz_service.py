# services/pathzz_service.py

from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import requests
import streamlit as st

PATHZZ_API_KEY = st.secrets.get("PATHZZ_API_KEY", None)
PATHZZ_BASE_URL = st.secrets.get("PATHZZ_BASE_URL", "https://api.pathzz.com")

# Gebruik jouw gedownloade file:
SAMPLE_PATHZZ_FILE = Path("data/pathzz_sample_weekly.csv")


def _normalize_number(val) -> float:
    """
    Probeert Pathzz getallen zoals '16.725' of '16,725' om te zetten naar een 'echte' waarde.
    - Verwijdert punten/komma's als duizendtalseparator
    - Converteert naar float
    """
    if pd.isna(val):
        return 0.0
    s = str(val).strip()
    # haal zowel . als , eruit, dan naar float
    s = s.replace(".", "").replace(",", "")
    if not s:
        return 0.0
    return float(s)


def _load_sample_pathzz() -> pd.DataFrame:
    """
    Laadt een lokale Pathzz-sample dataset.

    Ondersteunt twee structuren:
    1) Monthly:
       - kolommen: 'month', 'street_footfall'
    2) Weekly:
       - kolom 'Week' met tekst als '2024-01-07 To 2024-01-13'
       - één of meer kolommen met 'Visits' in de naam
       -> wordt geaggregeerd naar maandniveau.
    """
    if not SAMPLE_PATHZZ_FILE.exists():
        raise FileNotFoundError(
            f"Sample Pathzz-bestand niet gevonden: {SAMPLE_PATHZZ_FILE.resolve()}"
        )

    df = pd.read_csv(SAMPLE_PATHZZ_FILE)

    # CASE 1: al monthly
    if "month" in df.columns and "street_footfall" in df.columns:
        df["month"] = pd.to_datetime(df["month"])
        return df[["month", "street_footfall"]]

    # CASE 2: weekly export met 'Week' + 'Visits' kolom
    if "Week" in df.columns:
        # parse week_start (eerste datum vóór ' To ')
        df["week_start"] = (
            df["Week"]
            .astype(str)
            .str.split("To")
            .str[0]
            .str.strip()
            .pipe(pd.to_datetime, errors="coerce")
        )

        # Zoek eerste kolom met 'Visits' in de naam én niet 'Comparison'
        visit_cols = [
            c for c in df.columns
            if "Visits" in c and "Comparison" not in c
        ]
        if not visit_cols:
            raise ValueError(
                "Kon geen kolom met 'Visits' (zonder 'Comparison') vinden in Pathzz CSV."
            )

        visit_col = visit_cols[0]
        df["street_footfall"] = df[visit_col].apply(_normalize_number)

        # Aggregatie naar maandniveau
        df["month"] = df["week_start"].dt.to_period("M").dt.to_timestamp()

        monthly = (
            df.groupby("month", as_index=False)["street_footfall"]
            .sum()
            .sort_values("month")
            .reset_index(drop=True)
        )
        return monthly

    raise ValueError(
        "Onbekende Pathzz-sample structuur. Verwacht óf 'month' + 'street_footfall', óf 'Week' + '...Visits'."
    )


def fetch_monthly_street_traffic(
    lat: float,
    lon: float,
    start_date: datetime,
    end_date: datetime,
    radius_m: int = 100,
) -> pd.DataFrame:
    """
    Haalt maandelijkse straatdrukte op.

    - Als PATHZZ_API_KEY aanwezig is: roept de echte Pathzz API aan (endpoint = placeholder).
    - Als PATHZZ_API_KEY ontbreekt: gebruikt een lokale sample dataset.
    """

    # MODE B – GEEN API KEY: sampledataset
    if PATHZZ_API_KEY is None:
        st.warning(
            "PATHZZ_API_KEY ontbreekt – gebruik Pathzz sample dataset als placeholder.",
            icon="⚠️",
        )
        df = _load_sample_pathzz()
        mask = (df["month"].dt.date >= start_date.date()) & (
            df["month"].dt.date <= end_date.date()
        )
        return df.loc[mask].reset_index(drop=True)

    # MODE A – LIVE API CALL (placeholder; pas endpoint & params aan op jullie contract)
    url = f"{PATHZZ_BASE_URL.rstrip('/')}/v1/street-traffic-monthly"
    headers = {"Authorization": f"Bearer {PATHZZ_API_KEY}"}
    params = {
        "lat": lat,
        "lon": lon,
        "radius": radius_m,
        "start": start_date.strftime("%Y-%m-%d"),
        "end": end_date.strftime("%Y-%m-%d"),
        "aggregation": "month",
    }

    resp = requests.get(url, headers=headers, params=params, timeout=60)
    resp.raise_for_status()
    data: Dict[str, Any] = resp.json()

    if "data" in data:
        df = pd.DataFrame(data["data"])
    else:
        df = pd.DataFrame(data)

    if "month" not in df.columns:
        raise ValueError("Pathzz-response mist 'month'-kolom, pas mapping in pathzz_service aan.")

    df["month"] = pd.to_datetime(df["month"])

    if "street_footfall" not in df.columns:
        if "visits" in df.columns:
            df = df.rename(columns={"visits": "street_footfall"})
        else:
            raise ValueError(
                "Pathzz-response mist 'street_footfall' (of 'visits'), pas mapping in pathzz_service aan."
            )

    mask = (df["month"].dt.date >= start_date.date()) & (
        df["month"].dt.date <= end_date.date()
    )
    return df.loc[mask, ["month", "street_footfall"]].reset_index(drop=True)
