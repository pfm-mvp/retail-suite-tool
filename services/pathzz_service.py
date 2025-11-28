# services/pathzz_service.py

import os
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import requests
import streamlit as st
from pathlib import Path

PATHZZ_API_KEY = st.secrets.get("PATHZZ_API_KEY", None)
PATHZZ_BASE_URL = st.secrets.get("PATHZZ_BASE_URL", "https://api.pathzz.com")  # placeholder

# Pad naar de sampledataset
SAMPLE_PATHZZ_FILE = Path("data/pathzz_sample_monthly.csv")


def _load_sample_pathzz() -> pd.DataFrame:
    """
    Laadt een lokale sampledataset met maandelijkse street_footfall.

    Verwacht CSV-structuur:
    month,street_footfall
    2024-01-01,125000
    ...
    """
    if not SAMPLE_PATHZZ_FILE.exists():
        raise FileNotFoundError(
            f"Sample Pathzz-bestand niet gevonden: {SAMPLE_PATHZZ_FILE.resolve()}"
        )

    df = pd.read_csv(SAMPLE_PATHZZ_FILE)
    if "month" not in df.columns or "street_footfall" not in df.columns:
        raise ValueError(
            "pathzz_sample_monthly.csv moet ten minste kolommen 'month' en 'street_footfall' bevatten."
        )

    df["month"] = pd.to_datetime(df["month"])
    return df[["month", "street_footfall"]]


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

    Dit maakt het mogelijk om de AI Copilot nu al te demo'en,
    en later naadloos over te schakelen naar live Pathzz-data.
    """

    # MODE B – geen API key: sampledataset gebruiken
    if PATHZZ_API_KEY is None:
        st.warning(
            "PATHZZ_API_KEY ontbreekt – gebruik sample Pathzz dataset als placeholder.",
            icon="⚠️",
        )
        df = _load_sample_pathzz()

        # optioneel: filteren op periode
        mask = (df["month"].dt.date >= start_date.date()) & (
            df["month"].dt.date <= end_date.date()
        )
        return df.loc[mask].reset_index(drop=True)

    # MODE A – echte Pathzz API
    url = f"{PATHZZ_BASE_URL.rstrip('/')}/v1/street-traffic-monthly"  # voorbeeld endpoint
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

    if "data" not in data:
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame(data["data"])

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

    # eventueel filteren op periode
    mask = (df["month"].dt.date >= start_date.date()) & (
        df["month"].dt.date <= end_date.date()
    )
    return df.loc[mask, ["month", "street_footfall"]].reset_index(drop=True)
