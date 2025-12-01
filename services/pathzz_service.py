# services/pathzz_service.py

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Union

import pandas as pd

# Pad naar de demo-CSV met straatdrukte (wekelijkse visits)
PATHZZ_SAMPLE_FILE = Path("data/pathzz_sample_weekly.csv")


@lru_cache(maxsize=1)
def _load_pathzz_sample() -> pd.DataFrame:
    """
    Laadt de demo Pathzz-data uit /data/pathzz_sample_weekly.csv.

    Verwacht kolommen:
    - Week: "YYYY-MM-DD To YYYY-MM-DD"
    - Visits: numeriek (met punt als duizendtalseparator in jouw file)
    """
    if not PATHZZ_SAMPLE_FILE.exists():
        # Geen bestand gevonden -> lege DF teruggeven
        return pd.DataFrame(columns=["week_start", "week_end", "Visits"])

    # CSV met ; gescheiden
    df = pd.read_csv(PATHZZ_SAMPLE_FILE, sep=";")

    if "Week" not in df.columns or "Visits" not in df.columns:
        return pd.DataFrame(columns=["week_start", "week_end", "Visits"])

    # Week-bereik parsen naar start- en einddatum
    def parse_range(s: str):
        try:
            start_str, end_str = str(s).split(" To ")
            start_dt = datetime.strptime(start_str.strip(), "%Y-%m-%d").date()
            end_dt = datetime.strptime(end_str.strip(), "%Y-%m-%d").date()
            return start_dt, end_dt
        except Exception:
            return None, None

    starts = []
    ends = []
    for val in df["Week"]:
        s, e = parse_range(val)
        starts.append(s)
        ends.append(e)

    df["week_start"] = starts
    df["week_end"] = ends

    # Visits numeriek maken
    # Voor jouw data is "16.725" waarschijnlijk 16725 bezoeken.
    df["Visits"] = (
        df["Visits"]
        .astype(str)
        .str.replace(".", "", regex=False)  # duizendtallen eruit
        .str.replace(",", ".", regex=False)
    )
    df["Visits"] = pd.to_numeric(df["Visits"], errors="coerce")

    df = df[
        df["week_start"].notna()
        & df["week_end"].notna()
        & df["Visits"].notna()
    ].copy()

    return df


def fetch_monthly_street_traffic(
    lat: float,
    lon: float,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    radius_m: int = 100,
) -> pd.DataFrame:
    """
    Demo-implementatie voor straatdrukte op maandniveau.

    Voor nu:
    - Negeert lat/lon/radius en gekozen periode.
    - Gebruikt ALTIJD de volledige pathzz_sample_weekly.csv.
    - Zet weeks om naar 'mid_date' en groepeert naar maand.

    Return:
        DataFrame met kolommen:
        - month (Timestamp, begin van maand)
        - street_footfall (som van Visits in die maand)
    """
    df = _load_pathzz_sample()
    if df.empty:
        return pd.DataFrame(columns=["month", "street_footfall"])

    # Representatieve datum per week = midden van het bereik
    df["mid_date"] = df["week_start"] + (
        df["week_end"] - df["week_start"]
    ) / 2

    # Maand bepalen op basis van mid_date
    df["month"] = (
        pd.to_datetime(df["mid_date"])
        .to_period("M")
        .dt.to_timestamp()
    )

    monthly = (
        df.groupby("month", as_index=False)["Visits"]
        .sum()
        .rename(columns={"Visits": "street_footfall"})
    )

    return monthly
