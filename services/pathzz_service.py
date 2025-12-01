# services/pathzz_service.py

from __future__ import annotations

from datetime import datetime, date
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

# Pad naar de demo-CSV met straatdrukte (wekelijkse visits)
PATHZZ_SAMPLE_FILE = Path("data/pathzz_sample_weekly.csv")


@lru_cache(maxsize=1)
def _load_pathzz_sample() -> pd.DataFrame:
    """
    Laadt de demo Pathzz-data uit /data/pathzz_sample_weekly.csv.

    Verwacht kolommen:
    - Week: "YYYY-MM-DD To YYYY-MM-DD"
    - Visits: numeriek (kan met punt als decimaal)
    """

    if not PATHZZ_SAMPLE_FILE.exists():
        # Geen bestand gevonden -> lege DF teruggeven
        return pd.DataFrame(columns=["week_start", "week_end", "Visits"])

    df = pd.read_csv(PATHZZ_SAMPLE_FILE, sep=";")

    # Zorg dat kolomnamen kloppen
    if "Week" not in df.columns or "Visits" not in df.columns:
        return pd.DataFrame(columns=["week_start", "week_end", "Visits"])

    # Parseer weekbereik naar start- en einddatum
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
    df["Visits"] = (
        df["Visits"]
        .astype(str)
        .str.replace(".", "", regex=False)  # mocht er duizendtallen in zitten
        .str.replace(",", ".", regex=False)
    )
    df["Visits"] = pd.to_numeric(df["Visits"], errors="coerce")

    # Filter regels zonder geldige datums of visits
    df = df[
        df["week_start"].notna()
        & df["week_end"].notna()
        & df["Visits"].notna()
    ].copy()

    return df


def fetch_monthly_street_traffic(
    lat: float,
    lon: float,
    start_date: date | datetime,
    end_date: date | datetime,
    radius_m: int = 100,
) -> pd.DataFrame:
    """
    Demo-implementatie voor straatdrukte op maandniveau.

    Voor nu:
    - Negeert lat/lon/radius (één generieke straatlocatie).
    - Gebruikt pathzz_sample_weekly.csv.
    - Filtert alle weeks die overlappen met [start_date, end_date].
    - Berekent een 'mid_date' per week en groepeert naar maand.
    - Returnt DataFrame met:
        - month (Timestamp, begin van maand)
        - street_footfall (som van Visits in die maand)
    """

    df = _load_pathzz_sample()
    if df.empty:
        return pd.DataFrame(columns=["month", "street_footfall"])

    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()

    # Hou alleen weeks die overlappen met het gevraagde interval
    mask = (df["week_end"] >= start) & (df["week_start"] <= end)
    df_sel = df[mask].copy()
    if df_sel.empty:
        return pd.DataFrame(columns=["month", "street_footfall"])

    # Representatieve datum per week = midden van het bereik
    df_sel["mid_date"] = df_sel["week_start"] + (
        df_sel["week_end"] - df_sel["week_start"]
    ) / 2

    # Groepeer naar maand
    df_sel["month"] = (
        pd.to_datetime(df_sel["mid_date"])
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    monthly = (
        df_sel.groupby("month", as_index=False)["Visits"]
        .sum()
        .rename(columns={"Visits": "street_footfall"})
    )

    return monthly
