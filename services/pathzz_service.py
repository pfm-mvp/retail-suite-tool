# services/pathzz_service.py
from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

# Pad naar de demo-file
DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "pathzz_sample_weekly.csv"


@lru_cache(maxsize=1)
def _load_pathzz_weekly() -> pd.DataFrame:
    """
    Laadt de demo Pathzz data uit pathzz_sample_weekly.csv.

    Verwacht kolommen:
      - Week: "YYYY-MM-DD To YYYY-MM-DD"
      - Visits: string met punt als duizendtalscheiding, bijv "16.725"
    """
    if not DATA_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(DATA_FILE, sep=";")

    if "Week" not in df.columns or "Visits" not in df.columns:
        return pd.DataFrame()

    # Visits "16.725" => 16725
    df["Visits"] = (
        df["Visits"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .astype("float")
    )

    # Week "2025-10-26 To 2025-11-01" → week_start = 2025-10-26
    def parse_week_start(s: str) -> datetime:
        start_str = str(s).split(" To ")[0].strip()
        return datetime.strptime(start_str, "%Y-%m-%d")

    df["week_start"] = df["Week"].apply(parse_week_start)
    df["month"] = df["week_start"].dt.to_period("M").dt.to_timestamp()

    df = df.rename(columns={"Visits": "street_footfall"})
    return df[["month", "week_start", "street_footfall"]]


def fetch_monthly_street_traffic(
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    radius_m: int = 100,
) -> pd.DataFrame:
    """
    Demo-implementatie van Pathzz.

    - Gebruikt pathzz_sample_weekly.csv
    - Negeert lat/lon/radius (maar laat ze in de signatuur zodat andere tools niet breken)
    - Geeft maandtotalen 'street_footfall' terug tussen start_date en end_date
    """
    weekly = _load_pathzz_weekly()
    if weekly.empty:
        return weekly

    if start_date is not None:
        weekly = weekly[weekly["week_start"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        weekly = weekly[weekly["week_start"] <= pd.to_datetime(end_date)]

    if weekly.empty:
        return pd.DataFrame(columns=["month", "street_footfall"])

    monthly = (
        weekly.groupby("month", as_index=False)["street_footfall"]
        .sum(min_count=1)
    )
    return monthly
