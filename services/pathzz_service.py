# services/pathzz_service.py

from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd
from functools import lru_cache


# Pad naar de demo-file in /data
SAMPLE_FILE = Path(__file__).resolve().parent.parent / "data" / "pathzz_sample_weekly.csv"


@lru_cache(maxsize=1)
def _load_sample_pathzz() -> pd.DataFrame:
    """
    Laadt de demo Pathzz-data uit pathzz_sample_weekly.csv.

    Verwacht formaat:
        Week;Visits
        2025-10-26 To 2025-11-01;30.332

    - 'Week' wordt gesplitst in week_start en week_end
    - 'Visits' wordt geïnterpreteerd als voetgangersvolume
      (notatie 16.725 -> 16725)
    """
    if not SAMPLE_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(SAMPLE_FILE, sep=";")

    if df.empty:
        return df

    # 16.725 -> 16725 (punt als duizendtalscheiding)
    df["Visits"] = (
        df["Visits"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .astype(float)
    )

    # "2025-10-26 To 2025-11-01" -> start & end
    def parse_start(s: str):
        return pd.to_datetime(s.split(" To ")[0])

    def parse_end(s: str):
        return pd.to_datetime(s.split(" To ")[1])

    df["week_start"] = df["Week"].apply(parse_start)
    df["week_end"] = df["Week"].apply(parse_end)

    # Koppel de week aan een maand op basis van week_start
    df["month"] = df["week_start"].dt.to_period("M").dt.to_timestamp()

    return df


def fetch_monthly_street_traffic(
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    radius_m: int = 100,
) -> pd.DataFrame:
    """
    Demo-implementatie voor straatdrukte (Pathzz).

    In plaats van een echte API:
    - laadt pathzz_sample_weekly.csv
    - filtert op weeks die overlappen met [start_date, end_date]
    - aggregeert naar maandniveau met kolommen:
        ['month', 'street_footfall']

    lat / lon / radius_m zijn nu alleen voor toekomstige uitbreiding;
    voor deze pilot worden ze genegeerd.
    """
    df = _load_sample_pathzz()
    if df.empty:
        return df

    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        df = df[df["week_end"] >= start_date]

    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        df = df[df["week_start"] <= end_date]

    if df.empty:
        return pd.DataFrame(columns=["month", "street_footfall"])

    monthly = (
        df.groupby("month", as_index=False)
        .agg(street_footfall=("Visits", "sum"))
        .sort_values("month")
        .reset_index(drop=True)
    )

    return monthly
