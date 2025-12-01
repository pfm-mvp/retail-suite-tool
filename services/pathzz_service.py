# services/pathzz_service.py

import pandas as pd
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def _load_pathzz_weekly() -> pd.DataFrame:
    """
    Laadt de demo-Pathzz data uit data/pathzz_sample_weekly.csv.

    Verwacht structuur:
    - kolom 'Week': 'YYYY-MM-DD To YYYY-MM-DD'
    - kolom 'Visits': waarden zoals 16.725 / 20.000 / 35.000

    We interpreteren 'Visits' als duizendtallen (16.725 => 16.725 bezoekers).
    Omdat het CSV met een punt werkt, leest pandas dit als 16.725 (float).
    Daarom schalen we alles x 1000.
    """

    csv_path = Path("data") / "pathzz_sample_weekly.csv"
    if not csv_path.exists():
        # Geen bestand → lege DF terug
        return pd.DataFrame(columns=["week_start", "Visits"])

    df = pd.read_csv(csv_path, sep=";")

    # Weekstart eruit trekken: eerste datum vóór " To "
    df["week_start"] = (
        df["Week"]
        .astype(str)
        .str.split(" To ")
        .str[0]
    )
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")

    # Visits naar numeriek
    df["Visits"] = pd.to_numeric(df["Visits"], errors="coerce")

    # Demo-aanname: waarden zijn in duizendtallen (10–40)
    # → schaal x 1000 naar "aantal passanten"
    if df["Visits"].max() is not None and df["Visits"].max() < 1000:
        df["Visits"] = df["Visits"] * 1000

    df = df.dropna(subset=["week_start", "Visits"])
    return df[["week_start", "Visits"]]


def fetch_monthly_street_traffic(
    start_date,
    end_date,
) -> pd.DataFrame:
    """
    Maakt van de wekelijkse Pathzz-sample data een maandelijkse time series:

    Input:
    - start_date, end_date: datums (date/datetime/str)

    Output:
    - DataFrame met kolommen:
      - 'month' (Timestamp, eerste dag van maand)
      - 'street_footfall' (som van Visits in die maand)
    """

    df = _load_pathzz_weekly()
    if df.empty:
        return pd.DataFrame(columns=["month", "street_footfall"])

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    mask = (df["week_start"] >= start) & (df["week_start"] <= end)
    df_sel = df.loc[mask].copy()
    if df_sel.empty:
        return pd.DataFrame(columns=["month", "street_footfall"])

    df_sel["month"] = df_sel["week_start"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df_sel
        .groupby("month", as_index=False)["Visits"]
        .sum()
        .rename(columns={"Visits": "street_footfall"})
    )

    return monthly
