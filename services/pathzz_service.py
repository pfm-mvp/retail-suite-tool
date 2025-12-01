# services/pathzz_service.py

import pandas as pd
from pathlib import Path
from functools import lru_cache

@lru_cache(maxsize=1)
def load_pathzz_weekly() -> pd.DataFrame:
    """
    Laadt wekelijkse Pathzz demo data.

    CSV format:
    Week;Visits
    2025-11-23 To 2025-11-29;35.000

    Visits is al in 'echte' waarde (35.000 bezoekers per week).
    We converteren de Europese notatie naar integer.
    """

    csv_path = Path("data") / "pathzz_sample_weekly.csv"
    if not csv_path.exists():
        return pd.DataFrame(columns=["week_start", "street_footfall"])

    df = pd.read_csv(csv_path, sep=";")

    # Datumparsing
    df["week_start"] = df["Week"].str.split(" To ").str[0]
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")

    # Europese duizendtallen naar int (16.725 → 16725)
    df["Visits"] = (
        df["Visits"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .astype(int)
    )

    df = df.dropna(subset=["week_start"])
    df = df.rename(columns={"Visits": "street_footfall"})

    return df[["week_start", "street_footfall"]]


def fetch_weekly_street_traffic(start_date, end_date) -> pd.DataFrame:
    """
    Filter weekly Pathzz data op geselecteerde periode.
    Retourneert DataFrame:

    week_start | street_footfall
    """
    df = load_pathzz_weekly()

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    mask = (df["week_start"] >= start) & (df["week_start"] <= end)
    df_sel = df.loc[mask].copy()

    return df_sel
