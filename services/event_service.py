# services/event_service.py

from __future__ import annotations
import datetime as dt
from typing import Iterable, List
import pandas as pd


def _normalize_dates(dates: Iterable) -> pd.Series:
    """Zorgt dat we een genormaliseerde datetime.date-serie hebben."""
    s = pd.to_datetime(pd.Series(list(dates)), errors="coerce").dt.normalize()
    return s


def _black_friday(year: int) -> dt.date:
    """
    Black Friday = 4e vrijdag van november.
    """
    # 1 november
    d = dt.date(year, 11, 1)
    # naar eerste vrijdag
    while d.weekday() != 4:
        d += dt.timedelta(days=1)
    # dan + 3 weken = 4e vrijdag
    d += dt.timedelta(weeks=3)
    return d


def _date_range(start: dt.date, end: dt.date) -> List[dt.date]:
    """Inclusief start en end."""
    days = (end - start).days
    return [start + dt.timedelta(days=i) for i in range(days + 1)]


def build_event_flags_for_dates(
    dates: Iterable,
    country: str = "NL",
) -> pd.DataFrame:
    """
    Geeft per datum simpele event-flags terug.
    Gebruik: features voor forecast + uitleg voor de storemanager.

    Output kolommen:
    - date (Timestamp, genormaliseerd)
    - is_weekend (0/1)
    - is_national_holiday (0/1)
    - is_school_holiday (0/1)  [grof benaderd]
    - is_black_friday (0/1)
    - is_december_trade (0/1)  [Sinterklaas/Kerstdrukte]
    - is_summer_sale (0/1)     [juli/august]
    """
    s = _normalize_dates(dates)
    if s.empty:
        return pd.DataFrame(columns=[
            "date",
            "is_weekend",
            "is_national_holiday",
            "is_school_holiday",
            "is_black_friday",
            "is_december_trade",
            "is_summer_sale",
        ])

    df = pd.DataFrame({"date": s})
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"]).reset_index(drop=True)
    if df.empty:
        return df

    df["date_only"] = df["date"].dt.date
    df["year"] = df["date"].dt.year

    # Basisflag: weekend
    df["is_weekend"] = df["date"].dt.weekday.isin([5, 6]).astype(int)

    # --- Nationale feestdagen & vakanties – NL (grof benaderd demo) ---
    nat_holidays = set()
    school_holidays = set()
    summer_sale = set()
    december_trade = set()
    black_fridays = set()

    years = sorted(df["year"].unique().tolist())
    for y in years:
        # Black Friday
        bf = _black_friday(y)
        black_fridays.add(bf)

        # Grof: zomer-vakantie NL: 15 juli – 31 augustus
        summer_hol = _date_range(dt.date(y, 7, 15), dt.date(y, 8, 31))
        school_holidays.update(summer_hol)
        summer_sale.update(summer_hol)

        # Kerstvakantie globaal: 24 dec – 1 jan (van y en y+1)
        xmas_start = dt.date(y, 12, 24)
        xmas_end = dt.date(y + 1, 1, 1)
        school_holidays.update(_date_range(xmas_start, xmas_end))

        # Voorjaarsvakantie: 15 feb – 1 maart
        spring_start = dt.date(y, 2, 15)
        spring_end = dt.date(y, 3, 1)
        school_holidays.update(_date_range(spring_start, spring_end))

        # December trade-peak: 1 dec – 31 dec
        december_trade.update(_date_range(dt.date(y, 12, 1), dt.date(y, 12, 31)))

        # Nationale feestdagen (NL, +/−):
        for d in [
            dt.date(y, 1, 1),   # Nieuwjaar
            dt.date(y, 4, 27),  # Koningsdag
            dt.date(y, 12, 25), # 1e Kerstdag
            dt.date(y, 12, 26), # 2e Kerstdag
        ]:
            nat_holidays.add(d)

    # Flags mappen naar df
    df["is_national_holiday"] = df["date_only"].apply(lambda d: int(d in nat_holidays))
    df["is_school_holiday"] = df["date_only"].apply(lambda d: int(d in school_holidays))
    df["is_black_friday"] = df["date_only"].apply(lambda d: int(d in black_fridays))
    df["is_december_trade"] = df["date_only"].apply(lambda d: int(d in december_trade))
    df["is_summer_sale"] = df["date_only"].apply(lambda d: int(d in summer_sale))

    # Opruimen
    df = df.drop(columns=["date_only", "year"])

    cols = [
        "date",
        "is_weekend",
        "is_national_holiday",
        "is_school_holiday",
        "is_black_friday",
        "is_december_trade",
        "is_summer_sale",
    ]
    return df[cols].copy()
