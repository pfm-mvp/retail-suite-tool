# services/event_service.py

from __future__ import annotations

import datetime as dt
from typing import Set

import pandas as pd


# -----------------------------------------------------------
# Basis: Pasen / NL-feestdagen / Black Friday
# -----------------------------------------------------------

def easter_date(year: int) -> dt.date:
    """
    Berekent paaszondag (Gregoriaanse kalender).
    Meeus/Jones/Butcher algoritme.
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


def nl_holidays_for_year(year: int) -> Set[dt.date]:
    """
    Basis-set met NL-feestdagen voor retail-analyses.
    Niet 100% juridisch compleet – wél praktisch als model-feature.
    """
    holidays: Set[dt.date] = set()

    # Vaste dagen
    holidays.add(dt.date(year, 1, 1))    # Nieuwjaarsdag
    holidays.add(dt.date(year, 4, 27))   # Koningsdag
    holidays.add(dt.date(year, 5, 5))    # Bevrijdingsdag
    holidays.add(dt.date(year, 12, 25))  # 1e Kerstdag
    holidays.add(dt.date(year, 12, 26))  # 2e Kerstdag

    # Paas- / pinksterreeks
    easter = easter_date(year)
    good_friday = easter - dt.timedelta(days=2)
    easter_monday = easter + dt.timedelta(days=1)
    ascension = easter + dt.timedelta(days=39)
    pentecost = easter + dt.timedelta(days=49)
    pentecost_monday = easter + dt.timedelta(days=50)

    holidays.update(
        {
            good_friday,
            easter,
            easter_monday,
            ascension,
            pentecost,
            pentecost_monday,
        }
    )

    return holidays


def black_friday_date(year: int) -> dt.date:
    """
    Black Friday = laatste vrijdag van november.
    """
    last_nov = dt.date(year, 11, 30)
    while last_nov.weekday() != 4:  # 0=maandag, 4=vrijdag
        last_nov -= dt.timedelta(days=1)
    return last_nov


# -----------------------------------------------------------
# DataFrame helper: retail-kalenderfeatures
# -----------------------------------------------------------

def add_retail_calendar_features(
    df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Voegt retail-kalenderfeatures toe op basis van een datumbepaalde kolom.

    Features:
    - dow                (0=ma, ..., 6=zo)
    - month
    - day_of_month
    - is_weekend
    - is_q4              (okt–dec)
    - is_december_peak   (december)
    - is_summer_holiday  (juli/aug)
    - is_nl_holiday      (NL-feestdag)
    - is_christmas_period (15–31 december)
    - is_black_friday_weekend (vr/za/zo rond Black Friday)
    """
    if df is None or df.empty or date_col not in df.columns:
        return df

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])

    if out.empty:
        return out

    d = out[date_col]

    out["dow"] = d.dt.weekday
    out["month"] = d.dt.month
    out["day_of_month"] = d.dt.day
    out["is_weekend"] = out["dow"].isin([5, 6]).astype(int)

    out["is_q4"] = out["month"].isin([10, 11, 12]).astype(int)
    out["is_december_peak"] = (out["month"] == 12).astype(int)
    out["is_summer_holiday"] = out["month"].isin([7, 8]).astype(int)

    years = d.dt.year.unique().tolist()

    # NL-feestdagen
    holiday_dates: Set[dt.date] = set()
    for y in years:
        holiday_dates |= nl_holidays_for_year(int(y))

    out["is_nl_holiday"] = d.dt.date.isin(holiday_dates).astype(int)

    # Kerstperiode 15–31 dec
    out["is_christmas_period"] = (
        (out["month"] == 12) & (out["day_of_month"] >= 15)
    ).astype(int)

    # Black Friday-weekend (vr–zo)
    bf_weekend_dates: Set[dt.date] = set()
    for y in years:
        bf = black_friday_date(int(y))
        for offset in range(0, 3):
            bf_weekend_dates.add(bf + dt.timedelta(days=offset))

    out["is_black_friday_weekend"] = d.dt.date.isin(bf_weekend_dates).astype(int)

    return out


# -----------------------------------------------------------
# Kleine helper voor labels (optioneel, voor UI)
# -----------------------------------------------------------

def describe_retail_event(date_value: dt.date) -> str:
    """
    Geeft een korte beschrijving van een event/kalendermoment.
    Handig voor tooltips / uitleg in dashboards.
    """
    if isinstance(date_value, dt.datetime):
        date_value = date_value.date()

    y = date_value.year
    holidays = nl_holidays_for_year(y)

    if date_value in holidays:
        # Heel eenvoudig: alleen "Feestdag"
        # (uitbreiden met naam per datum kan altijd later)
        return "Feestdag (NL)"

    bf = black_friday_date(y)
    if date_value == bf:
        return "Black Friday"
    if date_value == bf + dt.timedelta(days=1):
        return "Black Friday weekend (zaterdag)"
    if date_value == bf + dt.timedelta(days=2):
        return "Black Friday weekend (zondag)"

    if date_value.month == 12 and date_value.day >= 15:
        return "Kerstperiode"

    if date_value.month in (7, 8):
        return "Zomerperiode"

    return ""
