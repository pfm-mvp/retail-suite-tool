# services/event_service.py

import pandas as pd
from datetime import date, timedelta


def _to_date_series(dates):
    """
    Zorgt dat we altijd een nette pandas Series met datums hebben.
    Input kan zijn: list, DatetimeIndex, Series, etc.
    """
    if isinstance(dates, (pd.DatetimeIndex, pd.Series)):
        s = pd.to_datetime(dates)
    else:
        s = pd.to_datetime(pd.Series(dates))

    s = s.dropna().reset_index(drop=True)
    return s


def _get_black_friday(year: int) -> date:
    """
    Black Friday = laatste vrijdag van november.
    Simpele benadering, maar goed genoeg voor retail-usecases.
    """
    d = date(year, 11, 30)
    while d.weekday() != 4:  # 0=ma, 4=vr
        d = d - timedelta(days=1)
    return d


def build_event_flags_for_dates(dates, country: str = "NL") -> pd.DataFrame:
    """
    Bouwt simpele event-flags voor een reeks datums.

    Input:
        dates: iterabel met datums (str, datetime, etc.)
        country: future-proof (nu alleen NL-logic)

    Output DataFrame met kolommen:
    - date (datetime64[ns])
    - is_december_trade (0/1)
    - is_summer_sale (0/1)
    - is_black_friday (0/1)
    - is_school_holiday (0/1, grove benadering)

    Logica (NL, demo-niveau):
    - december = volledige handels-/feestmaand
    - summer_sale = juni / juli / augustus
    - black_friday = laatste vrijdag van november
    - schoolvakanties:
        * meivakantie: ~25 april – 10 mei
        * zomervakantie: ~10 juli – 31 augustus
    """
    s = _to_date_series(dates)
    if s.empty:
        return pd.DataFrame(
            {
                "date": [],
                "is_december_trade": [],
                "is_summer_sale": [],
                "is_black_friday": [],
                "is_school_holiday": [],
            }
        )

    df = pd.DataFrame({"date": s})
    df["date_only"] = df["date"].dt.date
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    # December handelsmaand
    df["is_december_trade"] = (df["month"] == 12).astype(int)

    # Summer sale: juni/juli/augustus
    df["is_summer_sale"] = df["month"].isin([6, 7, 8]).astype(int)

    # Black Friday per jaar
    bf_map = {y: _get_black_friday(y) for y in df["year"].unique()}
    df["is_black_friday"] = df.apply(
        lambda r: 1 if r["date_only"] == bf_map.get(r["year"]) else 0,
        axis=1,
    )

    # Schoolvakanties (NL, grove benaderingen)
    def _is_mei_holiday(d: date) -> bool:
        if d.month == 4 and d.day >= 25:
            return True
        if d.month == 5 and d.day <= 10:
            return True
        return False

    def _is_summer_holiday(d: date) -> bool:
        if d.month == 7 and d.day >= 10:
            return True
        if d.month == 8:
            return True
        return False

    def _is_school_holiday(d: date) -> bool:
        return _is_mei_holiday(d) or _is_summer_holiday(d)

    df["is_school_holiday"] = df["date_only"].apply(
        lambda d: 1 if _is_school_holiday(d) else 0
    )

    return df[
        [
            "date",
            "is_december_trade",
            "is_summer_sale",
            "is_black_friday",
            "is_school_holiday",
        ]
    ].reset_index(drop=True)
