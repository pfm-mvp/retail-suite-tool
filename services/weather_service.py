# services/forecast_service.py

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Pro-model support (LightGBM). Als dit niet geïnstalleerd is,
# valt de pro-forecast automatisch terug op de simpele variant.
try:
    import lightgbm as lgb  # type: ignore
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False

import requests
import streamlit as st

# Visual Crossing API key uit Streamlit secrets
VISUALCROSSING_KEY = st.secrets.get("visualcrossing_key", None)


# -----------------------------------------------------------
# HELPER: sales_per_visitor
# -----------------------------------------------------------

def _ensure_sales_per_visitor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zorgt dat 'sales_per_visitor' bestaat als 'turnover' en 'footfall' aanwezig zijn.
    Past df in-place NIET aan; returned een copy.
    """
    df = df.copy()
    if "sales_per_visitor" not in df.columns:
        if "turnover" in df.columns and "footfall" in df.columns:
            df["sales_per_visitor"] = np.where(
                df["footfall"] > 0,
                df["turnover"] / df["footfall"],
                np.nan,
            )
        else:
            df["sales_per_visitor"] = np.nan
    return df


# -----------------------------------------------------------
# HELPER: NL-feestdagen & kalenderfeatures
# -----------------------------------------------------------

def _easter_date(year: int) -> dt.date:
    """
    Berekent paaszondag (Gregoriaanse kalender).
    Standaard algoritme (Meeus/Jones/Butcher).
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


def _nl_holidays_for_year(year: int) -> set[dt.date]:
    """
    Basis-set met NL-feestdagen voor retail (indicatief).
    Dit zijn 'signaal'-dagen die als feature in het model komen.
    """
    holidays: set[dt.date] = set()

    # Vaste dagen
    holidays.add(dt.date(year, 1, 1))   # Nieuwjaarsdag
    holidays.add(dt.date(year, 4, 27))  # Koningsdag
    holidays.add(dt.date(year, 5, 5))   # Bevrijdingsdag (we nemen hem elk jaar mee)
    holidays.add(dt.date(year, 12, 25))  # 1e Kerstdag
    holidays.add(dt.date(year, 12, 26))  # 2e Kerstdag

    # Paas- en pinkster-reeks
    easter = _easter_date(year)
    good_friday = easter - dt.timedelta(days=2)
    easter_monday = easter + dt.timedelta(days=1)
    ascension = easter + dt.timedelta(days=39)
    pentecost = easter + dt.timedelta(days=49)      # Pinksterzondag
    pentecost_monday = easter + dt.timedelta(days=50)

    holidays.update(
        {
            good_friday,
            easter,             # Paaszondag
            easter_monday,      # Paasmaandag
            ascension,          # Hemelvaart
            pentecost,          # Pinksterzondag
            pentecost_monday,   # Pinkstermaandag
        }
    )

    return holidays


def _black_friday_date(year: int) -> dt.date:
    """
    Black Friday = laatste vrijdag van november.
    """
    last_nov = dt.date(year, 11, 30)
    while last_nov.weekday() != 4:  # 0=maandag, 4=vrijdag
        last_nov -= dt.timedelta(days=1)
    return last_nov


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kalenderfeatures voor elke rij op basis van 'date':

    - dow, month, day_of_month, is_weekend
    - is_q4
    - is_december_peak
    - is_summer_holiday (juli/aug)
    - is_nl_holiday (NL kalender)
    - is_christmas_period (15 dec – 31 dec)
    - is_black_friday_weekend (vr/za/zo Black Friday-weekend)
    """
    if df is None or df.empty or "date" not in df.columns:
        return df

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])

    if out.empty:
        return out

    out["dow"] = out["date"].dt.weekday
    out["month"] = out["date"].dt.month
    out["day_of_month"] = out["date"].dt.day
    out["is_weekend"] = out["dow"].isin([5, 6]).astype(int)

    out["is_q4"] = out["month"].isin([10, 11, 12]).astype(int)
    out["is_december_peak"] = (out["month"] == 12).astype(int)
    out["is_summer_holiday"] = out["month"].isin([7, 8]).astype(int)

    years = out["date"].dt.year.unique().tolist()

    # NL-holidays
    holiday_dates: set[dt.date] = set()
    for y in years:
        holiday_dates |= _nl_holidays_for_year(int(y))

    out["is_nl_holiday"] = out["date"].dt.date.isin(holiday_dates).astype(int)

    # Kerst- / eindejaarsperiode (15–31 dec)
    out["is_christmas_period"] = (
        (out["month"] == 12) & (out["day_of_month"] >= 15)
    ).astype(int)

    # Black Friday weekend (vr-zo)
    bf_weekend_dates: set[dt.date] = set()
    for y in years:
        bf = _black_friday_date(int(y))
        for offset in range(0, 3):  # vr, za, zo
            bf_weekend_dates.add(bf + dt.timedelta(days=offset))

    out["is_black_friday_weekend"] = out["date"].dt.date.isin(bf_weekend_dates).astype(int)

    return out


# -----------------------------------------------------------
# HELPER: Visual Crossing weather
# -----------------------------------------------------------

def _fetch_visualcrossing_history_and_forecast(
    location_str: str,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    """
    Haalt weerdata (historisch + forecast) op bij Visual Crossing
    voor een gegeven locatie en datumbereik.

    Return DataFrame met kolommen: date, temp, precip, windspeed
    of een lege DataFrame bij fout.
    """
    if not VISUALCROSSING_KEY:
        return pd.DataFrame()

    start = start_date.strftime("%Y-%m-%d")
    end = end_date.strftime("%Y-%m-%d")

    url = (
        "https://weather.visualcrossing.com/VisualCrossingWebServices/"
        f"rest/services/timeline/{location_str}/{start}/{end}"
    )
    params = {
        "unitGroup": "metric",
        "key": VISUALCROSSING_KEY,
        "include": "days",
    }

    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return pd.DataFrame()

    if "days" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["days"])
    df["date"] = pd.to_datetime(df["datetime"])
    # temp: gemiddelde dagtemp; precip: mm; windspeed: km/u (VC)
    return df[["date", "temp", "precip", "windspeed"]]


# -----------------------------------------------------------
# SIMPLE BASELINE FORECAST (DoW-gemiddelden)
# -----------------------------------------------------------

def build_simple_footfall_turnover_forecast(
    df_all_raw: pd.DataFrame,
    horizon: int = 14,
    min_history_days: int = 30,
) -> dict:
    """
    Simpele day-of-week forecast voor footfall & omzet.

    Verwacht:
    - df_all_raw met minimaal kolommen: 'date', 'footfall'
    - optioneel: 'turnover'
    - 'sales_per_visitor' wordt berekend als die nog niet bestaat

    Return dict:
    {
        "enough_history": bool,
        "hist_recent": DataFrame,
        "forecast": DataFrame,
        "recent_footfall_sum": float,
        "recent_turnover_sum": float or NaN,
        "forecast_footfall_sum": float,
        "forecast_turnover_sum": float,
        "last_date": Timestamp,
        "model_type": str,               # "simple_dow"
        "used_simple_fallback": bool,    # altijd False hier
    }
    """

    df = df_all_raw.copy()

    if df.empty or "date" not in df.columns or "footfall" not in df.columns:
        return {"enough_history": False}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if df.empty:
        return {"enough_history": False}

    df = _ensure_sales_per_visitor(df)

    unique_days = df["date"].dt.normalize().nunique()
    if unique_days < min_history_days:
        return {"enough_history": False}

    df["dow"] = df["date"].dt.weekday  # 0=maandag
    grp = df.groupby("dow")

    dow_stats = grp["footfall"].agg(["mean", "std"]).rename(
        columns={"mean": "footfall_mean", "std": "footfall_std"}
    )

    if "sales_per_visitor" in df.columns:
        dow_spv = grp["sales_per_visitor"].mean().rename("spv_mean")
        dow_stats = dow_stats.join(dow_spv)
    else:
        dow_stats["spv_mean"] = np.nan

    last_date = df["date"].max()

    # Forecast horizon
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    fc = pd.DataFrame({"date": future_dates})
    fc["dow"] = fc["date"].dt.weekday

    fc = fc.merge(dow_stats.reset_index(), on="dow", how="left")

    global_spv = df["sales_per_visitor"].mean()
    fc["spv_mean"] = fc["spv_mean"].fillna(global_spv)

    fc["footfall_forecast"] = fc["footfall_mean"].clip(lower=0)
    fc["turnover_forecast"] = fc["footfall_forecast"] * fc["spv_mean"]

    last_14_start = last_date - pd.Timedelta(days=13)
    recent = df[(df["date"] >= last_14_start) & (df["date"] <= last_date)].copy()

    recent_foot = recent["footfall"].sum()
    recent_turn = recent["turnover"].sum() if "turnover" in recent.columns else np.nan

    fut_foot = fc["footfall_forecast"].sum()
    fut_turn = fc["turnover_forecast"].sum()

    hist_recent = df[df["date"] >= (last_date - pd.Timedelta(days=27))].copy()

    return {
        "enough_history": True,
        "hist_recent": hist_recent,
        "forecast": fc,
        "recent_footfall_sum": float(recent_foot),
        "recent_turnover_sum": float(recent_turn) if not pd.isna(recent_turn) else np.nan,
        "forecast_footfall_sum": float(fut_foot),
        "forecast_turnover_sum": float(fut_turn),
        "last_date": last_date,
        "model_type": "simple_dow",
        "used_simple_fallback": False,
    }


# -----------------------------------------------------------
# PRO FORECAST (LightGBM + lags/rolling + kalender + weer)
# -----------------------------------------------------------

def build_pro_footfall_turnover_forecast(
    df_all_raw: pd.DataFrame,
    horizon: int = 14,
    min_history_days: int = 60,
    weather_cfg: Optional[Dict[str, Any]] = None,
    use_weather: bool = False,
    **_: Any,
) -> dict:
    """
    Pro-forecast met:
    - LightGBM (als beschikbaar)
    - Features: dow, maand, dag, weekend, kalenderflags (NL-holidays, kerst, BF, zomer),
      lags & rolling means
    - Optioneel: weerfeatures via Visual Crossing (temp, neerslag, wind)

    Als LightGBM niet beschikbaar is of er te weinig historie is:
    -> automatische fallback naar build_simple_footfall_turnover_forecast(...),
       met 'used_simple_fallback' = True.

    Return: zelfde structuur als simple-forecast-result, plus:
        - "model_type": "lightgbm_dow_lags_calendar_weather" of "simple_dow"
        - "used_simple_fallback": bool
    """

    # Geen LightGBM? → fallback
    if not HAS_LIGHTGBM:
        simple = build_simple_footfall_turnover_forecast(
            df_all_raw,
            horizon=horizon,
            min_history_days=min_history_days,
        )
        simple["model_type"] = "simple_dow"
        simple["used_simple_fallback"] = True
        return simple

    df = df_all_raw.copy()

    if df.empty or "date" not in df.columns or "footfall" not in df.columns:
        simple = build_simple_footfall_turnover_forecast(
            df_all_raw,
            horizon=horizon,
            min_history_days=min_history_days,
        )
        simple["used_simple_fallback"] = True
        return simple

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        simple = build_simple_footfall_turnover_forecast(
            df_all_raw,
            horizon=horizon,
            min_history_days=min_history_days,
        )
        simple["used_simple_fallback"] = True
        return simple

    df = _ensure_sales_per_visitor(df)

    unique_days = df["date"].dt.normalize().nunique()
    if unique_days < min_history_days:
        simple = build_simple_footfall_turnover_forecast(
            df_all_raw,
            horizon=horizon,
            min_history_days=min_history_days,
        )
        simple["used_simple_fallback"] = True
        return simple

    # Sorteren en kalenderfeatures toevoegen
    df = df.sort_values("date").reset_index(drop=True)
    df["seq"] = np.arange(len(df))
    df = _add_calendar_features(df)

    # Lags & rolling stats op footfall
    for lag in [1, 7, 14]:
        df[f"footfall_lag_{lag}"] = df["footfall"].shift(lag)

    df["footfall_roll_7"] = df["footfall"].rolling(window=7, min_periods=3).mean()
    df["footfall_roll_28"] = df["footfall"].rolling(window=28, min_periods=7).mean()

    # Optioneel: weerfeatures ophalen & mergen
    weather_df = pd.DataFrame()
    if use_weather and weather_cfg and VISUALCROSSING_KEY:
        try:
            location = (
                weather_cfg.get("location")
                or f"{weather_cfg.get('city', '')},{weather_cfg.get('country', '')}"
            )
            location = str(location).strip()
            if location:
                start_date = df["date"].min().date()
                last_hist_date = df["date"].max().date()
                end_date = last_hist_date + dt.timedelta(days=horizon)
                weather_df = _fetch_visualcrossing_history_and_forecast(
                    location,
                    start_date,
                    end_date,
                )
        except Exception:
            weather_df = pd.DataFrame()

    if not weather_df.empty:
        weather_df = weather_df.copy()
        weather_df["date"] = pd.to_datetime(weather_df["date"])
        # Merge met historie
        df = df.merge(weather_df, on="date", how="left", suffixes=("", "_w"))
        # Consistente namen
        rename_map = {
            "temp": "w_temp",
            "precip": "w_precip",
            "windspeed": "w_wind",
        }
        df.rename(columns=rename_map, inplace=True)
        # Missing vullen met gemiddelde (zodat het model ze kan gebruiken)
        for col in ["w_temp", "w_precip", "w_wind"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
                df[col] = df[col].fillna(df[col].mean())
    else:
        # Als geen weerdata → geen weer features
        for col in ["w_temp", "w_precip", "w_wind"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    # Featurekolommen
    base_feature_cols = [
        "dow",
        "month",
        "day_of_month",
        "is_weekend",
        "is_q4",
        "is_december_peak",
        "is_summer_holiday",
        "is_nl_holiday",
        "is_christmas_period",
        "is_black_friday_weekend",
        "footfall_lag_1",
        "footfall_lag_7",
        "footfall_lag_14",
        "footfall_roll_7",
        "footfall_roll_28",
    ]

    feature_cols = base_feature_cols.copy()
    use_weather_features = False
    if use_weather and all(col in df.columns for col in ["w_temp", "w_precip", "w_wind"]):
        feature_cols += ["w_temp", "w_precip", "w_wind"]
        use_weather_features = True

    # Trainset
    df_model = df.dropna(subset=feature_cols + ["footfall"]).copy()
    if df_model.shape[0] < 30:
        simple = build_simple_footfall_turnover_forecast(
            df_all_raw,
            horizon=horizon,
            min_history_days=min_history_days,
        )
        simple["used_simple_fallback"] = True
        return simple

    X_train = df_model[feature_cols].values
    y_train = df_model["footfall"].values

    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Handige global means voor forecast
    global_spv = df["sales_per_visitor"].mean()
    global_w_temp = df["w_temp"].mean() if "w_temp" in df.columns else np.nan
    global_w_precip = df["w_precip"].mean() if "w_precip" in df.columns else np.nan
    global_w_wind = df["w_wind"].mean() if "w_wind" in df.columns else np.nan

    # Footfall historiek (lijst voor lags & rollings)
    foot_hist = df["footfall"].tolist()
    last_date = df["date"].max()

    # Kalender + eventueel weer voor toekomstige data
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    cal_future = pd.DataFrame({"date": future_dates})
    cal_future = _add_calendar_features(cal_future)

    if not weather_df.empty:
        # Gebruik dezelfde weather_df als bij historie-merge
        wf = weather_df.copy()
        wf["date"] = pd.to_datetime(wf["date"])
        wf.rename(
            columns={"temp": "w_temp", "precip": "w_precip", "windspeed": "w_wind"},
            inplace=True,
        )
        cal_future = cal_future.merge(wf, on="date", how="left")

    cal_future = cal_future.set_index("date")

    fc_rows = []

    for target_date in future_dates:
        # kalenderfeatures voor deze dag
        cal_row = cal_future.loc[target_date]

        # lags
        new_idx = len(foot_hist)

        def _lag_or_last(idx_offset: int) -> float:
            idx = new_idx - idx_offset
            if 0 <= idx < len(foot_hist):
                return float(foot_hist[idx])
            return float(foot_hist[0]) if len(foot_hist) > 0 else 0.0

        lag_1 = _lag_or_last(1)
        lag_7 = _lag_or_last(7)
        lag_14 = _lag_or_last(14)

        if len(foot_hist) > 0:
            roll_7 = float(np.mean(foot_hist[-7:])) if len(foot_hist) >= 3 else float(
                np.mean(foot_hist)
            )
            roll_28 = float(np.mean(foot_hist[-28:])) if len(foot_hist) >= 7 else float(
                np.mean(foot_hist)
            )
        else:
            roll_7 = 0.0
            roll_28 = 0.0

        # Basis-kalenderfeatures
        x_vals = [
            float(cal_row["dow"]),
            float(cal_row["month"]),
            float(cal_row["day_of_month"]),
            float(cal_row["is_weekend"]),
            float(cal_row["is_q4"]),
            float(cal_row["is_december_peak"]),
            float(cal_row["is_summer_holiday"]),
            float(cal_row["is_nl_holiday"]),
            float(cal_row["is_christmas_period"]),
            float(cal_row["is_black_friday_weekend"]),
            float(lag_1),
            float(lag_7),
            float(lag_14),
            float(roll_7),
            float(roll_28),
        ]

        # Eventueel weer
        if use_weather_features:
            def _safe_weather(col_name: str, global_val: float) -> float:
                val = float(cal_row[col_name]) if col_name in cal_row.index else np.nan
                if np.isnan(val):
                    if not np.isnan(global_val):
                        return float(global_val)
                    return 0.0
                return val

            x_vals.append(_safe_weather("w_temp", global_w_temp))
            x_vals.append(_safe_weather("w_precip", global_w_precip))
            x_vals.append(_safe_weather("w_wind", global_w_wind))

        y_pred = float(model.predict(np.array([x_vals]))[0])
        y_pred = max(y_pred, 0.0)

        foot_hist.append(y_pred)
        turnover_pred = y_pred * (global_spv if not np.isnan(global_spv) else 0.0)

        fc_rows.append(
            {
                "date": target_date,
                "footfall_forecast": y_pred,
                "turnover_forecast": turnover_pred,
            }
        )

    fc = pd.DataFrame(fc_rows)

    # Vergelijk met afgelopen 14 dagen
    last_14_start = last_date - pd.Timedelta(days=13)
    recent = df[(df["date"] >= last_14_start) & (df["date"] <= last_date)].copy()

    recent_foot = recent["footfall"].sum()
    recent_turn = recent["turnover"].sum() if "turnover" in recent.columns else np.nan

    fut_foot = fc["footfall_forecast"].sum()
    fut_turn = fc["turnover_forecast"].sum()

    hist_recent = df[df["date"] >= (last_date - pd.Timedelta(days=27))].copy()

    return {
        "enough_history": True,
        "hist_recent": hist_recent,
        "forecast": fc,
        "recent_footfall_sum": float(recent_foot),
        "recent_turnover_sum": float(recent_turn) if not pd.isna(recent_turn) else np.nan,
        "forecast_footfall_sum": float(fut_foot),
        "forecast_turnover_sum": float(fut_turn),
        "last_date": last_date,
        "model_type": "lightgbm_dow_lags_calendar_weather"
        if use_weather_features
        else "lightgbm_dow_lags_calendar",
        "used_simple_fallback": False,
    }
