# services/forecast_service.py

import numpy as np
import pandas as pd


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
    }
    """

    df = df_all_raw.copy()

    if df.empty or "date" not in df.columns or "footfall" not in df.columns:
        return {"enough_history": False}

    # Zorg dat date datetime is
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if df.empty:
        return {"enough_history": False}

    # Sales per visitor berekenen als dat nog niet bestaat
    if "sales_per_visitor" not in df.columns:
        if "turnover" in df.columns:
            df["sales_per_visitor"] = np.where(
                df["footfall"] > 0,
                df["turnover"] / df["footfall"],
                np.nan,
            )
        else:
            df["sales_per_visitor"] = np.nan

    # Minimaal aantal dagen historie
    unique_days = df["date"].dt.normalize().nunique()
    if unique_days < min_history_days:
        return {"enough_history": False}

    # Day-of-week statistieken
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

    # Fallback SPV: globaal gemiddelde
    global_spv = df["sales_per_visitor"].mean()
    fc["spv_mean"] = fc["spv_mean"].fillna(global_spv)

    # Footfall & omzet forecast
    fc["footfall_forecast"] = fc["footfall_mean"].clip(lower=0)
    fc["turnover_forecast"] = fc["footfall_forecast"] * fc["spv_mean"]

    # Vergelijk met afgelopen 14 dagen
    last_14_start = last_date - pd.Timedelta(days=13)
    recent = df[(df["date"] >= last_14_start) & (df["date"] <= last_date)].copy()

    recent_foot = recent["footfall"].sum()
    recent_turn = recent["turnover"].sum() if "turnover" in recent.columns else np.nan

    fut_foot = fc["footfall_forecast"].sum()
    fut_turn = fc["turnover_forecast"].sum()

    # Historiek voor grafiek: laatste 28 dagen
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
    }
