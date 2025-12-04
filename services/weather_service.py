# weather_service.py — Visual Crossing thin client (+ backwards compatible wrapper)
from __future__ import annotations

import os
import datetime as dt
from typing import Optional, Literal, List, Dict, Any

import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------
# Config & helpers
# --------------------------------------------------

VC_BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
LocationMode = Literal["latlon", "city_country", "postal_country"]


def _get_api_key() -> str:
    """
    Haalt de Visual Crossing API key op uit Streamlit secrets.
    Fallback: environment variable VISUALCROSSING_KEY (handig voor lokale tests).
    """
    if "visualcrossing_key" in st.secrets:
        return st.secrets["visualcrossing_key"]

    key = os.getenv("VISUALCROSSING_KEY")
    if not key:
        raise RuntimeError(
            "Visual Crossing API key niet gevonden. "
            "Zet 'visualcrossing_key' in st.secrets of VISUALCROSSING_KEY als env var."
        )
    return key


def build_location_query(
    mode: LocationMode,
    *,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    city: Optional[str] = None,
    country: Optional[str] = None,
    postal_code: Optional[str] = None,
) -> str:
    """
    Bouwt de location-string voor Visual Crossing.
    - latlon:         'lat,lon'
    - city_country:   'Amsterdam,Netherlands'
    - postal_country: '1102 DB,Netherlands'
    """
    if mode == "latlon":
        if lat is None or lon is None:
            raise ValueError("Voor mode 'latlon' zijn lat en lon verplicht.")
        return f"{lat},{lon}"

    if mode == "city_country":
        if not city or not country:
            raise ValueError("Voor mode 'city_country' zijn city én country verplicht.")
        return f"{city},{country}"

    if mode == "postal_country":
        if not postal_code or not country:
            raise ValueError("Voor mode 'postal_country' zijn postal_code én country verplicht.")
        return f"{postal_code},{country}"

    raise ValueError(f"Onbekende location mode: {mode}")


def _fetch_timeline(
    location: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include: str = "days",
    unit_group: str = "metric",
) -> dict:
    """
    Algemene wrapper voor de Visual Crossing timeline endpoint.
    - location: 'lat,lon' of 'city,country' etc.
    - start_date / end_date: 'YYYY-MM-DD' (optioneel)
    """
    api_key = _get_api_key()

    if start_date and end_date:
        path = f"{location}/{start_date}/{end_date}"
    else:
        path = location

    url = f"{VC_BASE_URL}/{path}"

    params = {
        "key": api_key,
        "unitGroup": unit_group,
        "include": include,
        "contentType": "json",
    }

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------
# Publieke VC-functies: historisch & forecast
# --------------------------------------------------

def get_historical_weather_daily(
    mode: LocationMode,
    *,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    city: Optional[str] = None,
    country: Optional[str] = None,
    postal_code: Optional[str] = None,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Historische dagelijkse weersdata tussen start_date en end_date (incl.).
    Returned DataFrame met o.a.:
      - date (datetime.date)
      - temp (gemiddelde)
      - tempmax/tempmin
      - precip (mm)
      - precipprob (%)
      - windspeed
      - cloudcover
      - conditions (tekst)
      - icon
    """
    location = build_location_query(
        mode,
        lat=lat,
        lon=lon,
        city=city,
        country=country,
        postal_code=postal_code,
    )

    data = _fetch_timeline(
        location=location,
        start_date=start_date,
        end_date=end_date,
        include="days",
        unit_group="metric",
    )

    days = data.get("days", [])
    if not days:
        return pd.DataFrame()

    df = pd.DataFrame(days)

    if "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"]).dt.date
    else:
        df["date"] = pd.to_datetime(df["datetimeEpoch"], unit="s").dt.date

    keep_cols = [
        "date",
        "temp",
        "tempmax",
        "tempmin",
        "precip",
        "precipprob",
        "snow",
        "windspeed",
        "cloudcover",
        "conditions",
        "icon",
    ]
    existing_cols = [c for c in keep_cols if c in df.columns]
    df = df[existing_cols].copy()

    return df.sort_values("date").reset_index(drop=True)


def get_forecast_weather_daily(
    mode: LocationMode,
    *,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    city: Optional[str] = None,
    country: Optional[str] = None,
    postal_code: Optional[str] = None,
    days_ahead: int = 14,
) -> pd.DataFrame:
    """
    Dagelijkse forecast vanaf vandaag voor 'days_ahead' dagen.
    Zelfde kolomstructuur als get_historical_weather_daily().
    """
    location = build_location_query(
        mode,
        lat=lat,
        lon=lon,
        city=city,
        country=country,
        postal_code=postal_code,
    )

    data = _fetch_timeline(
        location=location,
        start_date=None,
        end_date=None,
        include="days",
        unit_group="metric",
    )

    days = data.get("days", [])
    if not days:
        return pd.DataFrame()

    df = pd.DataFrame(days)

    if "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"]).dt.date
    else:
        df["date"] = pd.to_datetime(df["datetimeEpoch"], unit="s").dt.date

    keep_cols = [
        "date",
        "temp",
        "tempmax",
        "tempmin",
        "precip",
        "precipprob",
        "snow",
        "windspeed",
        "cloudcover",
        "conditions",
        "icon",
    ]
    existing_cols = [c for c in keep_cols if c in df.columns]
    df = df[existing_cols].copy()

    today = dt.date.today()
    df = df[df["date"] >= today].sort_values("date").head(days_ahead)

    return df.reset_index(drop=True)


# --------------------------------------------------
# Backwards compatibility: oude OpenWeather-signature
# --------------------------------------------------

def get_daily_forecast(
    lat: float,
    lon: float,
    api_key: str | None = None,
    days_ahead: int = 7,
) -> List[Dict[str, Any]]:
    """
    Backwards compatible wrapper voor oude OpenWeather get_daily_forecast.
    Signature blijft hetzelfde, maar data komt nu uit Visual Crossing.

    Output-structuur:
      - date (YYYY-MM-DD string)
      - temp (gem. temp)
      - feels_like (zelfde als temp)
      - pop (0..1, gebaseerd op precipprob % / 100)
      - rain_mm (precip)
      - wind (windspeed)
    """
    df = get_forecast_weather_daily(
        mode="latlon",
        lat=lat,
        lon=lon,
        days_ahead=days_ahead,
    )

    out: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        date_val = row["date"]
        if isinstance(date_val, dt.date):
            date_str = date_val.isoformat()
        else:
            date_str = str(date_val)

        pop = float(row.get("precipprob", 0.0) or 0.0) / 100.0
        rain = float(row.get("precip", 0.0) or 0.0)
        wind = float(row.get("windspeed", 0.0) or 0.0)
        temp = float(row.get("temp", 0.0) or 0.0)

        out.append(
            {
                "date": date_str,
                "temp": temp,
                "feels_like": temp,  # VC heeft geen losse feels_like op daily → zelfde als temp
                "pop": pop,
                "rain_mm": rain,
                "wind": wind,
            }
        )

    return out
