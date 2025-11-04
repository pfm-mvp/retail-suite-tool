# weather_service.py — OpenWeather One Call 3.0 thin client
from __future__ import annotations
import requests, datetime as dt
from typing import List, Dict, Any

BASE = "https://api.openweathermap.org/data/3.0/onecall"

def get_daily_forecast(lat: float, lon: float, api_key: str, days_ahead: int = 7) -> List[Dict[str, Any]]:
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric", "exclude": "minutely,alerts"}
    r = requests.get(BASE, params=params, timeout=20)
    r.raise_for_status()
    j = r.json()
    out = []
    for d in j.get("daily", [])[:days_ahead]:
        out.append({
            "date": dt.date.fromtimestamp(d["dt"]).isoformat(),
            "temp": float(d["temp"]["day"]),
            "feels_like": float(d["feels_like"]["day"]),
            "pop": float(d.get("pop", 0.0)),       # 0..1
            "rain_mm": float(d.get("rain", 0.0)),
            "wind": float(d.get("wind_speed", 0.0))
        })
    return out
