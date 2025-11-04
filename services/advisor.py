from statistics import median
from datetime import date
from .weather import get_daily_forecast
from .cbs import get_consumer_confidence

def compute_temp_anomaly(forecast_day_temp, hist_day_temps):
    if not hist_day_temps: return 0.0
    return forecast_day_temp - (sum(hist_day_temps)/len(hist_day_temps))

def advisor_for_store(store_name: str, hist_day, forecast_day, cci_value: float):
    """
    hist_day: dict met keys {'visitors','conversion','spv','temps': [..]} voor dezelfde weekdag (uit jouw al-werkende Vemcount views)
    forecast_day: dict van weather.py (temp, feels_like, pop, rain_mm, wind)
    returns: {'store_actions': [...], 'regional_actions':[...]}
    """
    acts_store, acts_region = [], []
    temp_anom = compute_temp_anomaly(forecast_day["temp"], hist_day.get("temps", []))
    pop = forecast_day["pop"]  # 0..1

    # Weer → staffing & floor actions
    if pop >= 0.60 and hist_day.get("visitors", 0) >= median([hist_day.get("visitors",0), 1]):
        acts_store.append("Verplaats pauzes 30 min vóór verwachte bui; focus op begroeten/fit-advies om conversiedip te voorkomen.")
    if temp_anom >= 4:
        acts_store.append("Verleng piekshift +1u en highlight lichte/outdoor sets bij entree (verwachte +SPV).")
    if forecast_day["feels_like"] <= 5:
        acts_store.append("Warmte/comfort-maatregelen bij entree; promoot cold-weather accessoires.")

    # Dead-hour/SPV taakset (hist-inzicht gebruik je al)
    if hist_day.get("visitors",0) >= hist_day.get("visitors_p30",0) and hist_day.get("spv",0) < hist_day.get("spv_median",0):
        acts_store.append("Dead hour: voer 3-stappen script uit (actieve begroeting, add-on prompt, kassa-script). Doel: +€0,50 SPV.")

    # Macro (CCI) → boodschap
    if cci_value < 0:
        acts_store.append("Leg nadruk op bundel/waardeproposities vandaag.")
    else:
        acts_store.append("Push premium add-ons tijdens drukste uren; richt SPV-doel op +€0,50.")

    # Regionale invalshoek
    if pop >= 0.60:
        acts_region.append(f"{store_name}: regen-risico → mobiele FTE inzetten op 16–19u, prioriteit coaching op queue-buster.")
    if abs(temp_anom) >= 4:
        acts_region.append(f"{store_name}: temperatuurafwijking {temp_anom:+.0f}°C → check schappen/thema-omslag.")

    return {"store_actions": acts_store, "regional_actions": acts_region}

def build_advice(company_name, stores_hist_by_weekday, lat, lon, api_key, days=7, cci_period_code=None):
    # 1) weer
    forecast = get_daily_forecast(lat, lon, api_key, days)
    # 2) CCI
    if not cci_period_code:
        today = date.today()
        cci_period_code = f"{today.year}MM{today.month:02d}"
    cci = get_consumer_confidence(cci_period_code)["value"]

    out = {"company": company_name, "cci": cci, "days": []}
    for f in forecast:
        wd = date.fromisoformat(f["date"]).weekday()  # 0=Mon
        day_hist = stores_hist_by_weekday.get(wd, {})
        stores_out = []
        for store_name, hist in day_hist.items():
            advice = advisor_for_store(store_name, hist, f, cci)
            stores_out.append({"store": store_name, **advice})
        out["days"].append({"date": f["date"], "weather": f, "stores": stores_out})
    return out
