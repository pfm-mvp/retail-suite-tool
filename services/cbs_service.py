# cbs_service.py — Consumer Confidence from CBS 83693NED
from __future__ import annotations
import requests
from datetime import date

def _period_code(d: date) -> str:
    return f"{d.year}MM{d.month:02d}"

def get_consumer_confidence(dataset: str = "83693NED", when: date | None = None) -> dict:
    when = when or date.today()
    period = _period_code(when)
    url = f"https://opendata.cbs.nl/ODataApi/OData/{dataset}/TypedDataSet?$filter=Periods%20eq%20%27{period}%27"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    val = r.json()["value"][0]["ConsumerConfidence_2"]
    return {"period": period, "value": val}
