# services/cbs_service.py
from __future__ import annotations
import requests
from datetime import date
from calendar import monthrange

def _period_code(d: date) -> str:
    return f"{d.year}MM{d.month:02d}"

def _prev_month(d: date) -> date:
    y, m = (d.year, d.month - 1)
    if m == 0:
        y, m = (d.year - 1, 12)
    # zet op 15e van de maand om dagproblemen te vermijden
    return date(y, m, min(15, monthrange(y, m)[1]))

def _pick_confidence_field(item: dict) -> str | None:
    # Zoek het juiste veld (nl/eng naam of suffixen kunnen wisselen)
    candidates = [k for k in item.keys() if "confidence" in k.lower() or "vertrouwen" in k.lower()]
    # prioriteer exact "ConsumerConfidence_2" als die bestaat
    for k in candidates:
        if k.lower() == "consumerconfidence_2":
            return k
    # anders: pak de eerste numerieke candidate
    for k in candidates:
        v = item.get(k, None)
        if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace("-", "").isdigit()):
            return k
    return candidates[0] if candidates else None

def get_consumer_confidence(dataset: str = "83693NED", when: date | None = None, max_backtrack: int = 3) -> dict:
    """
    Haalt consumentenvertrouwen op voor 'when'.
    Als de maand nog niet gepubliceerd is (lege resultset), backtrack tot max_backtrack maanden.
    Retourneert: {"period": "YYYYMMNN", "value": float, "field": "ConsumerConfidence_2"}
    """
    when = when or date.today()
    tries = 0
    last_error = None

    while tries <= max_backtrack:
        period = _period_code(when)
        url = f"https://opendata.cbs.nl/ODataApi/OData/{dataset}/TypedDataSet?$filter=Periods%20eq%20%27{period}%27"
        r = requests.get(url, timeout=20)
        try:
            r.raise_for_status()
            data = r.json().get("value", [])
            if not data:
                # niets gepubliceerd voor deze maand → stap 1 maand terug
                when = _prev_month(when)
                tries += 1
                continue

            item = data[0]
            field = _pick_confidence_field(item)
            if not field:
                raise KeyError("Kon confidence-veld niet vinden in CBS response.")

            raw = item[field]
            value = float(raw) if isinstance(raw, (int, float)) else float(str(raw).replace(",", "."))
            return {"period": period, "value": value, "field": field}

        except Exception as e:
            last_error = e
            when = _prev_month(when)
            tries += 1

    # als alles faalt: geef nette fout terug
    raise RuntimeError(f"CBS consumentenvertrouwen niet gevonden na {max_backtrack} maanden backtrack. Laatste fout: {last_error}")
