# services/cbs_service.py

import requests
from typing import List, Dict

BASE_URL = "https://opendata.cbs.nl/ODataApi/OData"


def _fetch_typed_dataset(dataset: str, top: int = 5000) -> List[Dict]:
    """
    Haalt ruwe records op uit CBS OData TypedDataSet.
    """
    url = f"{BASE_URL}/{dataset}/TypedDataSet?$top={top}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    js = resp.json()
    return js.get("value", [])


def get_cci_series(months_back: int = 24) -> List[Dict]:
    """
    Haalt consumentenvertrouwen (CCI) op uit dataset 83693NED.

    Return:
        [
            {"period": "1986MM04", "cci": 2.0},
            {"period": "1986MM05", "cci": 8.0},
            ...
        ]
    """
    rows = _fetch_typed_dataset("83693NED", top=5000)
    if not rows:
        return []

    series: List[Dict] = []

    for r in rows:
        period_code = r.get("Perioden")
        if not period_code:
            continue

        val = r.get("Consumentenvertrouwen_1")
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue

        series.append(
            {
                "period": str(period_code),
                "cci": v,
            }
        )

    # Sorteer op periode-string en neem de laatste N maanden
    series = sorted(series, key=lambda x: x["period"])
    if months_back and len(series) > months_back:
        series = series[-months_back:]

    return series


def get_retail_index(months_back: int = 24) -> List[Dict]:
    """
    Haalt een macro detailhandel-index uit dataset 85828NED.

    We gebruiken:
      - Perioden (bijv. '2000MM01')
      - Ongecorrigeerd_1 als waarde
      - Gemiddelde over alle branches → 1 macroreeks
    """
    # Belangrijk: CBS eist < 10000 records per query
    rows = _fetch_typed_dataset("85828NED", top=1000)
    if not rows:
        return []

    by_period: Dict[str, list] = {}

    for r in rows:
        period_code = r.get("Perioden")
        if not period_code:
            continue

        raw_val = r.get("Ongecorrigeerd_1")
        if raw_val is None:
            continue

        try:
            v = float(raw_val)
        except (TypeError, ValueError):
            continue

        key = str(period_code)
        by_period.setdefault(key, []).append(v)

    series: List[Dict] = []
    for period_code, vals in by_period.items():
        if not vals:
            continue
        avg_val = sum(vals) / len(vals)
        series.append(
            {
                "period": period_code,
                "retail_value": avg_val,
            }
        )

    # Sorteer en knip op months_back
    series = sorted(series, key=lambda x: x["period"])
    if months_back and len(series) > months_back:
        series = series[-months_back:]

    return series
