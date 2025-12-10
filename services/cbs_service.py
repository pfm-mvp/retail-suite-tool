# services/cbs_service.py
from __future__ import annotations
import requests
from typing import List, Dict

BASE = "https://opendata.cbs.nl/ODataApi/OData"


def _to_float(raw):
    """Probeer CBS-veld naar float te casten, anders None."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    try:
        return float(str(raw).replace(",", "."))
    except Exception:
        return None


def _pick_period_field(item: dict) -> str:
    """
    Zoek het periodeveld: 'Perioden', 'Periods', ...
    """
    for cand in ("Perioden", "Periods", "Period", "Periode"):
        if cand in item:
            return cand
    for k in item.keys():
        if "period" in k.lower() or "periode" in k.lower():
            return k
    raise KeyError(f"Periodeveld niet gevonden. Keys: {list(item.keys())}")


# ============================================================
# 1) Consumentenvertrouwen – 83693NED
# ============================================================
def get_cci_series(
    months_back: int = 24,
    dataset: str = "83693NED",
) -> List[Dict]:
    """
    Haalt de CCI-reeks op uit 83693NED.

    Return:
      [
        {"period": "YYYYMMxx", "cci": float},
        ...
      ]

    We gebruiken exact dezelfde URL als je healthcheck:
    https://opendata.cbs.nl/ODataApi/OData/83693NED/TypedDataSet?$top=6000
    en lezen hieruit:
    - Perioden
    - Consumentenvertrouwen_1
    """
    url = f"{BASE}/{dataset}/TypedDataSet?$top=6000"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    js = resp.json()
    rows = js.get("value", [])

    if not rows:
        return []

    # In jouw sample is dit duidelijk:
    # "Perioden": "1986MM04"
    # "Consumentenvertrouwen_1": 2
    period_field = _pick_period_field(rows[0])
    value_field = "Consumentenvertrouwen_1"

    out: List[Dict] = []
    for it in rows:
        per = it.get(period_field)
        raw = it.get(value_field)
        val = _to_float(raw)
        if not per or val is None:
            continue
        out.append({"period": str(per), "cci": val})

    # sorteer op periodecode (bv. '1986MM04' → string-sort is prima)
    out.sort(key=lambda x: x["period"])

    if months_back and len(out) > months_back:
        out = out[-months_back:]

    return out


def get_consumer_confidence(
    dataset: str = "83693NED",
    months_back: int = 3,
) -> Dict:
    """
    Convenience: geef de laatste maand met CCI terug.
    """
    series = get_cci_series(months_back=months_back, dataset=dataset)
    if not series:
        return {}
    return series[-1]  # laatste element = meest recent


# ============================================================
# 2) Detailhandelindex – 85828NED
# ============================================================
def get_retail_index(
    series: str = "Ongecorrigeerd_1",
    branch_code_or_title: str = "ALL",  # genegeerd, macro NL
    months_back: int = 24,
    dataset: str = "85828NED",
) -> List[Dict]:
    """
    Haalt een generieke detailhandelindex uit 85828NED.

    Belangrijk:
    - Jouw sample laat zien dat de waarde-kolom hier 'Ongecorrigeerd_1' is.
    - De eerste jaren zijn NULL, maar later komen er echte waarden.
    - We negeren branches en middelen alle niet-NULL waarden per periode.

    Return:
      [
        {
          "period": "YYYYMMxx",
          "retail_value": float,   # gemiddelde over alle branches
          "series": "Ongecorrigeerd_1",
          "branch": "ALL"
        },
        ...
      ]
    """
    value_field = "Ongecorrigeerd_1"

    url = f"{BASE}/{dataset}/TypedDataSet?$top=6000"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    js = resp.json()
    rows = js.get("value", [])

    if not rows:
        return []

    period_field = _pick_period_field(rows[0])

    # Verzamel per periode alle niet-lege waarden, gemiddeld per maand
    per_map: Dict[str, List[float]] = {}

    for it in rows:
        per = it.get(period_field)
        raw = it.get(value_field)
        val = _to_float(raw)
        if not per or val is None:
            continue
        per_str = str(per)
        per_map.setdefault(per_str, []).append(val)

    if not per_map:
        return []

    out: List[Dict] = []
    for per_str, vals in per_map.items():
        if not vals:
            continue
        avg_val = float(sum(vals) / len(vals))
        out.append(
            {
                "period": per_str,
                "retail_value": avg_val,
                "series": value_field,
                "branch": "ALL",
            }
        )

    # sorteer op periode
    out.sort(key=lambda x: x["period"])

    if months_back and len(out) > months_back:
        out = out[-months_back:]

    return out


# ============================================================
# 3) Postcode4 stub – voor context in de AI Copilot
# ============================================================
def get_cbs_stats_for_postcode4(postcode4: str) -> Dict:
    """
    Simpele placeholder voor CBS-context op postcode4-niveau.
    Nu nog demo-data; later kun je dit vervangen door echte buurt-/wijkstatistieken.
    """
    postcode4 = (postcode4 or "").strip()
    if not postcode4:
        return {}

    return {
        "postcode4": postcode4,
        "avg_income_index": 100,          # NL = 100
        "population_density_index": 110,  # iets boven het gemiddelde
        "note": (
            "Demo-indices op basis van CBS. "
            "Vervang deze stub later door een echte postcode4-koppeling."
        ),
    }
