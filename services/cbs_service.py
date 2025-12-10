# services/cbs_service.py
from __future__ import annotations
import requests
from datetime import date
from typing import List, Dict, Tuple

BASE = "https://opendata.cbs.nl/ODataApi/OData"


# -----------------------------
# Kleine generieke helpers
# -----------------------------
def _odata_select(dataset: str, path: str, params: str = "") -> List[dict]:
    """
    Eenvoudige wrapper om OData op te halen.

    Voorbeeld-URL:
    https://opendata.cbs.nl/ODataApi/OData/{dataset}/{path}?{params}
    """
    url = f"{BASE}/{dataset}/{path}"
    if params:
        url += f"?{params}"

    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()
    return js.get("value", [])


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


# -----------------------------
# 1) Consumentenvertrouwen – 83693NED
# -----------------------------
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

    We vragen alleen de relevante kolommen op:
    - Perioden
    - Consumentenvertrouwen_1
    """
    try:
        # Alleen de relevante kolommen + ruim genoeg top
        rows = _odata_select(
            dataset,
            "TypedDataSet",
            "$select=Perioden,Consumentenvertrouwen_1&$top=6000",
        )
    except requests.HTTPError:
        return []

    if not rows:
        return []

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

    # sorteren op periodecode (bv. '1986MM04' → string-sort is ok)
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


# -----------------------------
# 2) Detailhandelindex – 85828NED
# -----------------------------
def get_retail_index(
    series: str = "Ongecorrigeerd_1",
    branch_code_or_title: str = "DH_TOTAAL",  # wordt bewust genegeerd
    months_back: int = 24,
    dataset: str = "85828NED",
) -> List[Dict]:
    """
    Haalt een generieke detailhandelindex uit 85828NED.

    LET OP:
    - We negeren branch_code_or_title volledig.
    - We pakken simpelweg de kolom 'Ongecorrigeerd_1' over alle branches heen
      en middelen per periode.
    - De 'series' parameter wordt genegeerd en hard gecodeerd naar 'Ongecorrigeerd_1',
      zodat bestaande aanroepen met series="Omzetontwikkeling_1" niet crashen.

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
    # We forceren de daadwerkelijke kolomnaam in deze dataset:
    value_field = "Ongecorrigeerd_1"

    try:
        # Alleen Perioden + de gekozen kolom ophalen
        rows = _odata_select(
            dataset,
            "TypedDataSet",
            f"$select=Perioden,{value_field}&$top=6000",
        )
    except requests.HTTPError:
        return []

    if not rows:
        return []

    period_field = _pick_period_field(rows[0])

    # Verzamel per periode alle niet-lege waarden, en neem daarna het gemiddelde
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
                "branch": "ALL",  # geen branch-filtering: macro NL
            }
        )

    # sorteer op periode
    out.sort(key=lambda x: x["period"])

    if months_back and len(out) > months_back:
        out = out[-months_back:]

    return out


# -----------------------------
# 3) Postcode4 stub – simpele context voor de AI Copilot
# -----------------------------
def get_cbs_stats_for_postcode4(postcode4: str) -> Dict:
    """
    Simpele placeholder voor CBS-context op postcode4-niveau.

    Nu:
    - Levert generieke indices terug (NL = 100, iets erboven)
    - Voorkomt import errors in de AI Copilot

    Later:
    - Kun je dit vervangen door een echte koppeling
      (bv. CBS buurt-/wijkstatistieken op basis van postcode → wijkcode).
    """
    postcode4 = (postcode4 or "").strip()
    if not postcode4:
        return {}

    return {
        "postcode4": postcode4,
        "avg_income_index": 100,          # NL = 100
        "population_density_index": 110,  # iets boven het gemiddelde
        "note": "Demo-indices op basis van CBS. Vervang deze stub later door een echte postcode4-koppeling.",
    }
