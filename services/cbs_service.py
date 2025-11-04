# services/cbs_service.py
from __future__ import annotations
import requests
from datetime import date
from typing import List, Dict

BASE = "https://opendata.cbs.nl/ODataApi/OData"

def _period_code(d: date) -> str:
    return f"{d.year}MM{d.month:02d}"

def _prev_month(d: date) -> date:
    y, m = d.year, d.month - 1
    if m == 0:
        y, m = y - 1, 12
    return date(y, m, 15)

def _ym_list(months_back: int) -> List[str]:
    y, m = date.today().year, date.today().month
    out = []
    for _ in range(months_back):
        out.append(f"{y}MM{m:02d}")
        m -= 1
        if m == 0:
            y -= 1; m = 12
    return list(reversed(out))

def _pick_numeric_field(item: dict, preferred: List[str]) -> str:
    keys = {k.lower(): k for k in item.keys()}
    for p in preferred:
        if p.lower() in keys:
            return keys[p.lower()]
    for k, v in item.items():
        if isinstance(v, (int, float)): return k
        if isinstance(v, str):
            try:
                float(v.replace(",", ".")); return k
            except Exception:
                pass
    raise KeyError("Geen numeriek veld gevonden in CBS record")

def _pick_period_field(item: dict) -> str:
    """
    Vind veldnaam voor periode (kan 'Periods', 'Perioden', 'Period', etc. heten).
    """
    # 1) voorkeursnamen
    for cand in ("Periods", "Perioden", "Period", "Periode"):
        if cand in item: return cand
    # 2) fuzzy: keys die 'period' of 'periode' bevatten
    for k in item.keys():
        kl = k.lower()
        if "period" in kl or "periode" in kl:
            return k
    # 3) fallback: gooi alle keys terug zodat we kunnen debuggen
    raise KeyError(f"Periodeveld niet gevonden. Keys: {list(item.keys())}")

def _odata_select(dataset: str, filter_q: str, select: str = None, top: int = None) -> List[dict]:
    url = f"{BASE}/{dataset}/TypedDataSet?{filter_q}"
    if select: url += f"&$select={select}"
    if top:    url += f"&$top={top}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json().get("value", [])

# -----------------------------
# 83693NED — Consumentenvertrouwen (1 maand, backtrack)
# -----------------------------
def get_consumer_confidence(dataset: str = "83693NED", when: date | None = None, max_backtrack: int = 3) -> Dict:
    when = when or date.today()
    tries = 0
    while tries <= max_backtrack:
        period = _period_code(when)
        rows = _odata_select(dataset, f"$filter=Periods eq '{period}'")
        if rows:
            item = rows[0]
            field = _pick_numeric_field(item, ["ConsumerConfidence_2", "Consumentenvertrouwen_2", "Consumerconfidence"])
            raw = item[field]
            val = float(raw) if isinstance(raw, (int, float)) else float(str(raw).replace(",", "."))
            return {"period": period, "value": val, "field": field}
        when = _prev_month(when)
        tries += 1
    raise RuntimeError(f"CBS Consumentenvertrouwen niet gevonden na {max_backtrack} maanden backtrack.")

# -----------------------------
# 83693NED — Consumentenvertrouwen (reeks)
# -----------------------------
def get_cci_series(months_back: int = 18, dataset: str = "83693NED") -> List[Dict]:
    periods = _ym_list(months_back)
    quoted = ",".join([f"%27{p}%27" for p in periods])
    rows = _odata_select(dataset, f"$filter=Periods in ({quoted})")
    if not rows: return []

    # detecteer velden
    period_field = _pick_period_field(rows[0])
    value_field  = _pick_numeric_field(rows[0], ["ConsumerConfidence_2", "Consumentenvertrouwen_2", "Consumerconfidence"])

    out = []
    for it in rows:
        raw = it[value_field]
        val = float(raw) if isinstance(raw, (int, float)) else float(str(raw).replace(",", "."))
        out.append({"period": it[period_field], "cci": val})
    out.sort(key=lambda x: x["period"])
    return out

# -----------------------------
# 85828NED — Detailhandel (per branche)
# -----------------------------
def get_retail_index(series: str = "Omzetontwikkeling_1",
                     branch_code: str = "DH_TOTAAL",
                     months_back: int = 18,
                     dataset: str = "85828NED") -> List[Dict]:
    periods = _ym_list(months_back)
    quoted = ",".join([f"%27{p}%27" for p in periods])

    rows = _odata_select(dataset, f"$filter=Periods in ({quoted}) and Branches_2 eq '{branch_code}'")
    if not rows:
        rows = _odata_select(dataset, f"$filter=Periods in ({quoted}) and Branches eq '{branch_code}'")
    if not rows: return []

    period_field = _pick_period_field(rows[0])
    value_field  = _pick_numeric_field(rows[0], [series])

    out = []
    for it in rows:
        raw = it[value_field]
        val = float(raw) if isinstance(raw, (int, float)) else float(str(raw).replace(",", "."))
        out.append({"period": it[period_field], "retail_value": val, "series": value_field, "branch": branch_code})
    out.sort(key=lambda x: x["period"])
    return out
