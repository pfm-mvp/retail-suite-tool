# services/cbs_service.py
from __future__ import annotations
import requests
from datetime import date
from typing import List, Dict, Tuple

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
    
def _ym_range(months_back: int) -> tuple[str, str]:
    ys = _ym_list(months_back)
    return ys[0], ys[-1]

def _odata_select(dataset: str, path: str, params: str = "") -> List[dict]:
    # path: e.g. "TypedDataSet" or "Branches_2"
    url = f"{BASE}/{dataset}/{path}"
    if params:
        url += f"?{params}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()
    return js.get("value", [])

def _pick_numeric_field(item: dict, preferred: List[str]) -> str:
    keys = {k.lower(): k for k in item.keys()}
    for p in preferred:
        if p.lower() in keys:
            return keys[p.lower()]
    for k, v in item.items():
        if isinstance(v, (int, float)):
            return k
        if isinstance(v, str):
            try:
                float(v.replace(",", ".")); return k
            except Exception:
                pass
    raise KeyError("Geen numeriek veld gevonden in CBS record")

def _pick_period_field(item: dict) -> str:
    for cand in ("Periods", "Perioden", "Period", "Periode"):
        if cand in item:
            return cand
    for k in item.keys():
        kl = k.lower()
        if "period" in kl or "periode" in kl:
            return k
    raise KeyError(f"Periodeveld niet gevonden. Keys: {list(item.keys())}")

# -----------------------------
# 83693NED — Consumentenvertrouwen (1 maand, backtrack)
# -----------------------------
def get_consumer_confidence(dataset: str = "83693NED", when: date | None = None, max_backtrack: int = 3) -> Dict:
    when = when or date.today()
    tries = 0
    while tries <= max_backtrack:
        period = _period_code(when)
        rows = _odata_select(dataset, "TypedDataSet", f"$filter=Periods eq '{period}'")
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
    start, end = _ym_range(months_back)
    try:
        rows = _odata_select(dataset, "TypedDataSet", f"$filter=Periods ge '{start}' and Periods le '{end}'&$top=5000")
    except requests.HTTPError:
        return []
    if not rows:
        return []

    period_field = _pick_period_field(rows[0])
    value_field  = _pick_numeric_field(rows[0], ["ConsumerConfidence_2", "Consumentenvertrouwen_2", "Consumerconfidence"])

    out = []
    for it in rows:
        raw = it.get(value_field)
        if raw is None: 
            continue
        val = float(raw) if isinstance(raw, (int, float)) else float(str(raw).replace(",", "."))
        out.append({"period": it[period_field], "cci": val})
    out.sort(key=lambda x: x["period"])
    return out

# -----------------------------
# 85828NED — Branches-lijst ophalen (Key ↔ Title)
# -----------------------------
def list_retail_branches(dataset: str = "85828NED") -> Tuple[str, List[Dict]]:
    """
    Retourneert (branch_dim_name, items) waarbij items = [{'key':..., 'title':...}, ...]
    Probeert Branches_2, daarna Branches.
    """
    for dim in ("Branches_2", "Branches"):
        try:
            rows = _odata_select(dataset, dim, "$select=Key,Title&$top=5000")
            if rows:
                return dim, [{"key": r["Key"], "title": r.get("Title", str(r["Key"]))} for r in rows]
        except requests.HTTPError:
            continue
    return "", []

def _find_branch_key(branches: List[Dict], query: str) -> str | None:
    q = (query or "").strip().lower()
    # exacte match op key of title
    for b in branches:
        if str(b["key"]).lower() == q or str(b["title"]).lower() == q:
            return str(b["key"])
    # bevat-match op title
    for b in branches:
        if q and q in str(b["title"]).lower():
            return str(b["key"])
    return None

# -----------------------------
# 85828NED — Detailhandel (per branche, server-side filter met Key; fallback client-side)
# -----------------------------
def get_retail_index(series: str = "Omzetontwikkeling_1",
                     branch_code_or_title: str = "DH_TOTAAL",
                     months_back: int = 18,
                     dataset: str = "85828NED") -> List[Dict]:
    start, end = _ym_range(months_back)

    # 1) Haal branch-dimensie op → probeer server-side met Key
    dim_name, branches = list_retail_branches(dataset)
    branch_key = _find_branch_key(branches, branch_code_or_title) if branches else None

    rows = []
    if dim_name and branch_key:
        try:
            # server-side filter met range + branch-key
            rows = _odata_select(
                dataset, "TypedDataSet",
                f"$filter=Periods ge '{start}' and Periods le '{end}' and {dim_name} eq '{branch_key}'&$top=5000"
            )
        except requests.HTTPError:
            rows = []

    # 2) Fallback: alles binnen range ophalen en client-side filteren op branch
    if not rows:
        try:
            rows = _odata_select(dataset, "TypedDataSet", f"$filter=Periods ge '{start}' and Periods le '{end}'&$top=5000")
        except requests.HTTPError:
            return []
        if not rows:
            return []

        period_field = _pick_period_field(rows[0])
        branch_field = None
        for k in rows[0].keys():
            kl = k.lower()
            if "branch" in kl or "branches" in kl or "branche" in kl:
                branch_field = k
                break
        if not branch_field:
            return []

        def _match(item: dict) -> bool:
            val = str(item.get(branch_field, "")).strip().lower()
            if branch_key is not None:
                return val == str(branch_key).lower()
            if branches:
                return any(val == str(b["key"]).lower() or val == str(b["title"]).lower() for b in branches)
            return branch_code_or_title.lower() in val

        rows = [r for r in rows if _match(r)]
        if not rows:
            return []

        value_field  = _pick_numeric_field(rows[0], [series])
    else:
        period_field = _pick_period_field(rows[0])
        value_field  = _pick_numeric_field(rows[0], [series])

    out = []
    for it in rows:
        raw = it.get(value_field)
        if raw is None:
            continue
        val = float(raw) if isinstance(raw, (int, float)) else float(str(raw).replace(",", "."))
        out.append({
            "period": it[period_field],
            "retail_value": val,
            "series": value_field,
            "branch": branch_code_or_title
        })
    out.sort(key=lambda x: x["period"])
    return out

# -----------------------------
# Postcode4 stub – simpele context voor de AI Copilot
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
        "note": "Demo-indices op basis van CBS. Vervang deze stub later door een echte postcode4-koppeling."
    }
