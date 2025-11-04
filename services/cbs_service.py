# services/cbs_service.py
from __future__ import annotations
import requests
from datetime import date
from typing import List, Dict
import calendar

BASE = "https://opendata.cbs.nl/ODataApi/OData"

def _period_code(d: date) -> str:
    # CBS periodenotation: YYYYMMNN, voor maand volstaat YYYYMM
    return f"{d.year}MM{d.month:02d}"

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
    # Kies veld uit candidates (verschillen tussen datasets/versies komen voor)
    keys = {k.lower(): k for k in item.keys()}
    for p in preferred:
        if p.lower() in keys:
            return keys[p.lower()]
    # fallback: pak eerste numerieke veld
    for k, v in item.items():
        if isinstance(v, (int, float)): return k
        if isinstance(v, str):
            try:
                float(v.replace(",", ".")); return k
            except: pass
    raise KeyError("Geen numeriek veld gevonden in CBS record")

def _odata_select(dataset: str, filter_q: str, select: str = None, top: int = None) -> List[dict]:
    url = f"{BASE}/{dataset}/TypedDataSet?{filter_q}"
    if select: url += f"&$select={select}"
    if top:    url += f"&$top={top}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json().get("value", [])

# -----------------------------
# 83693NED — Consumentenvertrouwen (maandelijks)
# -----------------------------
def get_cci_series(months_back: int = 18, dataset: str = "83693NED") -> List[Dict]:
    # Haal reeks voor de opgegeven maanden (exclusief toekomst); filter op Periods IN (...)
    periods = _ym_list(months_back)
    quoted = ",".join([f"%27{p}%27" for p in periods])
    rows = _odata_select(dataset, f"$filter=Periods in ({quoted})")
    if not rows: return []

    # Vind de juiste kolom (vaak ConsumerConfidence_2)
    field = _pick_numeric_field(rows[0], ["ConsumerConfidence_2", "Consumentenvertrouwen_2", "Consumerconfidence"])
    out = []
    for it in rows:
        raw = it[field]
        val = float(raw) if isinstance(raw, (int, float)) else float(str(raw).replace(",", "."))
        out.append({"period": it["Periods"], "cci": val})
    # sorteren op period
    out.sort(key=lambda x: x["period"])
    return out

# -----------------------------
# 85828NED — Detailhandel; omzet/volume (index of %), per branche
# -----------------------------
def get_retail_index(series: str = "Omzetontwikkeling_1",  # voorbeeld: "Omzetontwikkeling_1", "Volumeontwikkeling_2", of een andere meeteenheid
                     branch_code: str = "DH_TOTAAL",       # totaal detailhandel; anders bv. "DH_NONFOOD", "DH_FOOD" of specifieke SBI
                     months_back: int = 18,
                     dataset: str = "85828NED") -> List[Dict]:
    """
    Haalt maanddata voor detailhandel op. 'series' = kolomnaam (zie dataset), 'branch_code' = Branches_2 code.
    Voorbeeld codes: DH_TOTAAL (totaal), DH_FOOD, DH_NONFOOD, DH_NONFOOD_Kleding (afhankelijk van dataset).
    """
    periods = _ym_list(months_back)
    quoted = ",".join([f"%27{p}%27" for p in periods])
    # In veel CBS-datasets heet de branche-dimensie 'Branches_2' of 'Branches'
    # We proberen eerst Branches_2, daarna Branches
    rows = _odata_select(dataset, f"$filter=Periods in ({quoted}) and Branches_2 eq '{branch_code}'")
    if not rows:
        rows = _odata_select(dataset, f"$filter=Periods in ({quoted}) and Branches eq '{branch_code}'")
    if not rows: return []

    # series kolom kiezen (fallback naar numeriek veld)
    field = _pick_numeric_field(rows[0], [series])
    out = []
    for it in rows:
        raw = it[field]
        val = float(raw) if isinstance(raw, (int, float)) else float(str(raw).replace(",", "."))
        out.append({"period": it["Periods"], "retail_value": val, "series": field, "branch": branch_code})
    out.sort(key=lambda x: x["period"])
    return out
