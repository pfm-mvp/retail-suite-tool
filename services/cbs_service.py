# services/cbs_service.py
from __future__ import annotations
import requests
from datetime import date
from typing import List, Dict, Tuple

BASE = "https://opendata.cbs.nl/ODataApi/OData"


# -----------------------------
# Generieke helpers
# -----------------------------
def _ym_list(months_back: int) -> List[str]:
    """Return lijst met periodecodes 'YYYYMM' style, nieuwste laatst."""
    y, m = date.today().year, date.today().month
    out: List[str] = []
    for _ in range(months_back):
        out.append(f"{y}MM{m:02d}")
        m -= 1
        if m == 0:
            y -= 1
            m = 12
    return list(reversed(out))


def _odata_select(dataset: str, path: str, params: str = "") -> List[dict]:
    """
    Kleine wrapper om OData op te halen.

    path: bv. "TypedDataSet" of "Branches_2".
    params: bv. "$top=5000" (geen $filter meer in deze versie – dat doen we client-side).
    """
    url = f"{BASE}/{dataset}/{path}"
    if params:
        url += f"?{params}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()
    return js.get("value", [])


def _pick_numeric_field(item: dict, preferred: List[str]) -> str:
    """Zoek een numeriek veld, bij voorkeur uit 'preferred'."""
    keys = {k.lower(): k for k in item.keys()}
    for p in preferred:
        if p.lower() in keys:
            return keys[p.lower()]

    # Fallback: eerste veld dat naar float te casten is
    for k, v in item.items():
        if isinstance(v, (int, float)):
            return k
        if isinstance(v, str):
            try:
                float(v.replace(",", "."))
                return k
            except Exception:
                pass
    raise KeyError("Geen numeriek veld gevonden in CBS record")


def _pick_period_field(item: dict) -> str:
    """Zoek 'Perioden' / 'Periods' / etc."""
    for cand in ("Perioden", "Periods", "Period", "Periode"):
        if cand in item:
            return cand
    for k in item.keys():
        kl = k.lower()
        if "period" in kl or "periode" in kl:
            return k
    raise KeyError(f"Periodeveld niet gevonden. Keys: {list(item.keys())}")


# ============================================================
# 1) Consumentenvertrouwen – 83693NED
# ============================================================

def get_consumer_confidence(
    dataset: str = "83693NED",
    months_back: int = 3,
) -> Dict:
    """
    Eenvoudige helper: pak de laatste maand *met* waarde uit de serie.
    Handig als je alleen de meest recente index nodig hebt.
    """
    series = get_cci_series(months_back=months_back, dataset=dataset)
    if not series:
        return {}
    return series[-1]  # laatste item


def get_cci_series(
    months_back: int = 18,
    dataset: str = "83693NED",
) -> List[Dict]:
    """
    Geeft een lijst terug met:
      [{'period': 'YYYYMMxx', 'cci': waarde}, ...]

    We halen maximaal 5000 records op (zonder server-side filter),
    zoeken zelf het periodeveld en numeriek veld en nemen daarna
    de laatste `months_back` maanden.
    """
    try:
        rows = _odata_select(dataset, "TypedDataSet", "$top=5000")
    except requests.HTTPError:
        return []
    if not rows:
        return []

    period_field = _pick_period_field(rows[0])
    value_field = _pick_numeric_field(
        rows[0],
        ["Consumentenvertrouwen_1", "ConsumerConfidence_1", "Consumerconfidence"],
    )

    out: List[Dict] = []
    for it in rows:
        period_code = it.get(period_field)
        raw = it.get(value_field)
        if period_code is None or raw is None:
            continue

        if isinstance(raw, (int, float)):
            val = float(raw)
        else:
            try:
                val = float(str(raw).replace(",", "."))
            except Exception:
                continue

        out.append({"period": str(period_code), "cci": val})

    out.sort(key=lambda x: x["period"])

    if months_back and len(out) > months_back:
        out = out[-months_back:]

    return out


# ============================================================
# 2) Detailhandelindex – 85828NED
# ============================================================

def list_retail_branches(dataset: str = "85828NED") -> Tuple[str, List[Dict]]:
    """
    Retourneert (branch_dim_name, items) waarbij items = [{'key':..., 'title':...}, ...]
    Probeert Branches_2, daarna Branches.

    Let op: voor 85828NED bestaan deze tabellen inmiddels níet meer,
    dus in de praktijk krijg je vaak ("", []) terug.
    """
    for dim in ("Branches_2", "Branches"):
        try:
            rows = _odata_select(dataset, dim, "$select=Key,Title&$top=5000")
            if rows:
                return dim, [
                    {"key": r["Key"], "title": r.get("Title", str(r["Key"]))}
                    for r in rows
                ]
        except requests.HTTPError:
            continue
    return "", []


def _find_branch_key(branches: List[Dict], query: str) -> str | None:
    q = (query or "").strip().lower()
    for b in branches:
        if str(b["key"]).lower() == q or str(b["title"]).lower() == q:
            return str(b["key"])
    for b in branches:
        if q and q in str(b["title"]).lower():
            return str(b["key"])
    return None


def get_retail_index(
    series: str = "Omzetontwikkeling_1",
    branch_code_or_title: str = "DH_TOTAAL",
    months_back: int = 18,
    dataset: str = "85828NED",
) -> List[Dict]:
    """
    Geeft een lijst terug met:
      [{'period': 'YYYYMMxx', 'retail_value': ..., 'series': ..., 'branch': ...}, ...]

    - Haalt maximaal 5000 regels op (zonder server-side filter).
    - Probeert een branch-kolom te vinden (bv. 'BedrijfstakkenBranchesSBI2008').
    - Probeert te filteren op `branch_code_or_title` (via dimensietabel of substring).
    - Als er géén matches zijn, valt het terug op *alle* rijen, zodat je nooit een
      lege serie terugkrijgt puur door een mismatch in branch-codes.
    """
    try:
        rows_all = _odata_select(dataset, "TypedDataSet", "$top=5000")
    except requests.HTTPError:
        return []
    if not rows_all:
        return []

    period_field = _pick_period_field(rows_all[0])

    # branch-dimensie bepalen: o.a. 'BedrijfstakkenBranchesSBI2008'
    branch_field = None
    for k in rows_all[0].keys():
        kl = k.lower()
        if "branch" in kl or "branches" in kl or "branche" in kl:
            branch_field = k
            break

    # Welke numerieke serie gebruiken? (bijv. Ongecorrigeerd_1)
    value_field = _pick_numeric_field(rows_all[0], [series])

    # --- Branch-filter (best-effort) ---
    rows = rows_all

    if branch_field and (branch_code_or_title or "").strip():
        dim_name, branches = list_retail_branches(dataset)
        branch_key = _find_branch_key(branches, branch_code_or_title) if branches else None

        def _match_branch(item: dict) -> bool:
            val = str(item.get(branch_field, "")).strip().lower()
            if not val:
                return False
            if branch_key is not None:
                # match op exacte key uit dimensietabel
                return val == str(branch_key).lower()
            if branches:
                # match tegen bekende keys/titels
                return any(
                    val == str(b["key"]).lower() or val == str(b["title"]).lower()
                    for b in branches
                )
            # laatste redmiddel: substring-match op de ruwe code
            return branch_code_or_title.lower() in val

        filtered = [r for r in rows_all if _match_branch(r)]

        # Belangrijk: als er geen enkele match is, niet alles weggooien,
        # maar gewoon alle rijen gebruiken.
        if filtered:
            rows = filtered

    # --- Records omzetten naar outputlijst ---
    out: List[Dict] = []
    for it in rows:
        period_code = it.get(period_field)
        raw = it.get(value_field)
        if period_code is None or raw is None:
            continue

        if isinstance(raw, (int, float)):
            val = float(raw)
        else:
            try:
                val = float(str(raw).replace(",", "."))
            except Exception:
                continue

        out.append(
            {
                "period": str(period_code),
                "retail_value": val,
                "series": value_field,
                "branch": branch_code_or_title,
            }
        )

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
