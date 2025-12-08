# services/cbs_service.py
from __future__ import annotations
import requests
from datetime import date
from typing import List, Dict, Tuple

BASE = "https://opendata.cbs.nl/ODataApi/OData"


# -----------------------------
# Helpers
# -----------------------------
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
            y -= 1
            m = 12
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
                float(v.replace(",", "."))
                return k
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
# 83693NED — Consumentenvertrouwen (laatste waarde rond een datum)
# -----------------------------
def get_consumer_confidence(
    dataset: str = "83693NED",
    when: date | None = None,
    max_backtrack: int = 3,
) -> Dict:
    """
    Haalt de consumentenvertrouwen-index op, rond een gegeven datum.

    Implementatie:
    - We lezen simpelweg de meest recente rijen o.b.v. ID (desc),
      zodat we niet hoeven te gokken of het veld 'Periods' of 'Perioden' heet.
    - Daarna zoeken we de eerste rij waarvan de periode <= target maandcode.
    """
    when = when or date.today()
    target_period = _period_code(when)

    # Haal wat recente punten op (ruim genoeg voor max_backtrack)
    try:
        rows = _odata_select(dataset, "TypedDataSet", "$orderby=ID desc&$top=24")
    except requests.HTTPError:
        return {}

    if not rows:
        raise RuntimeError("CBS Consumentenvertrouwen: geen data gevonden.")

    period_field = _pick_period_field(rows[0])
    value_field = _pick_numeric_field(
        rows[0],
        [
            "Consumentenvertrouwen_1",
            "ConsumerConfidence_1",
            "Consumentenvertrouwen_2",
            "ConsumerConfidence_2",
            "Consumerconfidence",
        ],
    )

    chosen = None
    # rows staan aflopend op ID; we willen de eerste <= target_period
    for it in rows:
        p = str(it.get(period_field, ""))
        if p <= target_period:
            chosen = it
            break

    # Als niets <= target_period, neem dan de meest recente
    if chosen is None:
        chosen = rows[0]

    raw = chosen[value_field]
    val = float(raw) if isinstance(raw, (int, float)) else float(str(raw).replace(",", "."))
    return {"period": chosen[period_field], "value": val, "field": value_field}


# -----------------------------
# 83693NED — Consumentenvertrouwen (reeks, laatste N maanden)
# -----------------------------
def get_cci_series(months_back: int = 18, dataset: str = "83693NED") -> List[Dict]:
    start, end = _ym_range(months_back)

    try:
        # Deze tabel is relatief klein, dus we halen gewoon alles (tot 5000)
        rows = _odata_select(dataset, "TypedDataSet", "$top=5000")
    except requests.HTTPError:
        return []

    if not rows:
        return []

    period_field = _pick_period_field(rows[0])
    value_field = _pick_numeric_field(
        rows[0],
        [
            "Consumentenvertrouwen_1",
            "ConsumerConfidence_1",
            "Consumentenvertrouwen_2",
            "ConsumerConfidence_2",
            "Consumerconfidence",
        ],
    )

    out: List[Dict] = []
    for it in rows:
        period = str(it.get(period_field, ""))
        if not (start <= period <= end):
            continue
        raw = it.get(value_field)
        if raw is None:
            continue
        val = float(raw) if isinstance(raw, (int, float)) else float(str(raw).replace(",", "."))
        out.append({"period": period, "cci": val})

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
                return dim, [
                    {"key": r["Key"], "title": r.get("Title", str(r["Key"]))}
                    for r in rows
                ]
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
# 85828NED — Detailhandelindex per branche
# -----------------------------
def get_retail_index(
    series: str = "Omzetontwikkeling_1",
    branch_code_or_title: str = "DH_TOTAAL",
    months_back: int = 18,
    dataset: str = "85828NED",
) -> List[Dict]:
    """
    Haalt de CBS detailhandelindex op voor een bepaalde branche.

    - Bepaalt dynamisch hoe de periodekolom heet (Periods / Perioden / …)
    - Filtert server-side op periode + branche, zodat we ruim onder de 10.000-recordlimiet blijven.
    """
    start, end = _ym_range(months_back)

    # 0) Bepaal periodeveld via een klein probe-request
    try:
        probe = _odata_select(dataset, "TypedDataSet", "$top=1")
    except requests.HTTPError:
        probe = []

    if not probe:
        return []

    period_col = _pick_period_field(probe[0])

    # 1) Haal branch-dimensie op → probeer server-side met Key
    dim_name, branches = list_retail_branches(dataset)
    branch_key = _find_branch_key(branches, branch_code_or_title) if branches else None

    rows: List[Dict] = []

    if dim_name and branch_key:
        # strikte filter: periode + branche
        try:
            rows = _odata_select(
                dataset,
                "TypedDataSet",
                (
                    f"$filter={period_col} ge '{start}' and {period_col} le '{end}' "
                    f"and {dim_name} eq '{branch_key}'&$top=5000"
                ),
            )
        except requests.HTTPError:
            rows = []

    # 2) Fallback: alleen periode-range, daarna client-side filteren op branch
    if not rows:
        try:
            rows = _odata_select(
                dataset,
                "TypedDataSet",
                f"$filter={period_col} ge '{start}' and {period_col} le '{end}'&$top=5000",
            )
        except requests.HTTPError:
            return []
        if not rows:
            return []

        # zoek branchveld
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
                return any(
                    val == str(b["key"]).lower() or val == str(b["title"]).lower()
                    for b in branches
                )
            # laatste escape: string-match op meegegeven branch_code_or_title
            return branch_code_or_title.lower() in val

        rows = [r for r in rows if _match(r)]
        if not rows:
            return []

        value_field = _pick_numeric_field(rows[0], [series])
    else:
        value_field = _pick_numeric_field(rows[0], [series])

    out: List[Dict] = []
    for it in rows:
        raw = it.get(value_field)
        if raw is None:
            continue
        val = float(raw) if isinstance(raw, (int, float)) else float(str(raw).replace(",", "."))
        out.append(
            {
                "period": it[period_col],
                "retail_value": val,
                "series": value_field,
                "branch": branch_code_or_title,
            }
        )
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
        "note": "Demo-indices op basis van CBS. Vervang deze stub later door een echte postcode4-koppeling.",
    }
