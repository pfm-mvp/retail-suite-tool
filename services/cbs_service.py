def get_retail_index(series: str = "Omzetontwikkeling_1",
                     branch_code: str = "DH_TOTAAL",
                     months_back: int = 18,
                     dataset: str = "85828NED") -> List[Dict]:
    """
    Haal maanddata voor detailhandel op en filter client-side op branch_code.
    Robuust tegen variaties in kolomnamen (Branches_2 / Branches / Branche / etc.).
    """
    periods = _ym_list(months_back)
    quoted = ",".join([f"%27{p}%27" for p in periods])

    # 1) Haal alle rijen voor deze periodes op (zonder branch-filter om HTTP 4xx te voorkomen)
    try:
        rows = _odata_select(dataset, f"$filter=Periods in ({quoted})")
    except Exception:
        return []

    if not rows:
        return []

    # 2) Vind de periode- en branchekolom dynamisch
    period_field = _pick_period_field(rows[0])

    # brancheveld zoeken (naam varieert)
    branch_field = None
    for k in rows[0].keys():
        kl = k.lower()
        if "branch" in kl or "branches" in kl or "branche" in kl:
            branch_field = k
            break
    # Geen duidelijke branchkolom? Dan geen data tonen
    if branch_field is None:
        return []

    # 3) Waardekolom kiezen (serie), met fallback naar numeriek veld
    value_field = _pick_numeric_field(rows[0], [series])

    # 4) Client-side filteren op branch_code
    out = []
    for it in rows:
        if str(it.get(branch_field, "")).strip() != branch_code:
            continue
        raw = it.get(value_field, None)
        if raw is None:
            continue
        val = float(raw) if isinstance(raw, (int, float)) else float(str(raw).replace(",", "."))
        out.append({
            "period": it[period_field],
            "retail_value": val,
            "series": value_field,
            "branch": branch_code
        })

    out.sort(key=lambda x: x["period"])
    return out
