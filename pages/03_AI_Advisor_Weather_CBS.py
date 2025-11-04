def mom_yoy(dfm: pd.DataFrame):
    """
    Robuuste MoM/YoY op regiomaand-aggregaten.
    Werkt ook als ym geen string is of er maar 1 maand aanwezig is.
    """
    if dfm is None or dfm.empty:
        return {}

    m = dfm.copy().sort_values("ym").reset_index(drop=True)

    # Forceer ym -> datetime (1e dag vd maand). Als dit faalt, stop netjes.
    try:
        m["ym_dt"] = pd.to_datetime(m["ym"].astype(str) + "-01", errors="coerce")
    except Exception:
        m["ym_dt"] = pd.NaT

    if m["ym_dt"].isna().all():
        return {}

    last = m.iloc[-1]
    prev = m.iloc[-2] if len(m) > 1 else None

    # Zoek dezelfde maand vorig jaar
    yoy_dt = last["ym_dt"] - pd.DateOffset(years=1)
    yoy_match = m.loc[m["ym_dt"] == yoy_dt]
    yoy_row = yoy_match.iloc[0] if not yoy_match.empty else None

    def pct(a, b):
        if b in [0, None] or pd.isna(b):
            return None
        try:
            return (float(a) / float(b) - 1) * 100
        except Exception:
            return None

    return {
        "last_ym": last["ym_dt"].strftime("%Y-%m"),
        "visitors": float(last.get("visitors", 0)),
        "turnover": float(last.get("turnover", 0)),
        "conversion": float(last.get("conversion", 0)),
        "spv": float(last.get("spv", 0)),
        "mom": {
            "turnover": pct(last.get("turnover", 0), prev.get("turnover", 0)) if prev is not None else None,
            "visitors": pct(last.get("visitors", 0), prev.get("visitors", 0)) if prev is not None else None,
            "conversion": pct(last.get("conversion", 0), prev.get("conversion", 0)) if prev is not None else None,
            "spv": pct(last.get("spv", 0), prev.get("spv", 0)) if prev is not None else None,
        },
        "yoy": {
            "turnover": pct(last.get("turnover", 0), yoy_row.get("turnover", 0)) if yoy_row is not None else None,
            "visitors": pct(last.get("visitors", 0), yoy_row.get("visitors", 0)) if yoy_row is not None else None,
            "conversion": pct(last.get("conversion", 0), yoy_row.get("conversion", 0)) if yoy_row is not None else None,
            "spv": pct(last.get("spv", 0), yoy_row.get("spv", 0)) if yoy_row is not None else None,
        },
    }
