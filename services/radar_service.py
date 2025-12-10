# services/radar_service.py

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _safe_median(s: pd.Series) -> float:
    """
    Robuuste median: converteert naar float, negeert NaN.
    """
    s_num = pd.to_numeric(s, errors="coerce")
    s_num = s_num.dropna()
    if s_num.empty:
        return np.nan
    return float(s_num.median())


def _index_vs_median(values: pd.Series, median_value: float) -> pd.Series:
    """
    Maakt een index  = value / median * 100.
    Median 0 of NaN → alles NaN.
    """
    v = pd.to_numeric(values, errors="coerce").astype(float)
    if not np.isfinite(median_value) or median_value == 0:
        return pd.Series(np.nan, index=v.index)
    return v / median_value * 100.0


def _status_from_score(score: float) -> tuple[str, str]:
    """
    Converteer de samengestelde score naar status + icoon.

    - >= 115 → groen
    - 95–115 → oranje
    - < 95  → rood
    """
    if pd.isna(score):
        return "unknown", "⚪"

    if score >= 115:
        return "green", "🟢"
    if score <= 95:
        return "red", "🔴"
    return "orange", "🟠"


def _macro_factor(
    cbs_retail_month: Optional[pd.DataFrame],
    cci_df: Optional[pd.DataFrame],
) -> tuple[float, str]:
    """
    Berekent een eenvoudige macro-factor obv:
    - CBS detailhandelindex (cbs_retail_index)
    - CCI index (cci_index)

    Factor wordt voor alle winkels in de regio gelijk toegepast.
    Dit verandert dus niet de onderlinge ranking, maar wel de "wind"
    (macro meewind / tegenwind) in de score en toelichting.
    """
    factor = 1.0
    comments: list[str] = []

    try:
        if (
            cbs_retail_month is not None
            and not cbs_retail_month.empty
            and "cbs_retail_index" in cbs_retail_month.columns
        ):
            last_cbs = (
                pd.to_numeric(
                    cbs_retail_month["cbs_retail_index"], errors="coerce"
                ).dropna()
            )
            if not last_cbs.empty:
                last_val = float(last_cbs.iloc[-1])
                if last_val < 95:
                    factor *= 0.97
                    comments.append("detailhandelindex ligt onder het niveau van de startperiode")
                elif last_val > 105:
                    factor *= 1.03
                    comments.append("detailhandelindex ligt duidelijk boven het niveau van de startperiode")

        if (
            cci_df is not None
            and not cci_df.empty
            and "cci_index" in cci_df.columns
        ):
            last_cci = (
                pd.to_numeric(cci_df["cci_index"], errors="coerce").dropna()
            )
            if not last_cci.empty:
                last_val = float(last_cci.iloc[-1])
                if last_val < 95:
                    factor *= 0.97
                    comments.append("consumentenvertrouwen is laag")
                elif last_val > 105:
                    factor *= 1.03
                    comments.append("consumentenvertrouwen is hoog")
    except Exception:
        # Macro-factor is nice-to-have; nooit het hele radar-model laten crashen.
        pass

    macro_comment = ""
    if comments:
        macro_comment = "Macro-context: " + ", ".join(comments) + "."

    return factor, macro_comment


def _capture_summary(capture_weekly: Optional[pd.DataFrame]) -> str:
    """
    Korte tekstuele duiding van regio-capture rate (Pathzz).
    """
    if capture_weekly is None or capture_weekly.empty:
        return ""

    if "capture_rate" not in capture_weekly.columns:
        return ""

    try:
        cap = float(
            pd.to_numeric(capture_weekly["capture_rate"], errors="coerce").dropna().mean()
        )
    except Exception:
        return ""

    if not np.isfinite(cap):
        return ""

    if cap < 10:
        return f"Regio capture rate is laag (~{cap:.1f}%), er blijft veel straattraffic onbenut."
    if cap > 25:
        return f"Regio capture rate is hoog (~{cap:.1f}%), winkels profiteren goed van straattraffic."
    return f"Regio capture rate is gemiddeld (~{cap:.1f}%)."


def _headline_from_status(status: str) -> str:
    if status == "green":
        return "Gaat goed"
    if status == "orange":
        return "Heeft aandacht nodig"
    if status == "red":
        return "Presteert onder verwachting"
    return "Onbekende status"


def _build_reason_row(
    row: pd.Series,
    macro_comment: str,
    capture_comment: str,
) -> str:
    """
    Bouwt een korte, krachtige toelichting per winkel op basis van de indices.
    """
    issues: list[str] = []
    positives: list[str] = []

    def add_metric(idx_col: str, label: str):
        val = row.get(idx_col, np.nan)
        if pd.isna(val):
            return
        try:
            val_f = float(val)
        except Exception:
            return

        if val_f < 90:
            issues.append(f"{label} ligt ~{100 - val_f:.0f}% onder regiomedian")
        elif val_f > 110:
            positives.append(f"{label} ligt ~{val_f - 100:.0f}% boven regiomedian")

    add_metric("idx_turnover", "Omzet")
    add_metric("idx_tps", "Omzet per m²")
    add_metric("idx_spv", "Besteding per bezoeker")
    add_metric("idx_footfall", "Footfall")

    parts: list[str] = []

    if issues:
        parts.append("; ".join(issues[:2]))
    elif positives:
        parts.append("; ".join(positives[:2]))

    if capture_comment:
        parts.append(capture_comment)

    if macro_comment:
        parts.append(macro_comment)

    if not parts:
        return "Prestaties liggen globaal rond het regiogemiddelde."

    return " ".join(parts)


def build_region_store_radar(
    df_period: pd.DataFrame,
    region_shops: pd.DataFrame,
    store_key_col: str,
    capture_weekly: Optional[pd.DataFrame] = None,
    cbs_retail_month: Optional[pd.DataFrame] = None,
    cci_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Bouwt een samengestelde 'radar'-score per winkel in de regio.

    Input:
    - df_period: dagdata met minimaal kolommen:
        - date
        - footfall
        - turnover
        - (optioneel) sales_per_visitor
        - (optioneel) sqm_effective
        - store-key (= store_key_col, bv. 'id')
    - region_shops: DataFrame met o.a. 'id', 'store_display', 'region', 'sqm_effective'
    - store_key_col: naam van de kolom in df_period met winkel-ID
    - capture_weekly: regio-weekdata met 'capture_rate' (optioneel)
    - cbs_retail_month: DataFrame met 'cbs_retail_index' (optioneel)
    - cci_df: DataFrame met 'cci_index' (optioneel)

    Output:
    DataFrame met per winkel:
    - store_id
    - store_name
    - region
    - footfall
    - turnover
    - sales_per_visitor
    - sqm_effective
    - turnover_per_sqm
    - idx_* (indices vs regiomedian)
    - radar_score (samengestelde index)
    - radar_status ('green' / 'orange' / 'red')
    - radar_icon (🟢 / 🟠 / 🔴)
    - headline
    - short_reason
    """
    if df_period is None or df_period.empty:
        return pd.DataFrame()

    if store_key_col is None or store_key_col not in df_period.columns:
        return pd.DataFrame()

    df = df_period.copy()

    # Zorg dat de basis-KPI's bestaan
    required_cols = {"turnover", "footfall"}
    if not required_cols.issubset(df.columns):
        return pd.DataFrame()

    # sales_per_visitor indien nodig opnieuw berekenen
    if "sales_per_visitor" not in df.columns:
        df["sales_per_visitor"] = np.where(
            df["footfall"] > 0,
            df["turnover"] / df["footfall"],
            np.nan,
        )

    # sqm_effective meenemen indien beschikbaar in df; anders via region_shops
    if "sqm_effective" not in df.columns and "sqm_effective" in region_shops.columns:
        df = df.merge(
            region_shops[["id", "sqm_effective"]],
            left_on=store_key_col,
            right_on="id",
            how="left",
            suffixes=("", "_region"),
        )

    # Aggregatie per winkel over de gekozen periode
    agg_dict: dict[str, str] = {
        "footfall": "sum",
        "turnover": "sum",
        "sales_per_visitor": "mean",
    }
    if "sqm_effective" in df.columns:
        agg_dict["sqm_effective"] = "max"

    store_agg = df.groupby(store_key_col, as_index=False).agg(agg_dict)

    # Omzet per m²
    if "sqm_effective" in store_agg.columns:
        store_agg["turnover_per_sqm"] = np.where(
            store_agg["sqm_effective"] > 0,
            store_agg["turnover"] / store_agg["sqm_effective"],
            np.nan,
        )
    else:
        store_agg["sqm_effective"] = np.nan
        store_agg["turnover_per_sqm"] = np.nan

    # Medianen voor indices
    med_turnover = _safe_median(store_agg["turnover"])
    med_footfall = _safe_median(store_agg["footfall"])
    med_spv = _safe_median(store_agg["sales_per_visitor"])
    med_tps = _safe_median(store_agg["turnover_per_sqm"])

    store_agg["idx_turnover"] = _index_vs_median(store_agg["turnover"], med_turnover)
    store_agg["idx_footfall"] = _index_vs_median(store_agg["footfall"], med_footfall)
    store_agg["idx_spv"] = _index_vs_median(
        store_agg["sales_per_visitor"], med_spv
    )
    store_agg["idx_tps"] = _index_vs_median(
        store_agg["turnover_per_sqm"], med_tps
    )

    # Samengestelde score (zwaarste weging op omzet per m² + SPV)
    weights = {
        "idx_turnover": 0.30,
        "idx_tps": 0.30,
        "idx_spv": 0.25,
        "idx_footfall": 0.15,
    }

    radar_raw = np.zeros(len(store_agg), dtype=float)
    for metric, w in weights.items():
        if metric in store_agg.columns:
            radar_raw += store_agg[metric].fillna(100.0).astype(float) * w

    # Macro-factor (CCI + CBS detailhandelindex)
    macro_factor, macro_comment = _macro_factor(
        cbs_retail_month=cbs_retail_month, cci_df=cci_df
    )
    capture_comment = _capture_summary(capture_weekly)

    # Radar-score met lichte clipping; baseline ~100
    store_agg["radar_raw"] = radar_raw
    store_agg["radar_score"] = store_agg["radar_raw"] * macro_factor
    store_agg["radar_score"] = store_agg["radar_score"].clip(lower=60, upper=140)

    # Status + icoon
    statuses = store_agg["radar_score"].apply(_status_from_score)
    store_agg["radar_status"] = statuses.apply(lambda t: t[0])
    store_agg["radar_icon"] = statuses.apply(lambda t: t[1])
    store_agg["headline"] = store_agg["radar_status"].apply(_headline_from_status)

    # Namen & regio eraan hangen
    name_map = {}
    region_map = {}
    if not region_shops.empty:
        for _, r in region_shops.iterrows():
            sid = r.get("id", np.nan)
            if pd.isna(sid):
                continue
            sid_int = int(sid)
            name_map[sid_int] = r.get("store_display", str(sid_int))
            region_map[sid_int] = r.get("region", "")

    store_agg["store_id"] = store_agg[store_key_col]
    store_agg["store_name"] = store_agg["store_id"].map(
        lambda x: name_map.get(int(x), str(x)) if pd.notna(x) else "Onbekend"
    )
    store_agg["region"] = store_agg["store_id"].map(
        lambda x: region_map.get(int(x), "") if pd.notna(x) else ""
    )

    # Korte reden per winkel
    store_agg["short_reason"] = store_agg.apply(
        lambda row: _build_reason_row(
            row=row,
            macro_comment=macro_comment,
            capture_comment=capture_comment,
        ),
        axis=1,
    )

    # Kolommen ordenen
    cols_order = [
        "store_id",
        "store_name",
        "region",
        "radar_score",
        "radar_status",
        "radar_icon",
        "headline",
        "short_reason",
        "turnover",
        "footfall",
        "sales_per_visitor",
        "sqm_effective",
        "turnover_per_sqm",
        "idx_turnover",
        "idx_footfall",
        "idx_spv",
        "idx_tps",
        "radar_raw",
    ]

    cols_existing = [c for c in cols_order if c in store_agg.columns]
    result = store_agg[cols_existing].copy()

    # Slechtste winkels eerst, zodat de regiomanager direct ziet waar de pijn zit
    result = result.sort_values("radar_score", ascending=True).reset_index(drop=True)

    return result
