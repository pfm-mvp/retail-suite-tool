# services/svi_service.py

import numpy as np
import pandas as pd


def _normalize(series: pd.Series) -> pd.Series:
    """
    Normaliseer naar 0–100.
    Als er geen variatie is (of alleen NaN), dan alles op 50 = neutraal.
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return pd.Series([50] * len(s), index=s.index)
    min_v, max_v = s.min(), s.max()
    if max_v - min_v == 0:
        return pd.Series([50] * len(s), index=s.index)
    return ((s - min_v) / (max_v - min_v) * 100).clip(0, 100)

def sqm_calibration_factor(store_sqm: float, benchmark_sqm_series: pd.Series) -> float:
    """
    Calibrate benchmark based on relative store size.
    Returns factor ~1.0 for median-sized stores.
    """
    if pd.isna(store_sqm) or benchmark_sqm_series.dropna().empty:
        return 1.0

    median_sqm = benchmark_sqm_series.median()
    if median_sqm <= 0:
        return 1.0

    return float(store_sqm / median_sqm)

def build_store_vitality(
    df_period: pd.DataFrame,
    region_shops: pd.DataFrame,
    store_key_col: str,
) -> pd.DataFrame:
    """
    Bouwt per winkel een Store Vitality Index (SVI, 0–100) + footfall/capture index
    en omzetpotentieel.

    Verwacht:
    - df_period: dagdata met kolommen:
        - store_key_col (bijv. 'id')
        - footfall
        - turnover
        - sales_per_visitor (optioneel, mag NaN zijn)
        - sqm_effective (optioneel, mag NaN zijn)
    - region_shops: mapping met per winkel:
        - id
        - store_display
        - sqm_effective (optioneel)

    Returnt één rij per winkel met o.a.:
    - store_id
    - store_name
    - footfall, turnover, sales_per_visitor, sqm_effective
    - turnover_per_sqm
    - footfall_index_region (100 = regiomedian)
    - capture_index_region (100 = fair share vs m²-aandeel)
    - svi_score (0–100)
    - svi_status (tekst)
    - svi_icon (emoji)
    - reason_short (korte toelichting)
    - profit_potential_period (extra omzetpotentieel voor de geanalyseerde periode)
    """

    if df_period is None or df_period.empty:
        return pd.DataFrame()

    # ---------------------------
    # 1. Basis aggregatie per winkel
    # ---------------------------
    group_cols = [store_key_col]
    agg = (
        df_period.groupby(group_cols)
        .agg(
            footfall=("footfall", "sum"),
            turnover=("turnover", "sum"),
            sales_per_visitor=("sales_per_visitor", "mean"),
            sqm_effective=("sqm_effective", "max"),
        )
        .reset_index()
    )

    # Omzet per m²
    agg["turnover_per_sqm"] = np.where(
        (agg["sqm_effective"] > 0) & (~agg["sqm_effective"].isna()),
        agg["turnover"] / agg["sqm_effective"],
        np.nan,
    )

    # Store-namen erbij
    name_map = {}
    for _, row in region_shops.iterrows():
        try:
            sid = int(row["id"])
        except Exception:
            continue
        name_map[sid] = row.get("store_display", str(sid))

    agg["store_name"] = agg[store_key_col].map(name_map).fillna(
        agg[store_key_col].astype(str)
    )

    # ---------------------------
    # 2. Footfall index vs regio
    # ---------------------------
    median_footfall = agg["footfall"].median(skipna=True)
    if not np.isnan(median_footfall) and median_footfall > 0:
        agg["footfall_index_region"] = (
            agg["footfall"] / median_footfall * 100.0
        )
    else:
        agg["footfall_index_region"] = np.nan

    # ---------------------------
    # 3. Capture index vs regio (fair share op basis van m²)
    # ---------------------------
    total_footfall = agg["footfall"].sum(skipna=True)
    total_sqm = agg["sqm_effective"].sum(skipna=True)

    if total_footfall > 0 and total_sqm > 0:
        # expected_share = m²-aandeel
        exp_share = np.where(
            (agg["sqm_effective"] > 0) & (~agg["sqm_effective"].isna()),
            agg["sqm_effective"] / total_sqm,
            1.0 / max(len(agg), 1),
        )
        # actual_share = footfall-aandeel
        act_share = np.where(
            agg["footfall"] > 0,
            agg["footfall"] / total_footfall,
            0.0,
        )
        capture_idx = np.where(
            exp_share > 0,
            act_share / exp_share * 100.0,
            np.nan,
        )
        agg["capture_index_region"] = capture_idx
    else:
        agg["capture_index_region"] = np.nan

    # ---------------------------
    # 4. Pijlers (0–100)
    # ---------------------------

    # 4.1 Commercial output (omzet + omzet/m² + SPV)
    commercial_metric = (
        agg["turnover"].fillna(0) * 0.5
        + agg["turnover_per_sqm"].fillna(0) * 0.3
        + agg["sales_per_visitor"].fillna(0) * 0.2
    )
    agg["p_commercial"] = _normalize(commercial_metric)

    # 4.2 Market power (footfall + capture index)
    market_metric = (
        agg["footfall_index_region"].fillna(100) * 0.6
        + agg["capture_index_region"].fillna(100) * 0.4
    )
    agg["p_market"] = _normalize(market_metric)

    # 4.3 Customer value (SPV)
    agg["p_customer"] = _normalize(agg["sales_per_visitor"])

    # 4.4 Space efficiency (omzet/m²)
    agg["p_space"] = _normalize(agg["turnover_per_sqm"])

    # ---------------------------
    # 5. Eindscore (SVI 0–100)
    # ---------------------------
    agg["svi_score"] = (
        agg["p_commercial"] * 0.45
        + agg["p_market"] * 0.30
        + agg["p_customer"] * 0.15
        + agg["p_space"] * 0.10
    )

    # ---------------------------
    # 6. Status & icon + korte reason (0–100 schaal)
    # ---------------------------
    def classify(score: float):
        if pd.isna(score):
            return "Onbekend", "⚪"
        if score >= 75:
            return "High performance", "🟢"
        elif score >= 60:
            return "Good / stable", "🟠"
        elif score >= 45:
            return "Attention required", "🟠"
        else:
            return "Under pressure", "🔴"

    statuses = agg["svi_score"].apply(classify)
    agg["svi_status"] = statuses.map(lambda x: x[0])
    agg["svi_icon"] = statuses.map(lambda x: x[1])

    reasons = []
    for _, r in agg.iterrows():
        s = r["svi_score"]
        pc = r["p_commercial"]
        pm = r["p_market"]
        pv = r["p_customer"]
        ps = r["p_space"]

        if pd.isna(s):
            reasons.append("Te weinig data om een goede beoordeling te maken.")
            continue

        if s >= 75:
            if pc >= pm:
                reasons.append(
                    "Sterke omzet en m²-productiviteit; focus op vasthouden en premium beleving."
                )
            else:
                reasons.append(
                    "Sterke positie in traffic en marktaandeel; benut dit met hogere SPV."
                )
        elif s >= 60:
            if pv < 50:
                reasons.append(
                    "Boven regiogemiddelde, maar besteding per bezoeker blijft achter – focus op ATV/upsell."
                )
            else:
                reasons.append(
                    "Goede basis; optimaliseer traffic of m²-benutting voor extra groei."
                )
        elif s >= 45:
            if pm < 50:
                reasons.append(
                    "Onder regio op traffic/capture – meer instroom en zichtbaarheid nodig."
                )
            elif pc < 50:
                reasons.append(
                    "Omzet en m²-productiviteit onder regio – kijk naar assortiment en pricing."
                )
            else:
                reasons.append(
                    "Meerdere KPI’s net onder regio; gericht plan op traffic én SPV gewenst."
                )
        else:
            if pm < 50 and pc < 50:
                reasons.append(
                    "Structureel onder regio op traffic én omzet – herijk formule, team en marketingmix."
                )
            else:
                reasons.append(
                    "Significante achterstand; plan nodig op traffic, conversie en m²-benutting."
                )

    agg["reason_short"] = reasons

    # ---------------------------
    # 7. Omzetpotentieel (periode)
    # ---------------------------
    median_tps = agg["turnover_per_sqm"].median(skipna=True)
    potentials = []
    for _, r in agg.iterrows():
        if pd.isna(r["turnover_per_sqm"]) or pd.isna(r["sqm_effective"]):
            potentials.append(0.0)
            continue
        if np.isnan(median_tps) or median_tps <= 0:
            potentials.append(0.0)
            continue
        ideal_turnover = median_tps * r["sqm_effective"]
        delta = ideal_turnover - r["turnover"]
        potentials.append(max(0.0, float(delta)))

    agg["profit_potential_period"] = potentials

    return agg
