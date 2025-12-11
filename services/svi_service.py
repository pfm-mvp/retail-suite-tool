# services/svi_service.py

import numpy as np
import pandas as pd

# -------------------------------------------------------------
# HELPER: Safe normalization (0–100)
# -------------------------------------------------------------
def _normalize(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return pd.Series([50] * len(s), index=s.index)  # neutral baseline
    min_v, max_v = s.min(), s.max()
    if max_v - min_v == 0:
        return pd.Series([50] * len(s), index=s.index)
    return ((s - min_v) / (max_v - min_v) * 100).clip(0, 100)


# -------------------------------------------------------------
# MAIN FUNCTION: Build full Store Vitality Index (SVI)
# -------------------------------------------------------------
def compute_svi(
    df_period: pd.DataFrame,
    region_shops: pd.DataFrame,
    store_key_col: str,
    capture_weekly: pd.DataFrame,
    cbs_retail_month: pd.DataFrame,
    cci_df: pd.DataFrame,
):
    """
    Bouwt de Store Vitality Index (SVI) 0–100 op basis van 8 pijlers.
    Werkt met missende data (fallback logic).
    """

    # ---------------------------------------------------------
    # 1. Aggregatie per winkel (basis KPI's)
    # ---------------------------------------------------------
    group_cols = [store_key_col]
    agg = (
        df_period.groupby(group_cols)
        .agg({
            "footfall": "sum",
            "turnover": "sum",
            "sales_per_visitor": "mean",
            "sqm_effective": "max",
        })
        .reset_index()
    )

    # omzet/m²
    agg["turnover_per_sqm"] = np.where(
        agg["sqm_effective"] > 0,
        agg["turnover"] / agg["sqm_effective"],
        np.nan,
    )

    # Store display labels
    name_map = {}
    for _, row in region_shops.iterrows():
        name_map[int(row["id"])] = row.get("store_display", str(row["id"]))

    agg["store_name"] = agg[store_key_col].map(name_map)

    # ---------------------------------------------------------
    # 2. Street traffic + Capture Rate integratie
    # ---------------------------------------------------------
    capture_stats = capture_weekly.groupby(store_key_col).agg({
        "capture_rate": "mean"
    }).rename(columns={"capture_rate": "avg_capture_rate"})

    agg = agg.merge(capture_stats, on=store_key_col, how="left")

    # ---------------------------------------------------------
    # 3. Macro integratie (CCI en CBS retail)
    # ---------------------------------------------------------
    if not cbs_retail_month.empty:
        agg["macro_cbs"] = cbs_retail_month["cbs_retail_index"].iloc[-1]
    else:
        agg["macro_cbs"] = np.nan

    if not cci_df.empty:
        agg["macro_cci"] = cci_df["cci_index"].iloc[-1]
    else:
        agg["macro_cci"] = np.nan

    # ---------------------------------------------------------
    # 4. Pijlers bouwen (0–100)
    # ---------------------------------------------------------

    # Commercial Output
    agg["p_commercial"] = _normalize(
        agg["turnover"] * 0.5 +
        agg["turnover_per_sqm"] * 0.3 +
        agg["sales_per_visitor"] * 0.2
    )

    # Market Power (footfall + capture rate)
    agg["p_market"] = _normalize(
        agg["footfall"] * 0.6 +
        agg["avg_capture_rate"].fillna(0) * 0.4
    )

    # Customer Value (SPV)
    agg["p_customer"] = _normalize(agg["sales_per_visitor"])

    # Space Efficiency
    agg["p_space"] = _normalize(agg["turnover_per_sqm"])

    # Macro Resilience
    agg["p_macro"] = _normalize(
        agg["macro_cbs"].fillna(100) * 0.5 +
        agg["macro_cci"].fillna(100) * 0.5
    )

    # ---------------------------------------------------------
    # 5. EIND SCORE (weighted)
    # ---------------------------------------------------------
    agg["svi_score"] = (
        agg["p_commercial"] * 0.40 +
        agg["p_market"] * 0.20 +
        agg["p_customer"] * 0.15 +
        agg["p_space"] * 0.10 +
        agg["p_macro"] * 0.15
    )

    # ---------------------------------------------------------
    # 6. Classificatie
    # ---------------------------------------------------------
    def classify(score):
        if score >= 75:
            return "High performance", "🟢"
        elif score >= 60:
            return "Good / stable", "🟠"
        elif score >= 45:
            return "Attention required", "🟠"
        else:
            return "Under pressure", "🔴"

    agg["status"], agg["icon"] = zip(*agg["svi_score"].apply(classify))

    # ---------------------------------------------------------
    # 7. Opportunity Engine (€)
    # ---------------------------------------------------------
    pot = []

    median_tps = agg["turnover_per_sqm"].median(skipna=True)

    for _, row in agg.iterrows():
        if pd.isna(row["turnover_per_sqm"]) or pd.isna(row["sqm_effective"]):
            pot.append(0)
            continue
        ideal_rev = median_tps * row["sqm_effective"]
        pot.append(max(0, ideal_rev - row["turnover"]))

    agg["profit_potential"] = pot
    agg["profit_potential_year"] = agg["profit_potential"] * 12

    return agg
