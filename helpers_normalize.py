
# helpers_normalize.py
from __future__ import annotations

from typing import Dict, Any, Iterable, List, Optional
import pandas as pd
from datetime import datetime

KPI_KEYS_DEFAULT = [
    "count_in",
    "turnover",
    "conversion_rate",
    "sales_per_visitor",
    "inside",
    "count_out",
]

META_FIELDS = {
    "id", "parent_id", "parent_data_id", "company_id", "segment_id",
    "custom_location_id", "level", "name", "data_type"
}

def _coerce_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _extract_metrics(data_node: Dict[str, Any], kpi_keys: Iterable[str]) -> Dict[str, float]:
    out = {}
    for k in kpi_keys:
        if k in data_node:
            out[k] = _coerce_float(data_node.get(k))
    for k, v in data_node.items():
        if k in META_FIELDS or k in out:
            continue
        fv = _coerce_float(v)
        if fv is not None:
            out[k] = fv
    return out

def normalize_vemcount_response(resp_json: Dict[str, Any],
                                shop_name_map: Optional[Dict[int, str]] = None,
                                kpi_keys: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Flatten both day-level and timestamp-level nodes to a tidy DataFrame."""
    if not isinstance(resp_json, dict) or "data" not in resp_json:
        return pd.DataFrame()

    kpi_keys = list(kpi_keys) if kpi_keys is not None else KPI_KEYS_DEFAULT
    rows: List[Dict[str, Any]] = []
    data_block = resp_json.get("data", {})

    # possible shapes:
    #  A) {"data":{"date_YYYY-MM-DD": {"<shop_id>":{"data":{...}}}}}
    #  B) {"data":{"date_YYYY-MM-DD": {"<shop_id>":{"dates":{"YYYY-MM-DD HH:MM":{"data":{...}}}}}}}
    for bucket_key, shops_dict in data_block.items():
        date_str = bucket_key.replace("date_", "", 1) if str(bucket_key).startswith("date_") else None
        if not isinstance(shops_dict, dict):
            continue

        for shop_id_str, shop_node in shops_dict.items():
            try:
                shop_id = int(shop_id_str)
            except Exception:
                continue
            shop_name = shop_name_map.get(shop_id) if shop_name_map else None
            if not isinstance(shop_node, dict):
                continue

            if "dates" in shop_node and isinstance(shop_node["dates"], dict):
                for ts, node in shop_node["dates"].items():
                    data_node = node.get("data", {}) if isinstance(node, dict) else {}
                    metrics = _extract_metrics(data_node, kpi_keys)
                    ts_str = str(ts)
                    try:
                        date_iso = datetime.fromisoformat(ts_str.replace(" ", "T")).date().isoformat()
                    except Exception:
                        date_iso = date_str
                    row = {"date": date_iso, "timestamp": ts_str, "shop_id": shop_id, "shop_name": shop_name}
                    row.update(metrics)
                    rows.append(row)

            if "data" in shop_node and isinstance(shop_node["data"], dict):
                data_node = shop_node["data"]
                metrics = _extract_metrics(data_node, kpi_keys)
                row = {"date": date_str, "timestamp": None, "shop_id": shop_id, "shop_name": shop_name}
                row.update(metrics)
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    present_kpis = [k for k in (kpi_keys or []) if k in df.columns]
    extra_kpis = [c for c in df.columns if c not in {"date","timestamp","shop_id","shop_name"} and c not in present_kpis]
    ordered_cols = ["date","timestamp","shop_id","shop_name"] + present_kpis + extra_kpis
    df = df.reindex(columns=ordered_cols)
    df = df.sort_values(by=["date","shop_id","timestamp"], kind="stable", na_position="last").reset_index(drop=True)
    return df
