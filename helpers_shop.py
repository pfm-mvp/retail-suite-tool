# helpers_shop.py  — werkt met oud én nieuw shop_mapping schema
from __future__ import annotations
from typing import Dict, List, Optional
from shop_mapping import SHOP_NAME_MAP as RAW_MAP

# Intern uniform model: {id: {"name": str, "region": str}}
def _normalize(raw: Dict) -> Dict[int, Dict[str, str]]:
    norm: Dict[int, Dict[str, str]] = {}
    for sid, val in raw.items():
        sid_int = int(sid)
        if isinstance(val, dict):
            name = val.get("name") or str(sid_int)
            region = val.get("region") or "All"
            norm[sid_int] = {"name": name, "region": region}
        else:
            # oud schema: alleen naam (string)
            norm[sid_int] = {"name": str(val), "region": "All"}
    return norm

_MAP = _normalize(RAW_MAP)

# Publieke mappings
ID_TO_NAME: Dict[int, str]    = {sid: meta["name"]   for sid, meta in _MAP.items()}
NAME_TO_ID: Dict[str, int]    = {meta["name"]: sid   for sid, meta in _MAP.items()}
ID_TO_REGION: Dict[int, str]  = {sid: meta["region"] for sid, meta in _MAP.items()}

# Unieke regio’s (minstens "All")
REGIONS: List[str] = sorted(set(ID_TO_REGION.values())) or ["All"]

def get_ids_by_region(region: str) -> List[int]:
    """Alle shop_ids die bij een regio horen. 'All' → alle ids."""
    if region == "All":
        return list(_MAP.keys())
    return [sid for sid, meta in _MAP.items() if meta["region"] == region]

def get_region_by_id(shop_id: int) -> Optional[str]:
    return ID_TO_REGION.get(int(shop_id))

def get_name_by_id(shop_id: int) -> Optional[str]:
    return ID_TO_NAME.get(int(shop_id))

def get_id_by_name(name: str) -> Optional[int]:
    return NAME_TO_ID.get(name)