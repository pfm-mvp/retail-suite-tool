# helpers_shop.py — robust & backward compatible
from __future__ import annotations
from typing import Dict, List, Optional

# --- Probeer verschillende bestandsnamen voor de mapping ---
_RAW_MAP = None
_IMPORT_ERR = None
for modname in ("shop_mapping", "store_mapping", "shopping_map"):
    try:
        mod = __import__(modname, fromlist=["SHOP_NAME_MAP"])
        if hasattr(mod, "SHOP_NAME_MAP"):
            _RAW_MAP = getattr(mod, "SHOP_NAME_MAP")
            break
    except Exception as e:
        _IMPORT_ERR = e

if _RAW_MAP is None:
    # Geen mapping gevonden: geef heldere fout voor Streamlit
    try:
        import streamlit as st
        st.error(
            "Kan geen `SHOP_NAME_MAP` vinden. Plaats een bestand `shop_mapping.py` (of `store_mapping.py`/`shopping_map.py`) "
            "in de root met bijvoorbeeld:\n\n"
            "SHOP_NAME_MAP = {\n"
            "    32224: 'Amersfoort',\n"
            "    31977: 'Amsterdam',\n"
            "}\n\n"
            "Of gebruik het nieuwe schema:\n"
            "SHOP_NAME_MAP = {\n"
            "    32224: {'name': 'Amersfoort', 'region': 'Utrecht'},\n"
            "}\n"
        )
    except Exception:
        pass
    # Val terug op lege mapping om import niet hard te laten crashen
    _RAW_MAP = {}

# --- Normaliseer naar uniform intern model ---
# Intern format: {id: {"name": str, "region": str}}
def _normalize(raw: Dict) -> Dict[int, Dict[str, str]]:
    norm: Dict[int, Dict[str, str]] = {}
    if not isinstance(raw, dict):
        return norm
    for sid, val in raw.items():
        try:
            sid_int = int(sid)
        except Exception:
            continue
        if isinstance(val, dict):
            name = val.get("name") or str(sid_int)
            region = val.get("region") or "All"
            norm[sid_int] = {"name": name, "region": region}
        else:
            # oud schema: alleen naam (string)
            norm[sid_int] = {"name": str(val), "region": "All"}
    return norm

_MAP = _normalize(_RAW_MAP)

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