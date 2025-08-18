# helpers_shop.py
from typing import Dict, List, Any

try:
    # Altijd uit de root importeren
    from shop_mapping import SHOP_NAME_MAP as RAW_MAP
except Exception:
    RAW_MAP = {}

def _as_int(x) -> int | None:
    try:
        return int(x)
    except Exception:
        return None

def _get_name(meta: Any) -> str | None:
    # Ondersteun oud {id: "Naam"} en nieuw {id: {"name": "...", "region": "..."}}
    if isinstance(meta, dict):
        for key in ["name", "Name", "shop_name", "title", "city"]:
            if key in meta and isinstance(meta[key], str) and meta[key].strip():
                return meta[key].strip()
        return None
    if isinstance(meta, str) and meta.strip():
        return meta.strip()
    return None

def _get_region(meta: Any) -> str | None:
    if isinstance(meta, dict):
        reg = meta.get("region") or meta.get("Region")
        if isinstance(reg, str) and reg.strip():
            return reg.strip()
    return None

def normalize_shop_map(raw: dict) -> tuple[Dict[int, str], Dict[str, int], Dict[int, str], List[str]]:
    id_to_name: Dict[int, str] = {}
    name_to_id: Dict[str, int] = {}
    id_to_region: Dict[int, str] = {}
    regions_set = set()

    if not isinstance(raw, dict) or not raw:
        return {}, {}, {}, []

    for sid, meta in raw.items():
        sid_int = _as_int(sid)
        if sid_int is None:
            continue
        name = _get_name(meta)
        if not name:
            continue
        id_to_name[sid_int] = name
        # Als dubbele namen bestaan, laatste wint – dat is ok voor demo
        name_to_id[name] = sid_int

        reg = _get_region(meta)
        if reg:
            id_to_region[sid_int] = reg
            regions_set.add(reg)

    return id_to_name, name_to_id, id_to_region, sorted(regions_set)

ID_TO_NAME, NAME_TO_ID, ID_TO_REGION, REGIONS = normalize_shop_map(RAW_MAP)

def get_ids_by_region(region: str) -> List[int]:
    if not region or region == "ALL":
        return list(ID_TO_NAME.keys())
    return [sid for sid, reg in ID_TO_REGION.items() if reg == region]

def get_region_by_id(shop_id: int) -> str | None:
    return ID_TO_REGION.get(shop_id)

def get_name_by_id(shop_id: int) -> str | None:
    return ID_TO_NAME.get(shop_id)

# Voor snelle sanity-checks bij debugging (optioneel)
if __name__ == "__main__":
    print("Shops loaded:", len(ID_TO_NAME))
    # print(ID_TO_NAME)
