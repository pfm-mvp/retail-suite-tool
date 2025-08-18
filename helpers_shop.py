# helpers_shop.py  — uniforme helpers voor winkels & regio's
from __future__ import annotations

try:
    # Verwacht: shop_mapping.py in de project-root
    from shop_mapping import SHOP_NAME_MAP as RAW_MAP
except Exception:
    RAW_MAP = {}

def _normalize_id(val):
    try:
        return int(val)
    except Exception:
        return val

def _normalized_id_to_name(raw: dict) -> dict[int, str]:
    """
    Ondersteunt:
      {id: "Naam"}  of  {id: {"name": "Naam", "region": "...", ...}}
    """
    if not isinstance(raw, dict):
        return {}

    result: dict[int, str] = {}
    for sid, meta in raw.items():
        sid_norm = _normalize_id(sid)
        if isinstance(meta, dict):
            name = (
                meta.get("name")
                or meta.get("shop_name")
                or meta.get("title")
                or f"Shop {sid_norm}"
            )
        else:
            # string / iets anders
            name = str(meta) if meta is not None else f"Shop {sid_norm}"
        result[sid_norm] = name
    return result

def _id_to_region(raw: dict) -> dict[int, str]:
    """Region uit nieuw formaat; anders 'All'."""
    regions: dict[int, str] = {}
    for sid, meta in raw.items():
        sid_norm = _normalize_id(sid)
        reg = "All"
        if isinstance(meta, dict):
            reg = (
                meta.get("region")
                or meta.get("regio")
                or "All"
            )
        regions[sid_norm] = reg
    return regions

# --------- Publieke objecten ----------
ID_TO_NAME   = _normalized_id_to_name(RAW_MAP)        # {id -> naam}
NAME_TO_ID   = {name: sid for sid, name in ID_TO_NAME.items()}  # {naam -> id}
ID_TO_REGION = _id_to_region(RAW_MAP)                  # {id -> regio}
REGIONS      = ["All"] + sorted(
    {r for r in ID_TO_REGION.values() if r and r != "All"}
)

# Backwards compat: sommige pagina’s importeerden SHOP_NAME_MAP als {id->naam}
SHOP_NAME_MAP = ID_TO_NAME

def get_ids_by_region(region: str | None):
    """Alle shop_ids in regio; bij 'All'/None: alle winkels."""
    if not region or region == "All":
        return list(ID_TO_NAME.keys())
    return [sid for sid, r in ID_TO_REGION.items() if r == region]

def get_name_by_id(shop_id: int) -> str | None:
    return ID_TO_NAME.get(shop_id)

def get_region_by_id(shop_id: int) -> str | None:
    return ID_TO_REGION.get(shop_id)
