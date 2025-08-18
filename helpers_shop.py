# helpers_shop.py
# Backwards-compatible helper voor winkelmetadata + regio’s

try:
    from shop_mapping import SHOP_NAME_MAP as RAW_MAP   # kan oud of nieuw formaat zijn
except Exception:
    RAW_MAP = {}

def _normalize(raw: dict) -> dict:
    """
    Normaliseer naar: { shop_id: {"name": str, "region": str} }
    - Oud formaat: {id: "Naam"}  -> {"name": "Naam", "region": "All"}
    - Nieuw formaat: laat "name" en "region" intact, default "region" = "All"
    """
    norm = {}
    for sid, meta in (raw or {}).items():
        if isinstance(meta, dict):
            name = meta.get("name")
            region = meta.get("region") or "All"
        else:
            name = str(meta) if meta is not None else None
            region = "All"
        if name:
            norm[sid] = {"name": name, "region": region}
    return norm

SHOP_META = _normalize(RAW_MAP)             # genormaliseerde bron
# Handige exports:
ID_TO_NAME   = {sid: m["name"]   for sid, m in SHOP_META.items()}
NAME_TO_ID   = {m["name"]: sid   for sid, m in SHOP_META.items()}
ID_TO_REGION = {sid: m["region"] for sid, m in SHOP_META.items()}

# Regio-lijst (inclusief "All" als fallback)
REGIONS = sorted(set(ID_TO_REGION.values())) or ["All"]

def get_ids_by_region(region: str):
    if region == "All":
        return list(ID_TO_NAME.keys())
    return [sid for sid, r in ID_TO_REGION.items() if r == region]

def get_region_by_id(shop_id: int):
    return ID_TO_REGION.get(shop_id, "All")

def get_name_by_id(shop_id: int):
    return ID_TO_NAME.get(shop_id)
