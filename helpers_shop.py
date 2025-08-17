from shop_mapping import SHOP_NAME_MAP

# Id → naam
ID_TO_NAME = {sid: meta["name"] for sid, meta in SHOP_NAME_MAP.items()}
# Naam → id
NAME_TO_ID = {meta["name"]: sid for sid, meta in SHOP_NAME_MAP.items()}
# Id → regio
ID_TO_REGION = {sid: meta["region"] for sid, meta in SHOP_NAME_MAP.items()}
# Alle regio’s
REGIONS = sorted(set(meta["region"] for meta in SHOP_NAME_MAP.values()))

def get_ids_by_region(region: str):
    """Geeft alle shop_ids die bij een regio horen."""
    return [sid for sid, meta in SHOP_NAME_MAP.items() if meta["region"] == region]

def get_region_by_id(shop_id: int):
    return SHOP_NAME_MAP.get(shop_id, {}).get("region", None)

def get_name_by_id(shop_id: int):
    return SHOP_NAME_MAP.get(shop_id, {}).get("name", None)