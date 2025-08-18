# helpers_shop.py
import os, sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from shop_mapping import SHOP_NAME_MAP as RAW_MAP  # kan oud of nieuw formaat zijn

# ---- Normaliseer naar: {id: {"name": str, "region": str}}
NORM = {}
for sid, val in RAW_MAP.items():
    if isinstance(val, dict):
        name = (val.get("name") or val.get("shop_name") or str(sid)).strip()
        region = (val.get("region") or "All").strip() or "All"
    else:
        name = str(val).strip()
        region = "All"
    NORM[int(sid)] = {"name": name, "region": region}

# ---- Publieke helpers
ID_TO_NAME   = {sid: meta["name"]   for sid, meta in NORM.items()}
NAME_TO_ID   = {meta["name"]: sid   for sid, meta in NORM.items()}
ID_TO_REGION = {sid: meta["region"] for sid, meta in NORM.items()}

# Altijd 'All' aanbieden + alfabetische lijst van overige regio’s
REGIONS = ["All"] + sorted({meta["region"] for meta in NORM.values() if meta["region"] and meta["region"] != "All"})

SHOP_IDS = list(ID_TO_NAME.keys())

def get_ids_by_region(region: str | None):
    """Geef alle shop_ids in regio. Bij None of 'All' → alle shops."""
    if not region or region == "All":
        return SHOP_IDS
    return [sid for sid, r in ID_TO_REGION.items() if r == region]

def get_region_by_id(shop_id: int):
    return ID_TO_REGION.get(int(shop_id), None)

def get_name_by_id(shop_id: int):
    return ID_TO_NAME.get(int(shop_id), None)
