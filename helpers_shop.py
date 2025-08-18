# helpers_shop.py
import os, sys

# zorg dat de root zichtbaar is
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    # kan oud of nieuw formaat zijn
    from shop_mapping import SHOP_NAME_MAP as RAW_MAP
except Exception:
    RAW_MAP = {}

def _normalize(raw: dict):
    """Normaliseer naar: {id: {'name': str, 'region': str}} + lijst regio's."""
    out, regions = {}, set()
    if isinstance(raw, dict) and raw:
        sample = next(iter(raw.values()))
        if isinstance(sample, dict):  # nieuw formaat
            for sid, meta in raw.items():
                name   = meta.get("name") or meta.get("shop_name") or str(sid)
                region = meta.get("region") or "All"
                out[int(sid)] = {"name": name, "region": region}
                regions.add(region)
        else:  # oud formaat: {id: "Naam"}
            for sid, name in raw.items():
                out[int(sid)] = {"name": str(name), "region": "All"}
            regions.add("All")
    else:
        # lege / ontbrekende mapping → veilige default
        out, regions = {}, {"All"}
    return out, sorted(regions)

NORM_MAP, REGIONS_ONLY = _normalize(RAW_MAP)

# Publieke helpers
ID_TO_NAME   = {sid: meta["name"]   for sid, meta in NORM_MAP.items()}
NAME_TO_ID   = {meta["name"]: sid   for sid, meta in NORM_MAP.items()}
ID_TO_REGION = {sid: meta["region"] for sid, meta in NORM_MAP.items()}

# “All” altijd beschikbaar, en verder de echte regio’s
REGIONS = ["All"] + [r for r in REGIONS_ONLY if r != "All"]

def get_ids_by_region(region: str | None):
    """Geef alle shop_ids in een regio. ‘All’ of None ⇒ alles."""
    if not region or region == "All":
        return list(ID_TO_NAME.keys())
    return [sid for sid, reg in ID_TO_REGION.items() if reg == region]

def get_region_by_id(shop_id: int):
    return ID_TO_REGION.get(shop_id)

def get_name_by_id(shop_id: int):
    return ID_TO_NAME.get(shop_id)
