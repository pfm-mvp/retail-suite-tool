# helpers_shop.py  — tolerant voor oud én nieuw shop_mapping formaat
from __future__ import annotations

try:
    from shop_mapping import SHOP_NAME_MAP as RAW_MAP
except Exception as e:
    RAW_MAP = {}
    # Laat de caller dit zien in zijn UI (geen raise hier).

def _normalize(raw: dict) -> dict[int, dict]:
    """Zet RAW_MAP om naar {id: {name, region}} ongeacht input-formaat."""
    norm: dict[int, dict] = {}
    if not isinstance(raw, dict) or not raw:
        return norm

    # Eerste value inspecteren om formaat te herkennen
    first_val = next(iter(raw.values()))
    if isinstance(first_val, dict):
        # Nieuw formaat: {id: {"name": "...", "region": "...", ...}}
        for sid, meta in raw.items():
            try:
                sid_int = int(sid)
            except Exception:
                continue
            name   = (meta or {}).get("name") or str(meta)
            region = (meta or {}).get("region") or "All"
            norm[sid_int] = {"name": name, "region": region}
    else:
        # Oud formaat: {id: "Naam"}
        for sid, name in raw.items():
            try:
                sid_int = int(sid)
            except Exception:
                continue
            norm[sid_int] = {"name": str(name), "region": "All"}

    return norm

# ---- Genormaliseerde bron als enkele waarheid
SHOP_NAME_MAP: dict[int, dict] = _normalize(RAW_MAP)

# ---- Afgeleide mapping-helpers
ID_TO_NAME   : dict[int, str] = {sid: meta["name"]   for sid, meta in SHOP_NAME_MAP.items()}
NAME_TO_ID   : dict[str, int] = {meta["name"]: sid   for sid, meta in SHOP_NAME_MAP.items()}
ID_TO_REGION : dict[int, str] = {sid: meta["region"] for sid, meta in SHOP_NAME_MAP.items()}

# Regiolijst (inclusief 'All' bovenaan)
_regions = {meta["region"] for meta in SHOP_NAME_MAP.values() if meta.get("region")}
REGIONS   : list[str] = ["All"] + sorted(r for r in _regions if r != "All")

def get_ids_by_region(region: str | None) -> list[int]:
    """Alle shop_ids voor de gekozen regio; 'All'/None geeft alles terug."""
    if not SHOP_NAME_MAP:
        return []
    if not region or region == "All":
        return list(SHOP_NAME_MAP.keys())
    return [sid for sid, meta in SHOP_NAME_MAP.items() if meta.get("region") == region]

def get_region_by_id(shop_id: int) -> str | None:
    return SHOP_NAME_MAP.get(int(shop_id), {}).get("region")

def get_name_by_id(shop_id: int) -> str | None:
    return SHOP_NAME_MAP.get(int(shop_id), {}).get("name")
