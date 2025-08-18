# helpers_shop.py
from __future__ import annotations

try:
    from shop_mapping import SHOP_NAME_MAP as RAW_MAP   # can be old or new format
except Exception:
    RAW_MAP = {}

def _coerce_id(x):
    try:
        return int(x)
    except Exception:
        return x

def build_maps(raw: dict):
    """Return (ID_TO_NAME, NAME_TO_ID, REGIONS) for both old and new mapping formats."""
    id_to_name = {}
    name_to_id = {}
    regions = set()

    if not isinstance(raw, dict) or not raw:
        return id_to_name, name_to_id, ["All"]

    sample = next(iter(raw.values()))

    # New format: {id: {"name": "...", "region": "..."}}
    if isinstance(sample, dict):
        for sid, meta in raw.items():
            sid_i = _coerce_id(sid)
            name = str(meta.get("name") or meta.get("shop_name") or meta)
            region = meta.get("region") or "All"
            id_to_name[sid_i] = name
            name_to_id[name] = sid_i
            regions.add(region)
    else:
        # Old format: {id: "name"}
        for sid, name in raw.items():
            sid_i = _coerce_id(sid)
            nm = str(name)
            id_to_name[sid_i] = nm
            name_to_id[nm] = sid_i
        regions = {"All"}

    return id_to_name, name_to_id, sorted(regions)

ID_TO_NAME, NAME_TO_ID, REGIONS = build_maps(RAW_MAP)
