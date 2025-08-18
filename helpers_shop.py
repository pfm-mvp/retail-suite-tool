# helpers_shop.py
from __future__ import annotations

# We proberen het export-object te pakken, ongeacht vorm
try:
    from shop_mapping import SHOP_NAME_MAP as RAW_MAP
except Exception:
    RAW_MAP = {}

ID_TO_NAME: dict[int, str] = {}
ID_TO_REGION: dict[int, str] = {}

def _build_from_raw():
    """
    Ondersteunt deze vormen:
    1) {123: "Amsterdam", 124: "Rotterdam"}
    2) {123: {"name":"Amsterdam", "region":"Noord"}, ...}
    3) {"shops": [{"id":123,"name":"Amsterdam","region":"Noord"}, ...]}
    4) [ {"id":123,"name":"Amsterdam","region":"Noord"}, ... ]
    Alles wordt genormaliseerd naar:
      - ID_TO_NAME[id]   -> naam
      - ID_TO_REGION[id] -> regio (default 'All')
    """
    def _push(sid, name, region="All"):
        try:
            sid = int(sid)
            name = (name or "").strip()
            region = (region or "All").strip() or "All"
            if sid and name:
                ID_TO_NAME[sid] = name
                ID_TO_REGION[sid] = region
        except Exception:
            pass

    m = RAW_MAP

    if isinstance(m, dict) and "shops" in m and isinstance(m["shops"], (list, tuple)):
        for row in m["shops"]:
            if isinstance(row, dict):
                _push(row.get("id"),
                      row.get("name") or row.get("shop_name") or row.get("title"),
                      row.get("region"))
        return

    if isinstance(m, dict):
        for sid, val in m.items():
            if isinstance(val, str):
                _push(sid, val)
            elif isinstance(val, dict):
                _push(sid,
                      val.get("name") or val.get("shop_name") or val.get("title"),
                      val.get("region"))
        return

    if isinstance(m, (list, tuple)):
        for row in m:
            if isinstance(row, dict):
                _push(row.get("id"),
                      row.get("name") or row.get("shop_name") or row.get("title"),
                      row.get("region"))

# bouw de mapping
ID_TO_NAME.clear(); ID_TO_REGION.clear()
_build_from_raw()

# handige afgeleiden
NAME_TO_ID: dict[str, int] = {v: k for k, v in ID_TO_NAME.items()}
REGIONS: list[str] = ["All"] + sorted({r for r in ID_TO_REGION.values() if r and r != "All"})

def get_ids_by_region(region: str) -> list[int]:
    if not region or region == "All":
        return list(ID_TO_NAME.keys())
    return [sid for sid, reg in ID_TO_REGION.items() if reg == region]

def get_region_by_id(shop_id: int) -> str | None:
    return ID_TO_REGION.get(int(shop_id))

def get_name_by_id(shop_id: int) -> str | None:
    return ID_TO_NAME.get(int(shop_id))
