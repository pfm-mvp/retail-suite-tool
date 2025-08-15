
# utils_pfmx.py
import os
import requests
from typing import List, Tuple, Union
from urllib.parse import quote

API_URL = os.getenv("API_URL", "").rstrip("/")  # e.g. https://vemcount-agent.onrender.com/get-report
ARRAY_KEYS = {"data", "data_output"}  # extend if needed: zones, sensors, groups

def _normalize_key_for_brackets(key: str, prefer_brackets: bool, is_list: bool) -> str:
    base = key.replace("[]", "")
    if prefer_brackets and is_list and base in ARRAY_KEYS:
        return f"{base}[]"
    return base

def _make_qs(pairs: List[Tuple[str, Union[str, int, float, list, tuple]]],
             prefer_brackets: bool) -> str:
    """
    Build a querystring keeping [] literal (not %5B%5D) and ':' literal in times.
    """
    parts = []
    for k, v in pairs:
        is_list = isinstance(v, (list, tuple))
        k_norm = _normalize_key_for_brackets(str(k), prefer_brackets, is_list)
        k_enc  = quote(k_norm, safe="[]")  # leave [] as-is
        if is_list:
            for vi in v:
                parts.append(f"{k_enc}={quote(str(vi), safe=':')}")
        else:
            parts.append(f"{k_enc}={quote(str(v),  safe=':')}")
    return "&".join(parts)

def api_get_report(params: List[Tuple[str, Union[str,int,float,list,tuple]]],
                   timeout: int = 60,
                   prefer_brackets: bool = True) -> dict:
    """
    Universal POST helper for /get-report (Vemcount via proxy).
    Default: bracket-style (data[]=..., data_output[]=...), exactly like your Postman calls.
    """
    if not API_URL:
        raise RuntimeError("API_URL missing (set in env or Streamlit secrets).")
    qs = _make_qs(params, prefer_brackets=prefer_brackets)
    url = f"{API_URL}?{qs}"
    r = requests.post(url, timeout=timeout)
    r.raise_for_status()
    try:
        return r.json()
    except Exception as je:
        raise RuntimeError(f"JSON parse failed: {je}\nBody: {r.text[:500]}")
