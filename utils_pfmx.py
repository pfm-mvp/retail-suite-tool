# utils_pfmx.py
import os
from typing import List, Tuple, Union, Optional
from urllib.parse import quote
import requests

# ---- Config ----
API_URL = os.getenv("API_URL", "").rstrip("/")  # e.g. https://vemcount-agent.onrender.com/get-report
ARRAY_KEYS = {"data", "data_output"}  # extend if needed: zones, sensors, groups

# ---- UI helper (compat with existing Home.py) ----
def inject_css():
    """Lightweight CSS injector used by existing pages."""
    try:
        import streamlit as st
        st.markdown(
            """
            <style>
                .block-container{padding-top:1rem;padding-bottom:2rem;}
                .stMetric > div {background:#F8F9FA;border-radius:16px;padding:12px;}
                .stTabs [data-baseweb="tab-list"] {gap:0.5rem;}
                .stButton>button {border-radius:999px;padding:.5rem 1rem;}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception:
        # no-op if Streamlit not available
        pass

# ---- Querystring builder ----
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

# ---- API helper ----
def api_get_report(params: List[Tuple[str, Union[str,int,float,list,tuple]]],
                   timeout: int = 60,
                   prefer_brackets: bool = True) -> dict:
    """
    Universal POST helper for /get-report (Vemcount via proxy).
    Default: bracket-style (data[]=..., data_output[]=...), exactly like your Postman calls.
    Returns JSON on success; on HTTP error it returns a structured dict with '_error': True.
    """
    if not API_URL:
        return {"_error": True, "status": 500, "exception": "API_URL missing in environment/secrets."}
    qs = _make_qs(params, prefer_brackets=prefer_brackets)
    url = f"{API_URL}?{qs}"
    try:
        r = requests.post(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as http_err:
        return {
            "_error": True,
            "status": r.status_code if 'r' in locals() else None,
            "_url": url,
            "_method": "POST",
            "exception": str(http_err),
            "_text": r.text[:500] if 'r' in locals() else None
        }
    except Exception as e:
        return {"_error": True, "status": None, "_url": url, "_method": "POST", "exception": str(e)}

# ---- Friendly error reporter (backwards compat) ----
def friendly_error(resp_json: dict, period: Optional[str] = None) -> bool:
    """
    If resp_json indicates an error, show a Streamlit error and return True (caller may st.stop()).
    Safe to import when Streamlit isn't present (returns True/False only).
    """
    is_err = isinstance(resp_json, dict) and resp_json.get("_error") is True
    if not is_err:
        return False
    msg = f"API call failed"
    if period:
        msg += f" for period '{period}'"
    details = []
    for k in ("status","_url","_method","exception"):
        v = resp_json.get(k)
        if v is not None:
            details.append(f"{k}: {v}")
    detail_txt = " | ".join(details)
    try:
        import streamlit as st
        st.error(msg + (f" — {detail_txt}" if detail_txt else ""))
    except Exception:
        pass
    return True
