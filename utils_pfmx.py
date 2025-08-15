# utils_pfmx.py (no-encode brackets; literal ":"; robust errors; includes inject_css + friendly_error)
import os
from typing import List, Tuple, Union, Iterable, Optional
import requests

# ---- Config ----
API_URL = os.getenv("API_URL", "").rstrip("/")  # e.g. https://vemcount-agent.onrender.com/get-report
ARRAY_KEYS = {"data", "data_output"}

# ---- UI helper (optional) ----
def inject_css():
    try:
        import streamlit as st
        st.markdown("""
            <style>
            .stMetric { text-align: center; }
            .block-container { padding-top: 1rem; padding-bottom: 2rem; }
            </style>
        """, unsafe_allow_html=True)
    except Exception:
        pass

def _expand_params(pairs: List[Tuple[str, Union[str,int,float,list,tuple]]],
                   prefer_brackets: bool = True) -> List[Tuple[str,str]]:
    expanded: List[Tuple[str,str]] = []
    for k, v in pairs:
        base = str(k).replace("[]","")
        if isinstance(v, (list, tuple)):
            use_key = f"{base}[]" if (prefer_brackets and base in ARRAY_KEYS) else base
            for vi in v:
                expanded.append((use_key, str(vi)))
        else:
            # single value: never force [] (werkt zo in jouw eerdere tools)
            expanded.append((base, str(v)))
    return expanded

def _build_qs_literal(expanded_pairs: List[Tuple[str,str]]) -> str:
    # IMPORTANT: no URL encoding at all; keep [] and ":" literally,
    # like your working calculators.
    return "&".join(f"{k}={v}" for k, v in expanded_pairs)

def api_get_report(params: List[Tuple[str, Union[str,int,float,list,tuple]]],
                   timeout: int = 60,
                   prefer_brackets: bool = True,
                   headers: Optional[dict] = None) -> dict:
    """
    POST to /get-report with a querystring where [] remain literal (no %5B%5D) and ':' remains ':'.
    - Lists under keys in ARRAY_KEYS become repeated pairs with '[]' suffix when prefer_brackets=True.
    - Single values stay without brackets.
    - Returns JSON or an error dict { _error, status, _url, _method, exception }.
    """
    if not API_URL:
        return {"_error": True, "status": None, "_url": None, "_method": "POST",
                "exception": "API_URL missing (set env/Streamlit secrets)"}
    expanded = _expand_params(params, prefer_brackets=prefer_brackets)
    qs = _build_qs_literal(expanded)
    url = f"{API_URL}?{qs}"
    try:
        r = requests.post(url, timeout=timeout, headers=headers or {})
        try:
            r.raise_for_status()
        except requests.HTTPError as he:
            return {"_error": True, "status": r.status_code, "_url": url, "_method": "POST",
                    "exception": str(he), "_body": r.text[:500] if r.text else ""}
        try:
            return r.json()
        except Exception as je:
            return {"_error": True, "status": r.status_code, "_url": url, "_method": "POST",
                    "exception": f"JSON parse failed: {je}", "_body": r.text[:500] if r.text else ""}
    except Exception as e:
        return {"_error": True, "status": None, "_url": url, "_method": "POST",
                "exception": str(e)}

def friendly_error(resp_json: dict, period: Optional[str] = None) -> bool:
    """
    If resp_json contains an error dict from api_get_report, show it (Streamlit) and return True.
    Otherwise return False.
    """
    if not isinstance(resp_json, dict) or not resp_json.get("_error"):
        return False
    try:
        import streamlit as st
        hdr = f"API call failed — status: {resp_json.get('status')}"
        url = resp_json.get("_url")
        body = resp_json.get("_body", "")
        st.error(f"{hdr} | _url: {url} | _method: POST | exception: {resp_json.get('exception')}")
        if body:
            with st.expander("Response body (truncated)"):
                st.code(body)
        if period:
            st.caption(f"Period requested: {period}")
    except Exception:
        pass
    return True
