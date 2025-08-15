
import streamlit as st
import requests
import pandas as pd
import numpy as np
from urllib.parse import urlencode
from typing import Optional, List, Tuple, Dict, Any

def inject_css():
    st.markdown('<style>.kpi{padding:12px 14px;border:1px solid #EEE;border-radius:16px;}</style>', unsafe_allow_html=True)

def fmt_eur(x):
    try: return '€ {:,.0f}'.format(float(x)).replace(',', '.')
    except: return '€ 0'

def fmt_pct(x):
    try: return '{:.1f}%'.format(float(x)*100.0)
    except: return '0.0%'

def _normalize_base(url: str) -> str:
    if not url:
        return ""
    if "://" not in url:
        url = "https://" + url
    return url.rstrip("/")

def _api_base() -> str:
    return _normalize_base(st.secrets.get("API_URL","").strip())

def _with_get_report_prefix(base: str) -> str:
    # Zorg dat elk pad onder /get-report hangt
    b = base.rstrip("/")
    if b.endswith("/get-report"):
        return b
    return b + "/get-report"

def build_params_reports_plain(
    source: str,
    period: str,
    data_ids: List[int],
    outputs: List[str],
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    period_step: Optional[str] = None,
    extra: Optional[List[Tuple[str,str]]] = None,
) -> List[Tuple[str, Any]]:
    params: List[Tuple[str, Any]] = [("source", source), ("period", period)]
    if date_from: params.append(("date_from", date_from))
    if date_to: params.append(("date_to", date_to))
    if period_step: params.append(("period_step", period_step))

    # herhaal data=
    params += [("data", int(i)) for i in data_ids]

    # 2‑sporenbeleid voor outputs:
    #   1) herhaalde keys (backends die dit goed ondersteunen)
    #   2) een laatste "samengevoegde" key (backends die alleen de laatste lezen)
    if outputs:
        params += [("data_output", o) for o in outputs]                 # herhaald
        params.append(("data_output", ",".join(outputs)))                # joined als laatste

    if extra:
        params += list(extra)
    return params

def _post_json(full_url: str, timeout: int=90) -> Dict[str, Any]:
    try:
        r = requests.post(full_url, timeout=timeout); r.raise_for_status(); return r.json()
    except Exception as e:
        return {'_error': True, 'status': getattr(e,'response',None).status_code if hasattr(e,'response') and e.response is not None else None, '_url': full_url, '_method': 'POST', 'exception': str(e)}


def _encode_params_style_a(params: List[Tuple[str, Any]]) -> List[Tuple[str, str]]:
    """Repeated keys without [] for lists."""
    out: List[Tuple[str, str]] = []
    for k, v in params:
        if isinstance(v, (list, tuple)):
            for vi in v:
                out.append((k, str(vi)))
        else:
            out.append((k, str(v)))
    return out

def _encode_params_style_b(params: List[Tuple[str, Any]]) -> List[Tuple[str, str]]:
    """Array keys with [] for lists (e.g., data[])."""
    out: List[Tuple[str, str]] = []
    array_keys = {"data", "data_output", "zones", "sensors", "groups"}  # extend as needed
    for k, v in params:
        if isinstance(v, (list, tuple)):
            bk = f"{k}[]"
            for vi in v:
                out.append((bk, str(vi)))
        else:
            if k in array_keys:
                out.append((f"{k}[]", str(v)))
            else:
                out.append((k, str(v)))
    return out

def _post_query(url: str, kv_params: List[Tuple[str, str]], timeout: int = 90) -> Dict[str, Any]:
    # preserve current behaviour: POST with querystring
    full = url + "?" + urlencode(kv_params, doseq=True)
    return _post_json(full, timeout=timeout)

def api_get_report(params: List[Tuple[str, Any]], timeout: int = 90, prefer_brackets: bool = True) -> Dict[str, Any]:
    """
    Robust /get-report: tries both parameter styles.
    - prefer_brackets=True: first attempt with [] array-keys (for multi data/data_output),
      then fallback to repeated keys without [].
    - Lists are supported for e.g. ("data", [1,2]) and ("data_output", ["count_in","turnover"]).
    """
    base = _with_get_report_prefix(_api_base())
    encoders = [_encode_params_style_b, _encode_params_style_a] if prefer_brackets else [_encode_params_style_a, _encode_params_style_b]

    last_err = None
    for enc in encoders:
        try:
            kv = enc(params)
            return _post_query(base, kv, timeout=timeout)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"/get-report failed for both encoders. Last error: {last_err}")

# api_get_live_inside removed: use api_get_report with data_output=inside if needed.

def _as_float(x):
    try: return float(x)
    except: return 0.0

def normalize_vemcount_daylevel(js: Dict[str, Any]) -> pd.DataFrame:
    if not isinstance(js, dict) or 'data' not in js: return pd.DataFrame()
    rows = []
    for date_key, by_shop in (js.get('data') or {}).items():
        try: date = pd.to_datetime(date_key.replace('date_',''), errors='coerce').date()
        except: date = None
        if not isinstance(by_shop, dict): continue
        for sid, payload in by_shop.items():
            try: shop_id = int(sid)
            except:
                try: shop_id = int(payload.get('shop_id'))
                except: continue
            if not isinstance(payload, dict): continue
            if 'data' in payload and isinstance(payload['data'], dict):
                kpis = payload['data']
                rows.append({'date': date, 'shop_id': shop_id,
                             'count_in': _as_float(kpis.get('count_in')),
                             'conversion_rate': _as_float(kpis.get('conversion_rate')),
                             'turnover': _as_float(kpis.get('turnover')),
                             'sales_per_visitor': _as_float(kpis.get('sales_per_visitor') or 0)})
            elif 'dates' in payload and isinstance(payload['dates'], dict):
                cin=tov=conv_num=conv_den=spv_num=spv_cnt=0.0
                for ts, slot in payload['dates'].items():
                    if not isinstance(slot, dict): continue
                    k = slot.get('data', {})
                    cin += _as_float(k.get('count_in')); tov += _as_float(k.get('turnover'))
                    cr=k.get('conversion_rate'); spv=k.get('sales_per_visitor')
                    if cr is not None:
                        try: conv_num += float(cr); conv_den += 1
                        except: pass
                    if spv is not None:
                        try: spv_num += float(spv); spv_cnt += 1
                        except: pass
                rows.append({'date': date, 'shop_id': shop_id, 'count_in': cin, 'turnover': tov,
                             'conversion_rate': (conv_num/conv_den) if conv_den>0 else 0.0,
                             'sales_per_visitor': (spv_num/spv_cnt) if spv_cnt>0 else 0.0})
    df = pd.DataFrame(rows)
    if df.empty: return df
    if 'sales_per_visitor' in df.columns:
        mask = df['sales_per_visitor'].isna()
        if 'turnover' in df.columns and 'count_in' in df.columns:
            df.loc[mask, 'sales_per_visitor'] = df.loc[mask, 'turnover'] / df.loc[mask, 'count_in'].replace(0, np.nan)
    return df

def friendly_error(js, context: str=''):
    if isinstance(js, dict) and js.get('_error'):
        st.error(f'Geen data ontvangen ({context}). Controleer API_URL of periode/IDs. [status={js.get("status")}]')
        st.caption(f"↪ {js.get('_method','')} {js.get('_url','')}")
        return True
    return False
