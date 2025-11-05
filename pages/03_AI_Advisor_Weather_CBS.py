# pages/03_AI_Advisor_Weather_CBS.py
import os
import sys
import json
import re
import pandas as pd
import streamlit as st

# ───────── Mapping & API wrapper zoals in je benchmark-tool ─────────
try:
    from shop_mapping import SHOP_NAME_MAP  # {shop_id: "Store Name", ...}
except Exception:
    SHOP_NAME_MAP = None

from utils_pfmx import api_get_report, friendly_error

# ───────── Page setup ─────────
st.set_page_config(page_title="AI Advisor — Weer + CBS", page_icon="🧭", layout="wide")
st.title("🧭 AI Advisor — Weer + CBS (v1)")

# ───────── Project helpers ─────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from helpers_normalize import normalize_vemcount_response
from helpers_shop import ID_TO_NAME, get_ids_by_region, REGIONS

# ───────── Services ─────────
from services.weather_service import get_daily_forecast
from services.cbs_service import (
    get_consumer_confidence,
    get_cci_series,
    get_retail_index,
    list_retail_branches,
)

# (optioneel, nu niet gebruikt) – adviesregels
try:
    from services.advisor import build_advice
except Exception:
    def build_advice(company, baseline, forecast, cci):
        # simpele no-op fallback
        days = []
        for f in forecast:
            days.append({
                "date": f["date"],
                "weather": {"temp": f.get("temp", 0), "pop": f.get("pop", 0)},
                "stores": []
            })
        return {"days": days}

# ───────── Secrets ─────────
def _get_secret(key: str, env_fallback: str = "") -> str:
    val = st.secrets.get(key) or os.getenv(env_fallback or key.upper()) or ""
    return (val or "").strip()

OPENWEATHER_KEY = _get_secret("openweather_api_key", "OPENWEATHER_API_KEY")
CBS_DATASET     = _get_secret("cbs_dataset", "CBS_DATASET")
API_URL         = _get_secret("API_URL").rstrip("/")

missing = []
if not OPENWEATHER_KEY: missing.append("openweather_api_key")
if not CBS_DATASET:     missing.append("cbs_dataset")
if not API_URL:         missing.append("API_URL")
if missing:
    st.error("Missing secrets: " + ", ".join(missing) + "\n\nCheck Streamlit → Settings → Secrets.")
    st.stop()

# ───────── UI Controls ─────────
region = st.selectbox("Regio", options=["ALL"] + list(REGIONS), index=0)

# 7..30 dagen vooruit
days_ahead = st.slider("Dagen vooruit (weer-forecast)", min_value=7, max_value=30, value=14, step=1)

period_hist = st.selectbox(
    "Historische periode (voor baseline & regio-trend)",
    ["last_month", "this_year", "last_year"],
    index=0
)

st.subheader("Macro-context (CBS)")
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    months_back = st.slider("Maanden terug (CBS)", 6, 36, 18)
with c2:
    use_retail = st.checkbox("Toon detailhandel-index", value=False)
with c3:
    # Default “NONFOOD”, maar alleen relevant als use_retail True
    default_branch = "NONFOOD"
    dim_name, branch_items = list_retail_branches("85828NED")
    if branch_items:
        title_to_key = {b["title"]: str(b["key"]) for b in branch_items}
        titles = list(title_to_key.keys())
        # kies een ‘NONFOOD’-achtige default als beschikbaar
        df_idx = 0
        for i, t in enumerate(titles):
            if "nonfood" in t.lower():
                df_idx = i; break
        branch_title = st.selectbox("Branche (CBS, 85828NED)", titles, index=df_idx, disabled=not use_retail)
        branch_key = title_to_key[branch_title]
    else:
        branch_title = st.selectbox("Branche (CBS, 85828NED)", ["NONFOOD","FOOD","DH_TOTAAL"], index=0, disabled=not use_retail)
        branch_key = branch_title

# ───────── Macro reeksen (buiten de knop) ─────────
try:
    cci_series = get_cci_series(months_back=months_back, dataset=CBS_DATASET)
except Exception as e:
    cci_series = []
    st.info(f"CCI niet beschikbaar: {e}")

try:
    retail_series = []
    if use_retail:
        retail_series = get_retail_index(branch_code_or_title=branch_key, months_back=months_back) or \
                        get_retail_index(branch_code_or_title=branch_title, months_back=months_back) or \
                        get_retail_index(branch_code_or_title=default_branch, months_back=months_back)
except Exception as e:
    retail_series = []
    if use_retail:
        st.info(f"Detailhandelreeks (85828NED) niet beschikbaar: {e}")

# ───────── Helpers: datum, aggregaties, baseline, forecast ─────────
def add_effective_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Zet date_eff (date fallback timestamp) + weekday/ym/date_only."""
    d = df.copy()
    dt_date = pd.to_datetime(d.get("date"), errors="coerce")
    dt_ts   = pd.to_datetime(d.get("timestamp"), errors="coerce")
    d["date_eff"]  = dt_date.fillna(dt_ts)
    d["weekday"]   = d["date_eff"].dt.weekday
    d["ym"]        = d["date_eff"].dt.to_period("M").astype(str)
    d["date_only"] = d["date_eff"].dt.date
    return d

def fetch_hist_kpis_df(shop_ids, period: str) -> pd.DataFrame:
    """Meerdere data=, step=day — identiek aan benchmark-tool."""
    metrics = ["count_in", "conversion_rate", "turnover", "sales_per_visitor"]
    params = [("data", int(sid)) for sid in shop_ids]
    params += [("data_output", k) for k in metrics]
    params += [("source", "shops"), ("period", period), ("step", "day")]
    js = api_get_report(params)
    if friendly_error(js, period):
        return pd.DataFrame()
    name_map = SHOP_NAME_MAP or ID_TO_NAME
    return normalize_vemcount_response(js, name_map, kpi_keys=metrics)

def build_weekday_baselines(df: pd.DataFrame) -> dict:
    """Per weekday per store: gemiddelde visitors/conv/SPV + enkele kwantielen."""
    if df is None or df.empty:
        return {}
    d = add_effective_date_cols(df)
    out = {}
    for wd, g in d.groupby("weekday"):
        stores = {}
        for sid, gs in g.groupby("shop_id"):
            name = ID_TO_NAME.get(int(sid), f"Shop {sid}")
            # SPV
            if "sales_per_visitor" in gs.columns:
                spv_series = gs["sales_per_visitor"]
            else:
                denom = gs["count_in"].replace(0, pd.NA)
                spv_series = (gs["turnover"] / denom).fillna(0)
            # Conversie in Vemcount is vaak in % (bv. 10.75). Converteer naar ratio.
            conv_ratio = (gs["conversion_rate"] / 100.0)
            stores[name] = {
                "visitors":     float(gs["count_in"].mean()),
                "conversion":   float(conv_ratio.mean()),  # ratio 0..1
                "spv":          float(spv_series.mean()),
                "spv_median":   float(spv_series.median()),
                "visitors_p30": float(gs["count_in"].quantile(0.30)),
            }
        out[wd] = stores
    return out

def monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Maandtotalen: som visitors/turnover; conversie gewogen (ratio); SPV = omzet/bezoekers."""
    d = add_effective_date_cols(df)
    g = d.groupby("ym", as_index=False).agg(
        visitors=("count_in","sum"),
        turnover=("turnover","sum"),
    )
    # conversie-gewogen (aanname: conversion_rate is in procenten → deel door 100)
    d["weighted_conv"] = (d["conversion_rate"]/100.0) * d["count_in"]
    conv = d.groupby("ym", as_index=False).agg(conv_num=("weighted_conv","sum"),
                                              conv_den=("count_in","sum"))
    conv["conversion"] = (conv["conv_num"]/conv["conv_den"]).fillna(0)  # ratio 0..1
    out = g.merge(conv[["ym","conversion"]], on="ym", how="left")
    out["spv"] = (out["turnover"]/out["visitors"]).replace([float("inf")], 0).fillna(0)
    return out

def mom_yoy(dfm: pd.DataFrame):
    if dfm is None or dfm.empty:
        return {}
    m = dfm.copy().sort_values("ym").reset_index(drop=True)
    m["ym_dt"] = pd.to_datetime(m["ym"].astype(str) + "-01", errors="coerce")
    m = m.dropna(subset=["ym_dt"])
    if m.empty:
        return {}
    last = m.iloc[-1]
    prev = m.iloc[-2] if len(m) > 1 else None
    yoy_dt  = last["ym_dt"] - pd.DateOffset(years=1)
    yoy_row = m.loc[m["ym_dt"] == yoy_dt]
    yoy_row = yoy_row.iloc[0] if not yoy_row.empty else None

    def pct(a,b):
        if b in [0,None] or pd.isna(b): return None
        try: return (float(a)/float(b) - 1) * 100
        except: return None

    return {
        "last_ym":   last["ym_dt"].strftime("%Y-%m"),
        "visitors":  float(last.get("visitors", 0)),
        "turnover":  float(last.get("turnover", 0)),
        "conversion":float(last.get("conversion", 0)),   # ratio
        "spv":       float(last.get("spv", 0)),
        "mom": {
            "turnover":  pct(last.get("turnover", 0),  prev.get("turnover", 0))  if prev is not None else None,
            "visitors":  pct(last.get("visitors", 0),  prev.get("visitors", 0))  if prev is not None else None,
            "conversion":pct(last.get("conversion", 0),prev.get("conversion", 0))if prev is not None else None,
            "spv":       pct(last.get("spv", 0),       prev.get("spv", 0))       if prev is not None else None,
        },
        "yoy": {
            "turnover":  pct(last.get("turnover", 0),  yoy_row.get("turnover", 0))  if yoy_row is not None else None,
            "visitors":  pct(last.get("visitors", 0),  yoy_row.get("visitors", 0))  if yoy_row is not None else None,
            "conversion":pct(last.get("conversion", 0),yoy_row.get("conversion", 0))if yoy_row is not None else None,
            "spv":       pct(last.get("spv", 0),       yoy_row.get("spv", 0))       if yoy_row is not None else None,
        },
    }

# ───────── Postcode → weercoördinaten (PC2 + regio-fallbacks) ─────────
REGION_COORDS = {
    "Noord NL": (53.219, 6.566),     # Groningen
    "Midden NL": (52.090, 5.121),    # Utrecht
    "Zuid NL": (51.441, 5.469),      # Eindhoven
    "West NL": (52.373, 4.900),      # Amsterdam
    "Oost NL": (52.223, 6.000),      # Deventer-ish
    "ALL": (52.373, 4.900),          # landelijk
}
PC2_TO_COORD = {
    # West/Randstad
    "10": (52.37,4.90),"11": (52.37,4.90),"20": (52.0,4.36),"21": (52.0,4.36),
    "22": (52.0,4.36),"23": (52.39,4.64),"24": (52.39,4.64),"25": (52.39,4.64),
    "26": (52.39,4.64),"27": (52.08,4.30),"28": (52.08,4.30),"29": (51.85,4.28),
    # Midden
    "30": (51.92,4.48),"31": (51.92,4.48),"32": (51.92,4.48),"33": (51.83,4.67),
    "34": (52.09,5.12),"35": (52.13,5.19),"36": (52.35,5.22),"37": (52.13,5.19),
    # Oost
    "38": (52.51,6.09),"39": (52.51,6.09),"80": (52.52,6.09),"81": (52.38,6.64),
    "82": (52.50,5.47),"83": (52.70,5.75),
    # Noord
    "90": (53.20,6.57),"91": (53.20,6.57),"92": (53.19,5.79),"93": (53.05,5.67),
    "94": (52.99,6.56),"95": (53.32,6.92),"96": (53.10,6.00),"97": (53.22,6.56),
    "98": (53.22,6.56),"99": (53.22,6.56),
    # Zuid
    "50": (51.44,5.47),"51": (51.44,5.47),"52": (51.59,5.08),"53": (51.59,5.08),
    "54": (51.59,5.08),"55": (51.59,5.08),"56": (51.44,5.47),"57": (51.65,5.05),
    "58": (51.56,5.09),
    "60": (51.44,6.17),"61": (51.23,5.99),"62": (50.85,5.69),"63": (50.85,5.69),
    "64": (50.85,5.69),
}
PC2_RE = re.compile(r"^\s*(\d{2})")

def _pc2_from_text(txt: str) -> str | None:
    m = PC2_RE.match(str(txt)) if txt else None
    return m.group(1) if m else None

def _extract_postcodes_from_df(df: pd.DataFrame) -> list[str]:
    out = []
    if "shop_name" not in df.columns:
        return out
    for v in df["shop_name"].dropna().astype(str).head(500):
        pc = None
        if v.strip().startswith("{") and '"postcode"' in v:
            try:
                j = json.loads(v)
                pc = j.get("postcode")
            except Exception:
                pc = None
        if not pc:
            m = re.search(r"\b(\d{4})\s*[A-Z]{0,2}\b", v)
            pc = m.group(1) if m else None
        if pc:
            pc2 = _pc2_from_text(pc)
            if pc2:
                out.append(pc2)
    return out

def pick_coords_for_weather(region_label: str, df_hist: pd.DataFrame) -> tuple[float, float]:
    pc2_list = _extract_postcodes_from_df(df_hist)
    if pc2_list:
        seen, latlons = set(), []
        for pc2 in pc2_list:
            if pc2 in PC2_TO_COORD and pc2 not in seen:
                latlons.append(PC2_TO_COORD[pc2]); seen.add(pc2)
        if latlons:
            lat = sum(x[0] for x in latlons) / len(latlons)
            lon = sum(x[1] for x in latlons) / len(latlons)
            return (lat, lon)
    return REGION_COORDS.get(region_label, REGION_COORDS["ALL"])

# ───────── Tiny formatters ─────────
def _eur(x):
    try: return "€{:,.0f}".format(x).replace(",", ".")
    except: return "–"

def _fmt_pct(x):
    return "—" if (x is None or pd.isna(x)) else f"{x:+.1f}%"

# ───────── Shop selectie ─────────
st.caption("Selecteer regio en druk op de knop om aanbevelingen te genereren.")
if region == "ALL":
    shop_ids = sorted([int(k) for k in (SHOP_NAME_MAP or ID_TO_NAME).keys()])
else:
    shop_ids = get_ids_by_region(region) or sorted([int(k) for k in (SHOP_NAME_MAP or ID_TO_NAME).keys()])

st.write(f"**{len(shop_ids)}** winkels geselecteerd in regio: **{region}**")
st.text(f"ShopIDs → {shop_ids[:25]}{' …' if len(shop_ids)>25 else ''}")

# ───────── Action ─────────
if st.button("Genereer aanbevelingen"):
    # 1) Historische data
    df = fetch_hist_kpis_df(shop_ids, period_hist)

    # Debug-tabel: parse shop_name → nette kolommen; timestamp → date; verwijder 'date'
    if not df.empty:
        ddbg = df.copy()
        # parse shop_name JSON → name/postcode/region
        def parse_shopname(x):
            if isinstance(x, str) and x.strip().startswith("{"):
                try:
                    j = json.loads(x)
                    return j.get("name"), j.get("postcode"), j.get("region")
                except Exception:
                    return None, None, None
            return x, None, None
        triples = ddbg["shop_name"].apply(parse_shopname)
        ddbg[["name","postcode","region"]] = pd.DataFrame(triples.tolist(), index=ddbg.index)
        ddbg["date"] = pd.to_datetime(ddbg.get("timestamp"), errors="coerce").dt.date
        show_cols = ["date","shop_id","name","postcode","region","count_in","conversion_rate","turnover","sales_per_visitor"]
        show_cols = [c for c in show_cols if c in ddbg.columns]
        with st.expander("🛠️ Debug — eerste rijen (hist KPI’s)"):
            st.dataframe(ddbg[show_cols].head(15), use_container_width=True)
    else:
        st.warning("Geen historische KPI-data voor deze selectie/periode. Probeer ‘this_year’ of ‘last_year’.")
        st.stop()

    # 2) Baselines + regiotrend
    df = add_effective_date_cols(df)
    baseline = build_weekday_baselines(df)
    dfm = monthly_agg(df)
    trend = mom_yoy(dfm)

    # 3) Weer (coördinaten automatisch uit postcodes/regio) + CBS
    lat, lon = pick_coords_for_weather(region, df)
    forecast = get_daily_forecast(lat, lon, OPENWEATHER_KEY, days_ahead)

    try:
        cci_info = get_consumer_confidence(CBS_DATASET)
        cci_val = cci_info.get("value", 0.0)
        cci_period = cci_info.get("period", "")
    except Exception as e:
        st.warning(f"Kon CBS Consumentenvertrouwen niet ophalen (gebruik standaard 0). Details: {e}")
        cci_val, cci_period = 0.0, "n/a"

    # 4) AI-adviesregels (dag/winkel)
    advice = build_advice("Your Company", baseline, forecast, cci_val)

    # 5) Regionale baseline (gemiddelden van alle winkels per weekday)
    baseline_day = {}
    for wd, storemap in baseline.items():
        if not storemap:
            continue
        visitors = pd.Series([v["visitors"] for v in storemap.values()]).mean()
        spv      = pd.Series([v["spv"]      for v in storemap.values()]).mean()
        baseline_day[wd] = {"visitors": float(visitors), "spv": float(spv)}
    if not baseline_day:
        st.warning("Geen baseline beschikbaar (te weinig dagen). Kies een ruimere periode.")
        st.stop()

    # ── Silver-platter regio
    st.subheader("🔎 Silver-platter samenvatting (regio)")
    if trend:
        colA, colB, colC, colD = st.columns(4)
        colA.metric(f"Omzet {trend.get('last_ym','-')}", _eur(trend.get('turnover', 0)), _fmt_pct(trend['mom'].get('turnover')))
        colB.metric("Bezoekers", f"{trend.get('visitors', 0):,.0f}".replace(",", "."), _fmt_pct(trend['mom'].get('visitors')))
        # conversie als % (ratio→%)
        conv_pct = trend.get('conversion', 0)*100.0
        colC.metric("Conversie", f"{conv_pct:.2f}%", _fmt_pct(trend['mom'].get('conversion')))
        colD.metric("SPV", f"€{trend.get('spv', 0):.2f}", _fmt_pct(trend['mom'].get('spv')))
        yoy_line = (
            "YoY: "
            + (f"Omzet {trend['yoy']['turnover']:.1f}%  " if trend['yoy']['turnover'] is not None else "")
            + (f"• Bezoekers {trend['yoy']['visitors']:.1f}%  " if trend['yoy']['visitors'] is not None else "")
            + (f"• Conversie {trend['yoy']['conversion']:.1f}%  " if trend['yoy']['conversion'] is not None else "")
            + (f"• SPV {trend['yoy']['spv']:.1f}%" if trend['yoy']['spv'] is not None else "")
        ).strip()
        if yoy_line != "YoY:":
            st.caption(yoy_line)

    # ── Macro duiding (CCI)
    # Toon korte uitleg en waarschuwing bij onrealistische schalen (bv. 474)
    cci_help = (
        "CCI = Consumentenvertrouwen-index (CBS 83693NED). Rond 0 is neutraal; "
        "boven 0 positiever sentiment (meer kooplust), onder 0 negatiever sentiment. "
        "Gebruik hogere SPV-doelen bij positief sentiment; focus op bundel/waarde bij negatief."
    )
    st.metric("Consumentenvertrouwen (CBS)", f"{cci_val}", help=cci_help)
    if abs(float(cci_val or 0)) > 100:
        st.warning("De CCI-waarde oogt buiten de gebruikelijke bandbreedte (verwacht ~-60..+40). Controleer het gebruikte veld in de dataset.")

    # ── Komende dagen — verwachting & acties
    st.subheader("📅 Komende dagen — verwachting & acties")
    for d in advice["days"]:
        w = d.get("weather", {})
        temp = w.get("temp", 0.0); pop = int((w.get("pop", 0.0) or 0)*100)
        with st.expander(f'📆 {d["date"]} — temp {temp:.1f}°C • POP {pop}%'):
            if d.get("stores"):
                for s in d["stores"]:
                    st.markdown(f"**{s['store']}**")
                    st.write("— Storemanager:", " • ".join(s["store_actions"]))
                    st.write("— Regiomanager:", " • ".join(s["regional_actions"]))
            else:
                st.write("Voor deze prototype-pagina zijn winkel-specifieke acties (nog) niet ingevuld.")

    # ── Regiotrend (maandelijks)
    st.subheader("📊 Regiotrend (maandelijks)")
    if not dfm.empty:
        left, right = st.columns(2)
        left.line_chart(dfm.set_index("ym")[["turnover"]].rename(columns={"turnover": "Omzet"}))
        right.line_chart(dfm.set_index("ym")[["visitors"]].rename(columns={"visitors": "Bezoekers"}))

    # ── Macro tiles (context)
    if cci_series:
        try:
            last_cci = float(cci_series[-1].get("cci", 0))
            st.metric("CCI (laatste maand)", f"{last_cci:.1f}")
        except Exception:
            st.metric("CCI (laatste maand)", f"{cci_series[-1].get('cci')}")
        with st.expander("📈 CCI reeks (CBS)"):
            st.line_chart({"CCI": [x.get("cci", 0) for x in cci_series]})

    if retail_series:
        last_r = retail_series[-1]
        st.metric(f"Detailhandel ({last_r['branch']}) — {last_r['series']}", f"{last_r['retail_value']:.1f}")
        with st.expander("🛍️ Detailhandel reeks (CBS 85828NED)"):
            st.line_chart({"Retail": [x["retail_value"] for x in retail_series]})
    elif use_retail:
        st.info("Geen detailhandelreeks gevonden voor deze branche/periode (85828NED). Probeer een andere branche of zet de tegel uit.")
