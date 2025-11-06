# pages/03_AI_Advisor_Weather_CBS.py
import os, sys, json
from datetime import datetime, date
from collections import defaultdict
import pandas as pd
import numpy as np
import streamlit as st

# ───────── Mapping & API wrapper ─────────
try:
    from shop_mapping import SHOP_NAME_MAP  # {shop_id: "Store Name" or JSON-string}
except Exception:
    SHOP_NAME_MAP = None

from utils_pfmx import api_get_report, friendly_error

# ───────── Page setup ─────────
st.set_page_config(page_title="AI Advisor — Weer + CBS", page_icon="🧭", layout="wide")
st.title("🧭 AI Advisor — Weer + CBS")

# ───────── Project helpers ─────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from helpers_normalize import normalize_vemcount_response
from helpers_shop import ID_TO_NAME, get_ids_by_region, REGIONS

# ───────── Services ─────────
from services.weather_service import get_daily_forecast
from services.cbs_service import (
    get_consumer_confidence,  # laatste maand
    get_cci_series,           # reeks (83693NED)
    get_retail_index,         # optioneel (85828NED)
    list_retail_branches,     # optioneel
)

# ───────── Secrets ─────────
def _get_secret(key: str, env_fallback: str = "") -> str:
    v = st.secrets.get(key) or os.getenv(env_fallback or key.upper()) or ""
    return (v or "").strip()

OPENWEATHER_KEY = _get_secret("openweather_api_key", "OPENWEATHER_API_KEY")
CBS_DATASET     = _get_secret("cbs_dataset", "CBS_DATASET") or "83693NED"  # default veilig
API_URL         = _get_secret("API_URL").rstrip("/")

missing = []
if not OPENWEATHER_KEY: missing.append("openweather_api_key")
if not CBS_DATASET:     missing.append("cbs_dataset")
if not API_URL:         missing.append("API_URL")
if missing:
    st.error("Missing secrets: " + ", ".join(missing) + "\n\nCheck Streamlit → Settings → Secrets.")
    st.stop()

# ───────── Regio → (lat,lon) voor weer (centrumpunten) ─────────
REGION_COORDS = {
    "Noord NL": (53.219, 6.566),     # Groningen
    "West NL":  (52.372, 4.900),     # Amsterdam
    "Midden NL":(52.090, 5.121),     # Utrecht
    "Oost NL":  (52.222, 6.893),     # Enschede
    "Zuid NL":  (51.441, 5.469),     # Eindhoven
}
DEFAULT_COORDS = (52.372, 4.900)

# ───────── UI Controls ─────────
region = st.selectbox("Regio", options=["ALL"] + list(REGIONS), index=0)
days_ahead = st.slider("Dagen vooruit (weer)", 7, 30, 14)  # 7..30
period_hist = st.selectbox("Historische periode", ["last_month", "this_year", "last_year"], index=0)

st.subheader("Macro-context (CBS)")
c1, c2, c3 = st.columns([1,1,2])
with c1:
    months_back = st.slider("Maanden terug (CCI)", 6, 36, 18)
with c2:
    use_retail = st.checkbox("Toon detailhandel-index (85828NED)", value=False)  # optioneel, default uit
with c3:
    # standaard NONFOOD zoals gevraagd
    dim_name, branch_items = list_retail_branches("85828NED")
    if branch_items:
        # kies een titel die NONFOOD bevat als default, anders eerste
        titles = [b["title"] for b in branch_items]
        def_idx = 0
        for i, t in enumerate(titles):
            if "nonfood" in t.lower():
                def_idx = i; break
        branch_title = st.selectbox("Branche (CBS 85828NED)", titles, index=def_idx)
        branch_key = str(next(b["key"] for b in branch_items if b["title"] == branch_title))
    else:
        branch_title = st.selectbox("Branche (CBS)", ["DH_NONFOOD","DH_TOTAAL","DH_FOOD"], index=0)
        branch_key = branch_title

# ───────── Macro reeksen (buiten de knop) ─────────
try:
    cci_series = get_cci_series(months_back=months_back, dataset=CBS_DATASET) or []
except Exception as e:
    cci_series = []
    st.info(f"CCI niet beschikbaar: {e}")

try:
    retail_series = []
    if use_retail:
        # probeer key → title → fallback TOTAAL
        retail_series = get_retail_index(branch_code_or_title=branch_key, months_back=months_back) or \
                        get_retail_index(branch_code_or_title=branch_title, months_back=months_back) or \
                        get_retail_index(branch_code_or_title="DH_TOTAAL", months_back=months_back)
except Exception as e:
    retail_series = []
    if use_retail:
        st.info(f"Detailhandelreeks (85828NED) niet beschikbaar: {e}")

# ───────── Helpers: parsing, dates, aggregaties, baseline, forecast ─────────
def _parse_shop_meta(val):
    """val kan 'Amsterdam' of een JSON-string zoals {"name":"Amsterdam","postcode":"3811","region":"Noord NL"} zijn."""
    name, postcode, reg = None, None, None
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s.replace("''", '"').replace("“","\"").replace("”","\""))
                name = obj.get("name")
                postcode = obj.get("postcode")
                reg = obj.get("region")
            except Exception:
                name = val
        else:
            name = val
    return name, postcode, reg

def add_effective_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    dt_date = pd.to_datetime(d.get("date"), errors="coerce")
    dt_ts   = pd.to_datetime(d.get("timestamp"), errors="coerce")
    d["date_eff"]  = dt_date.fillna(dt_ts)
    d["ym"]        = d["date_eff"].dt.to_period("M").astype(str)
    # veilige week-helpers (zonder NA → int)
    iso = d["date_eff"].dt.isocalendar()
    d["iso_year"]  = iso.year.astype("Int64")
    d["iso_week"]  = iso.week.astype("Int64")
    d["weekday"]   = d["date_eff"].dt.weekday
    d["date_only"] = d["date_eff"].dt.date
    return d

def fetch_hist_kpis_df(shop_ids, period: str) -> pd.DataFrame:
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
    if df is None or df.empty:
        return {}
    d = add_effective_date_cols(df)
    out = {}
    for wd, g in d.groupby("weekday"):
        stores = {}
        for sid, gs in g.groupby("shop_id"):
            # SPV is al in € per bezoeker; niet vermenigvuldigen
            if "sales_per_visitor" in gs.columns:
                spv_series = gs["sales_per_visitor"]
            else:
                denom = gs["count_in"].replace(0, pd.NA)
                spv_series = (gs["turnover"] / denom).fillna(0)
            stores[str(int(sid))] = {
                "visitors": float(gs["count_in"].mean()),
                "conversion": float(gs["conversion_rate"].mean()),  # al in %
                "spv": float(spv_series.mean()),
            }
        out[int(wd)] = stores
    return out

def monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
    d = add_effective_date_cols(df)
    g = d.groupby("ym", as_index=False).agg(
        visitors=("count_in","sum"),
        turnover=("turnover","sum"),
    )
    # conversie gewogen op bezoekers; let op: conversion_rate staat reeds in procenten
    d["weighted_conv"] = d["conversion_rate"] * d["count_in"]
    conv = d.groupby("ym", as_index=False).agg(conv_num=("weighted_conv","sum"),
                                              conv_den=("count_in","sum"))
    conv["conversion"] = (conv["conv_num"]/conv["conv_den"]).replace([np.inf,-np.inf], np.nan)
    out = g.merge(conv[["ym","conversion"]], on="ym", how="left")
    out["spv"] = (out["turnover"]/out["visitors"]).replace([np.inf,-np.inf], np.nan)
    return out.sort_values("ym")

def mom_yoy(dfm: pd.DataFrame):
    if dfm is None or dfm.empty: return {}
    m = dfm.copy().reset_index(drop=True)
    m["ym_dt"] = pd.to_datetime(m["ym"].astype(str) + "-01", errors="coerce")
    if m["ym_dt"].isna().all(): return {}
    last = m.iloc[-1]
    prev = m.iloc[-2] if len(m) > 1 else None
    yoy_dt = last["ym_dt"] - pd.DateOffset(years=1)
    yoy_row = m.loc[m["ym_dt"] == yoy_dt]
    yoy_row = (yoy_row.iloc[0] if not yoy_row.empty else None)

    def pct(a,b):
        if b in [0,None] or pd.isna(b): return None
        try: return (float(a)/float(b) - 1) * 100
        except Exception: return None

    return {
        "this_label": last["ym_dt"].strftime("%Y-%m"),
        "turnover": float(last.get("turnover", 0)),
        "visitors": float(last.get("visitors", 0)),
        "conversion": float(last.get("conversion", 0)),  # al in %
        "spv": float(last.get("spv", 0)),
        "mom": {
            "turnover": pct(last.get("turnover", 0),  prev.get("turnover", 0))  if prev is not None else None,
            "visitors": pct(last.get("visitors", 0),  prev.get("visitors", 0))  if prev is not None else None,
            "conversion":pct(last.get("conversion",0), prev.get("conversion",0))if prev is not None else None,
            "spv":      pct(last.get("spv",0),        prev.get("spv",0))        if prev is not None else None,
        },
    }

def region_coords_for(region_name: str):
    if region_name and region_name in REGION_COORDS:
        return REGION_COORDS[region_name]
    return DEFAULT_COORDS

def forecast_week_blocks(forecast_days, baseline_day, cci_mm_delta: float):
    """Groepering per ISO-week; heuristische regen/temp-correctie + lichte CCI-correctie op SPV."""
    weeks = defaultdict(lambda: {"visitors":0.0,"turnover":0.0})
    for f in forecast_days:
        wd = pd.to_datetime(f["date"]).weekday()
        base = baseline_day.get(wd, {"visitors":0.0, "spv":0.0})
        base_vis = float(base["visitors"]); base_spv = float(base["spv"])
        # weer-correcties
        pop  = float(f.get("pop",0.0))   # 0..1
        temp = float(f.get("temp",15.0))
        adj_vis = base_vis * (1 - 0.20*pop) * (1 + 0.01*(temp-15.0))
        # cci-correctie: ±0.25% SPV per +1 pnt m/m (klein effect)
        spv = base_spv * (1 + 0.0025*(cci_mm_delta or 0))
        tur = adj_vis * spv

        iso = pd.to_datetime(f["date"]).isocalendar()
        key = f"{int(iso.year)}-W{int(iso.week):02d}"
        weeks[key]["visitors"] += adj_vis
        weeks[key]["turnover"] += tur

    # lijst + sortering
    out = []
    for k, v in weeks.items():
        out.append({"iso_week": k, "visitors": round(v["visitors"]), "turnover": float(v["turnover"])})
    out = sorted(out, key=lambda x: x["iso_week"])
    return out

def _eur0(x): 
    try: return "€{:,.0f}".format(float(x)).replace(",", ".")
    except: return "€0"

def _pct(x):
    return "—" if (x is None or pd.isna(x)) else f"{float(x):+.1f}%"

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
    df_raw = fetch_hist_kpis_df(shop_ids, period_hist)
    if df_raw is None or df_raw.empty:
        st.warning("Geen historische KPI-data voor deze selectie/periode. Probeer ‘this_year’ of ‘last_year’.")
        st.stop()

    # Netto debug-preview met nette kolommen
    dshow = df_raw.copy()
    if "shop_name" in dshow.columns:
        names, pcs, regs = [], [], []
        for v in dshow["shop_name"].astype(str).tolist():
            n, p, r = _parse_shop_meta(v)
            names.append(n); pcs.append(p); regs.append(r)
        dshow["name"] = names; dshow["postcode"] = pcs; dshow["region"] = regs
        # toon meest bruikbare kolommen
        cols = ["date","timestamp","shop_id","name","postcode","region","count_in","conversion_rate","turnover","sales_per_visitor"]
        dshow = dshow[[c for c in cols if c in dshow.columns]]
        # prettiger: timestamp → date als date leeg is
        if ("date" in dshow.columns) and (dshow["date"].isna().any()) and ("timestamp" in dshow.columns):
            dshow["date"] = pd.to_datetime(dshow["timestamp"], errors="coerce").dt.date
        dshow = dshow.rename(columns={"conversion_rate":"conversion_%","sales_per_visitor":"spv"})
    with st.expander("🛠️ Debug — eerste rijen (hist KPI’s)"):
        st.write(dshow.head(15))

    # 2) Baselines + regiotrend
    df = add_effective_date_cols(df_raw)
    baseline = build_weekday_baselines(df)
    dfm = monthly_agg(df)  # visitors / turnover / conversion(%) / spv(€)
    trend = mom_yoy(dfm)

    # 3) Weer + CCI
    lat, lon = region_coords_for(region)
    forecast = get_daily_forecast(lat, lon, OPENWEATHER_KEY, days_ahead)

    try:
        cci_info = get_consumer_confidence(CBS_DATASET) or {}
        cci_last_val = cci_info.get("value", None)
        cci_last_prd = cci_info.get("period", "n/a")
    except Exception:
        cci_last_val, cci_last_prd = None, "n/a"

    # CCI sanity: als duidelijk onjuist (>>100), toon waarschuwing en negeer absolute waarde
    if cci_last_val is not None and abs(float(cci_last_val)) > 100:
        st.warning("CCI-waarde lijkt buiten de gebruikelijke bandbreedte (verwacht ~-60..+40). We gebruiken alleen de trend (m/m) voor correcties.")
        cci_last_val = None

    # m/m delta uit cci_series (laatste 2)
    cci_mm_delta = None
    if cci_series and len(cci_series) >= 2:
        try:
            a = float(str(cci_series[-1]["cci"]).replace(",", "."))
            b = float(str(cci_series[-2]["cci"]).replace(",", "."))
            cci_mm_delta = a - b
        except Exception:
            cci_mm_delta = None

    # 4) Regionale baseline (gem. over winkels per weekday)
    baseline_day = {}
    for wd, storemap in baseline.items():
        if not storemap: continue
        visitors = np.mean([v["visitors"] for v in storemap.values()]) if storemap else 0.0
        spv      = np.mean([v["spv"]      for v in storemap.values()]) if storemap else 0.0
        baseline_day[int(wd)] = {"visitors": float(visitors), "spv": float(spv)}
    if not baseline_day:
        st.warning("Geen baseline beschikbaar (te weinig dagen). Kies een ruimere periode.")
        st.stop()

    # 5) Weekblokken op basis van weer + CCI
    week_blocks = forecast_week_blocks(forecast, baseline_day, cci_mm_delta)
    # maak twee eerstvolgende weken als bullets
    next_two = week_blocks[:2]

    # ── Silver-platter KPIs (MoM)
    st.subheader("🔎 Silver-platter samenvatting (regio)")
    if trend:
        cA, cB, cC, cD = st.columns(4)
        cA.metric(f"Omzet {trend['this_label']}", _eur0(trend['turnover']), _pct(trend['mom']['turnover']))
        cB.metric(f"Bezoekers {trend['this_label']}", f"{trend['visitors']:,.0f}".replace(",", "."), _pct(trend['mom']['visitors']))
        cC.metric(f"Conversie {trend['this_label']}", f"{trend['conversion']:.2f}%", _pct(trend['mom']['conversion']))  # let op: geen *100
        cD.metric(f"SPV {trend['this_label']}", f"€{trend['spv']:.2f}", _pct(trend['mom']['spv']))

    # ── Verwachting op week & maand (samengevat)
    st.subheader("🗺️ Verwachting per week & maand (regio)")
    if next_two:
        bullets = []
        for wb in next_two:
            wk = wb["iso_week"]
            bullets.append(f"• Week {wk}: verwacht {_eur0(wb['turnover'])} omzet en {int(wb['visitors']):,} bezoekers".replace(",", "."))
        st.markdown("\n\n".join(bullets))

    # conclusie-balk
    total_next = sum(w["turnover"] for w in week_blocks)
    tone = "↗ licht opwaarts" if (cci_mm_delta or 0) > 0 else ("↘ licht neerwaarts" if (cci_mm_delta or 0) < 0 else "→ stabiel")
    # simpele duiding weer per eerstvolgende 2 weken
    def week_weather_note(idx):
        if idx >= len(forecast): return "—"
        # neem 7 dagen blok-summary
        return None
    color = "green" if (cci_mm_delta or 0) > 0 else ("red" if (cci_mm_delta or 0) < 0 else "blue")
    st.markdown(
        f'<div style="padding:10px;border-radius:8px;background:{"#ECFDF3" if color=="green" else ("#FEF3F2" if color=="red" else "#EFF6FF")};color:#0C111D;">'
        f'<b>Conclusie:</b> {"opwaarts" if color=="green" else ("neerwaarts" if color=="red" else "stabiel")}. '
        f'Voor de komende {days_ahead} dagen verwachten we in totaal <b>{_eur0(total_next)}</b> omzet. '
        f'CCI-effect: <b>{tone}</b> (lichte SPV-correctie toegepast).'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── CCI tegel + mini-grafiek (dit jaar)
    st.subheader("Consumentenvertrouwen (CBS)")
    if cci_last_val is not None:
        st.metric("Laatste maand (niveau)", f"{float(cci_last_val):.1f}", help=f"Periode: {cci_last_prd}")
    else:
        st.caption(f"Laatste maand: n.v.t. (periode: {cci_last_prd}). We tonen alleen trend.")

    # filter op huidig jaar en maak (maand → waarde)
    yr = date.today().year
    cci_this_year = []
    for it in (cci_series or []):
        p = str(it.get("period") or it.get("Periods") or "")
        # Periodes kunnen als '2025MM10' of '2025M10' komen; pak jaartal en maand
        y = str(p)[:4]
        mm = "".join([ch for ch in str(p) if ch.isdigit()])[-2:]  # laatste 2 digits
        if y.isdigit() and int(y) == yr:
            try:
                v = float(str(it["cci"]).replace(",", "."))
                cci_this_year.append((f"{y}-{mm}", v))
            except Exception:
                pass
    cci_this_year = sorted(cci_this_year, key=lambda x: x[0])

    if cci_this_year:
        with st.expander("📈 CCI dit jaar (maandelijks)"):
            idx = pd.DataFrame(cci_this_year, columns=["ym","cci"]).set_index("ym")
            st.line_chart(idx)
            # m/m delta label
            if len(cci_this_year) >= 2:
                mm_delta = cci_this_year[-1][1] - cci_this_year[-2][1]
                st.caption(f"m/m: {mm_delta:+.1f} punt(en).")
    else:
        st.info("Nog geen CCI-punten voor het huidige jaar in de opgehaalde reeks.")

    # ── CCI vs Omzet (genormaliseerd op eerste overlappende maand)
    st.subheader("📊 CCI vs. Omzet — genormaliseerde trend")
    if not dfm.empty and cci_series:
        # maak omzet per maand (huidig jaar) en koppel aan CCI (zelfde jaar)
        m = dfm.copy()
        m["ym_dt"] = pd.to_datetime(m["ym"] + "-01", errors="coerce")
        m = m[m["ym_dt"].dt.year == yr][["ym","turnover"]]

        cci_df = []
        for it in cci_series:
            p = str(it.get("period") or it.get("Periods") or "")
            y = str(p)[:4]
            mm = "".join([ch for ch in str(p) if ch.isdigit()])[-2:]
            if y.isdigit() and int(y) == yr:
                try:
                    v = float(str(it["cci"]).replace(",", "."))
                    cci_df.append({"ym": f"{y}-{mm}", "cci": v})
                except: 
                    pass
        cci_df = pd.DataFrame(cci_df)

        if not cci_df.empty and not m.empty:
            merged = m.merge(cci_df, on="ym", how="inner").sort_values("ym")
            if len(merged) >= 2:
                base_turn = merged["turnover"].iloc[0]
                base_cci  = merged["cci"].iloc[0]
                merged["Omzet_idx"] = (merged["turnover"]/base_turn)*100 if base_turn else np.nan
                merged["CCI_idx"]   = (merged["cci"]/base_cci)*100      if base_cci else np.nan
                idx = merged.set_index("ym")[["CCI_idx","Omzet_idx"]]
                st.line_chart(idx)
                st.caption("Beide reeksen als index (100 = eerste gezamenlijke maand) om samenloop zichtbaar te maken.")
            else:
                st.info("Te weinig overlap tussen CCI-maanden en omzetmaanden om een trend te tonen.")
        else:
            st.info("Te weinig overlap tussen CCI-maanden en omzetmaanden om een trend te tonen.")
    else:
        st.info("CCI of omzetreeks ontbreekt; trend niet getoond.")

    # ── Macro tiles optioneel detailhandel
    if use_retail:
        if retail_series:
            last_r = retail_series[-1]
            st.metric(f"Detailhandel ({last_r['branch']}) — {last_r['series']}", f"{last_r['retail_value']:.1f}")
            with st.expander("🛍️ Detailhandel reeks (CBS 85828NED)"):
                st.line_chart({"Retail": [x["retail_value"] for x in retail_series]})
        else:
            st.info("Geen detailhandelreeks gevonden voor deze branche/periode (85828NED).")
