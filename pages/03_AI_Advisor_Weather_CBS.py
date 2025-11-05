# pages/03_AI_Advisor_Weather_CBS.py
import os, sys, json, ast
import pandas as pd
import streamlit as st

# ───────── Mapping & API wrapper ─────────
try:
    from shop_mapping import SHOP_NAME_MAP  # {shop_id: "Name" | {"name":..,"postcode":..,"region":..}}
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
    get_consumer_confidence,   # laatste maand {period, value}
    get_cci_series,            # [{'period':'YYYYMM','cci':float}, ...]
)

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
period_hist = st.selectbox("Historische periode", ["last_month", "this_year", "last_year"], index=0)
days_ahead = st.slider("Vooruitblik (dagen)", 7, 30, 14, help="Gebruik voor week-/maandverwachtingen.")

# ───────── Macro-context (CCI) ─────────
months_back = st.slider("CCI: maanden terug", 6, 36, 18, help="Voor grafiek/vergelijking met omzet.")

def _coerce_jsonish(x):
    """'{"name":"A","postcode":"1234","region":"Noord NL"}' → dict | 'Amsterdam' → {'name':'Amsterdam'}"""
    if isinstance(x, dict): return x
    if not isinstance(x, str) or not x: return {"name": str(x or "")}
    s = x.strip()
    try:
        # sommige payloads hebben verdubbelde quotes
        if s.count('""') > 0:
            s = s.replace('""', '"')
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return {"name": s}

def add_effective_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    dt_date = pd.to_datetime(d.get("date"), errors="coerce")
    dt_ts   = pd.to_datetime(d.get("timestamp"), errors="coerce")
    d["date_eff"]  = dt_date.fillna(dt_ts)
    d["weekday"]   = d["date_eff"].dt.weekday
    d["ym"]        = d["date_eff"].dt.to_period("M").astype(str)
    d["date_only"] = d["date_eff"].dt.date
    # ISO week veilig (Int64 laat NA toe)
    try:
        d["iso_week"] = d["date_eff"].dt.isocalendar().week.astype("Int64")
    except Exception:
        d["iso_week"] = pd.Series([pd.NA]*len(d), dtype="Int64")
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
    if df is None or df.empty: return {}
    d = add_effective_date_cols(df)
    out = {}
    for wd, g in d.groupby("weekday"):
        stores = {}
        for sid, gs in g.groupby("shop_id"):
            # SPV robuust
            if "sales_per_visitor" in gs.columns:
                spv_series = gs["sales_per_visitor"]
            else:
                denom = gs["count_in"].replace(0, pd.NA)
                spv_series = (gs["turnover"]/denom).fillna(0)
            stores[int(sid)] = {
                "visitors":     float(gs["count_in"].mean()),
                "conversion":   float(gs["conversion_rate"].mean()),
                "spv":          float(spv_series.mean()),
            }
        out[wd] = stores
    return out

def monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
    d = add_effective_date_cols(df)
    g = d.groupby("ym", as_index=False).agg(visitors=("count_in","sum"), turnover=("turnover","sum"))
    d["weighted_conv"] = d["conversion_rate"] * d["count_in"]
    conv = d.groupby("ym", as_index=False).agg(conv_num=("weighted_conv","sum"),
                                              conv_den=("count_in","sum"))
    conv["conversion"] = (conv["conv_num"]/conv["conv_den"]).fillna(0)
    out = g.merge(conv[["ym","conversion"]], on="ym", how="left")
    out["spv"] = (out["turnover"]/out["visitors"]).replace([float("inf")], 0).fillna(0)
    return out

def mom_yoy(dfm: pd.DataFrame):
    if dfm is None or dfm.empty: return {}
    m = dfm.copy().sort_values("ym").reset_index(drop=True)
    m["ym_dt"] = pd.to_datetime(m["ym"].astype(str) + "-01", errors="coerce")
    if m["ym_dt"].isna().all(): return {}
    last = m.iloc[-1]
    prev = m.iloc[-2] if len(m) > 1 else None
    yoy_dt  = last["ym_dt"] - pd.DateOffset(years=1)
    yoy_row = m.loc[m["ym_dt"] == yoy_dt]
    yoy_row = yoy_row.iloc[0] if not yoy_row.empty else None
    def pct(a,b):
        if b in [0,None] or pd.isna(b) or float(b)==0: return None
        try: return (float(a)/float(b) - 1) * 100
        except Exception: return None
    return {
        "this_label": last["ym_dt"].strftime("%Y-%m"),
        "prev_label": (prev["ym_dt"].strftime("%Y-%m") if prev is not None else "n.v.t."),
        "visitors":   float(last.get("visitors",0)),
        "turnover":   float(last.get("turnover",0)),
        "conversion": float(last.get("conversion",0)),
        "spv":        float(last.get("spv",0)),
        "mom": {
            "turnover":   pct(last.get("turnover",0),   prev.get("turnover",0))   if prev is not None else None,
            "visitors":   pct(last.get("visitors",0),   prev.get("visitors",0))   if prev is not None else None,
            "conversion": pct(last.get("conversion",0), prev.get("conversion",0)) if prev is not None else None,
            "spv":        pct(last.get("spv",0),        prev.get("spv",0))        if prev is not None else None,
        },
        "yoy": {
            "turnover":   pct(last.get("turnover",0),   yoy_row.get("turnover",0))   if yoy_row is not None else None,
            "visitors":   pct(last.get("visitors",0),   yoy_row.get("visitors",0))   if yoy_row is not None else None,
            "conversion": pct(last.get("conversion",0), yoy_row.get("conversion",0)) if yoy_row is not None else None,
            "spv":        pct(last.get("spv",0),        yoy_row.get("spv",0))        if yoy_row is not None else None,
        },
    }

def estimate_weather_uplift(baseline_day: dict, forecast_days: list[dict]) -> dict:
    daily = []
    for f in forecast_days:
        wd = pd.to_datetime(f["date"]).weekday()
        base = baseline_day.get(wd, {"visitors": 0.0, "spv": 0.0})
        base_vis, spv = float(base["visitors"]), float(base["spv"])
        pop  = float(f.get("pop", 0.0))
        temp = float(f.get("temp", 15.0))
        adj_vis = base_vis * (1 - 0.20*pop) * (1 + 0.01*(temp-15.0))
        row = {
            "date": f["date"], "weekday": wd, "temp": temp, "pop": pop,
            "base_turnover": base_vis * spv, "adj_turnover": adj_vis * spv
        }
        row["delta_turnover"] = row["adj_turnover"] - row["base_turnover"]
        daily.append(row)
    base_total = sum(x["base_turnover"] for x in daily)
    adj_total  = sum(x["adj_turnover"]  for x in daily)
    delta_total = adj_total - base_total
    delta_pct = (delta_total/base_total*100) if base_total else None
    return {"daily": daily, "base_total": base_total, "adj_total": adj_total,
            "delta_total": delta_total, "delta_pct": delta_pct}

def _eur0(x): 
    try: return "€{:,.0f}".format(float(x)).replace(",", ".")
    except: return "€0"

def _fmt_pct(x):
    return "—" if (x is None or pd.isna(x)) else f"{x:+.1f}%"

def _cci_df(cci_series):
    """→ DataFrame met kolommen ym (YYYY-MM), cci (float)."""
    if not cci_series: 
        return pd.DataFrame()
    rows=[]
    for it in cci_series:
        per = str(it.get("period",""))
        # '2025MM11' of '202511' → YYYY-MM
        if "MM" in per:
            y = per[:4]; m = per[-2:]
        else:
            y = per[:4]; m = per[-2:]
        ym = f"{y}-{m}"
        try:
            val = float(it.get("cci"))
        except Exception:
            try:
                val = float(str(it.get("cci")).replace(",", "."))
            except Exception:
                continue
        rows.append({"ym": ym, "cci": val})
    df = pd.DataFrame(rows).dropna().sort_values("ym").reset_index(drop=True)
    return df

def _merge_cci_vs_turnover(dfm_in: pd.DataFrame, cci_series_in):
    cci_df = _cci_df(cci_series_in)
    if dfm_in is None or dfm_in.empty or cci_df.empty:
        return pd.DataFrame()
    m = dfm_in[["ym","turnover"]].merge(cci_df, on="ym", how="inner")
    if m.empty: 
        return pd.DataFrame()
    # rebase naar 100 op eerste overlappende maand
    m = m.sort_values("ym").reset_index(drop=True)
    base_t = float(m["turnover"].iloc[0]) or 1.0
    base_c = float(m["cci"].iloc[0]) or 1.0
    m["Omzet_idx"] = m["turnover"] / base_t * 100.0
    m["CCI_idx"]   = m["cci"]      / base_c * 100.0
    return m[["ym","Omzet_idx","CCI_idx"]]

# ───────── Shop selectie ─────────
st.caption("Selecteer regio en druk op de knop om aanbevelingen te genereren.")
if region == "ALL":
    shop_ids = sorted([int(k) for k in (SHOP_NAME_MAP or ID_TO_NAME).keys()])
else:
    shop_ids = get_ids_by_region(region) or sorted([int(k) for k in (SHOP_NAME_MAP or ID_TO_NAME).keys()])

st.write(f"**{len(shop_ids)}** winkels geselecteerd in regio: **{region}**")
st.text(f"ShopIDs → {shop_ids[:25]}{' …' if len(shop_ids)>25 else ''}")

# ───────── CCI buiten de knop (voor tiles) ─────────
try:
    cci_series = get_cci_series(months_back=months_back, dataset=CBS_DATASET)
except Exception as e:
    cci_series = []
    st.info(f"CCI niet beschikbaar: {e}")

# ───────── Action ─────────
if st.button("Genereer aanbevelingen"):
    # 1) Historische data
    df = fetch_hist_kpis_df(shop_ids, period_hist)

    # Debug-tabel netjes: parse shop_name in kolommen
    with st.expander("🛠️ Debug — eerste rijen (hist KPI’s)"):
        ddebug = add_effective_date_cols(df).copy()
        meta = ddebug["shop_name"].apply(_coerce_jsonish)
        ddebug["name"]    = meta.apply(lambda m: m.get("name",""))
        ddebug["postcode"]= meta.apply(lambda m: str(m.get("postcode","")))
        ddebug["region"]  = meta.apply(lambda m: m.get("region",""))
        ddebug = ddebug.drop(columns=["shop_name","date"]).rename(columns={"timestamp":"date"})
        st.write(ddebug.head(15))

    if df is None or df.empty:
        st.warning("Geen historische KPI-data voor deze selectie/periode. Probeer ‘this_year’ of ‘last_year’.")
        st.stop()

    # 2) Baselines + regiotrend
    df = add_effective_date_cols(df)
    baseline = build_weekday_baselines(df)
    dfm = monthly_agg(df)
    trend = mom_yoy(dfm)

    # 3) Weer + CCI
    # (weer: zelfde H/V functie; days_ahead uit slider)
    forecast = get_daily_forecast(52.37, 4.90, OPENWEATHER_KEY, int(days_ahead))  # centroid NL (UI zonder velden)
    try:
        cci_info = get_consumer_confidence(CBS_DATASET)
        last_cci = float(cci_info.get("value", 0.0))
        cci_period = cci_info.get("period","n/a")
    except Exception:
        last_cci, cci_period = 0.0, "n/a"

    # 4) Regionale baseline (gemiddelden vd winkels per weekday)
    baseline_day = {}
    for wd, storemap in baseline.items():
        if not storemap: continue
        visitors = pd.Series([v["visitors"] for v in storemap.values()]).mean()
        spv      = pd.Series([v["spv"]      for v in storemap.values()]).mean()
        baseline_day[wd] = {"visitors": float(visitors), "spv": float(spv)}

    # 5) Verwachting met weer-correctie
    wx = estimate_weather_uplift(baseline_day, forecast)

    # ── SILVER-PLATTER ---------------------------------------------------------
    st.subheader("🔎 Silver-platter samenvatting (regio)")
    if trend:
        this_lab = trend.get("this_label","-")
        prev_lab = trend.get("prev_label","-")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric(f"Omzet {this_lab}", _eur0(trend.get('turnover',0)), _fmt_pct(trend['mom'].get('turnover')))
        c2.metric(f"Bezoekers {this_lab}", f"{trend.get('visitors',0):,.0f}".replace(",", "."), _fmt_pct(trend['mom'].get('visitors')))
        c3.metric(f"Conversie {this_lab}", f"{trend.get('conversion',0)*100:.2f}%", _fmt_pct(trend['mom'].get('conversion')))
        c4.metric(f"SPV {this_lab}", f"€{trend.get('spv',0):.2f}", _fmt_pct(trend['mom'].get('spv')))

    # Week- & maandverwachting
    # groepeer forecast in kalenderweken (simpel: neem 7-daagse slices)
    if wx and wx["daily"]:
        # weken (2 x 7 dagen)
        wk1 = round(sum(d["adj_turnover"] for d in wx["daily"][:7]), 2)
        wk2 = round(sum(d["adj_turnover"] for d in wx["daily"][7:14]), 2)
        vis1 = round(sum(baseline_day.get(pd.to_datetime(d["date"]).weekday(), {"visitors":0})["visitors"] for d in wx["daily"][:7]))
        vis2 = round(sum(baseline_day.get(pd.to_datetime(d["date"]).weekday(), {"visitors":0})["visitors"] for d in wx["daily"][7:14]))
        st.subheader("🗺️ Verwachting per week & maand (regio)")
        st.markdown(f"- Week 1: verwacht **{_eur0(wk1)}** omzet en **{vis1:,}** bezoekers".replace(",", "."))
        st.markdown(f"- Week 2: verwacht **{_eur0(wk2)}** omzet en **{vis2:,}** bezoekers".replace(",", "."))
        # conclusie kleur: neerwaarts = rood, opwaarts = groen
        tone = st.error if (wx.get("delta_pct",0) or 0) < 0 else st.success
        tone(
            f"**Conclusie:** {'neerwaarts' if (wx.get('delta_pct',0) or 0) < 0 else 'opwaarts'}. "
            f"Voor de komende {days_ahead} dagen verwachten we **{_eur0(wx['adj_total'])}** omzet. "
            "Weercomponent: regen/temperatuur t.o.v. normaal; CCI-effect: lichte SPV-correctie."
        )

    # ── CCI tegel + uitleg -----------------------------------------------------
    st.metric("Consumentenvertrouwen (CBS)", f"{last_cci:.1f}", help="CCI is een sentiment-indicator (hoger = kooplustiger). Toon: MoM-verandering.")
    cci_df = _cci_df(cci_series)
    if not cci_df.empty:
        if len(cci_df) >= 2:
            delta = cci_df["cci"].iloc[-1] - cci_df["cci"].iloc[-2]
            st.caption(f"YoY/MoM: Periode: {cci_period} • Laatste verandering: {delta:+.1f} punt(en).")
        with st.expander("📈 CCI reeks"):
            st.line_chart(cci_df.set_index("ym")["cci"].rename("CCI"))

    # ── Komende dagen — acties & weer (compact) --------------------------------
    with st.expander("📅 Komende dagen — acties & weer"):
        if isinstance(forecast, list):
            for f in forecast:
                st.write(f"- {f['date']}: {f.get('temp','?')}°C, regen {int(f.get('pop',0)*100)}%")

    # ── CCI vs Omzet (genormaliseerd) ------------------------------------------
    st.subheader("📊 CCI vs. Omzet — genormaliseerde trend")
    merged = _merge_cci_vs_turnover(dfm, cci_series)
    if merged.empty or len(merged) < 2:
        st.info("Te weinig overlap tussen CCI-maanden en omzetmaanden om een trend te tonen.")
    else:
        st.line_chart(merged.set_index("ym")[["CCI_idx","Omzet_idx"]])
