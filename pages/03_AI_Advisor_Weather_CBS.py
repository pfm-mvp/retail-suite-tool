# pages/03_AI_Advisor_Weather_CBS.py
import os, sys, json, re
import pandas as pd
import numpy as np
import streamlit as st

# ───────── Mapping & API wrapper ─────────
try:
    from shop_mapping import SHOP_NAME_MAP
except Exception:
    SHOP_NAME_MAP = None

from utils_pfmx import api_get_report, friendly_error

# ───────── Page setup ─────────
st.set_page_config(page_title="AI Advisor — Weer + CBS", page_icon="🧭", layout="wide")
st.title("🧭 AI Advisor — Weer + CBS (v3)")

# ───────── Project helpers ─────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from helpers_normalize import normalize_vemcount_response
from helpers_shop import ID_TO_NAME, get_ids_by_region, REGIONS

# ───────── Services ─────────
from services.weather_service import get_daily_forecast
from services.cbs_service import get_consumer_confidence, get_cci_series, get_retail_index, list_retail_branches

try:
    from services.advisor import build_advice
except Exception:
    def build_advice(company, baseline, forecast, cci):
        days = []
        for f in forecast:
            days.append({"date": f["date"], "weather": {"temp": f.get("temp",0), "pop": f.get("pop",0)}, "stores": []})
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
days_ahead = st.slider("Dagen vooruit (weer-forecast)", min_value=7, max_value=30, value=14, step=1)
period_hist = st.selectbox("Historische periode (baseline/trend)", ["last_month","this_year","last_year"], index=0)

st.subheader("Macro-context (CBS)")
c1, c2, c3 = st.columns([1,1,2])
with c1:
    months_back = st.slider("Maanden terug (CCI)", 6, 36, 18)
with c2:
    use_retail = st.checkbox("Toon detailhandel-index (85828NED)", value=False)
with c3:
    dim_name, branch_items = list_retail_branches("85828NED")
    if branch_items:
        title_to_key = {b["title"]: str(b["key"]) for b in branch_items}
        titles = list(title_to_key.keys())
        df_idx = next((i for i,t in enumerate(titles) if "nonfood" in t.lower()), 0)
        branch_title = st.selectbox("Branche (CBS 85828NED)", titles, index=df_idx, disabled=not use_retail)
        branch_key = title_to_key[branch_title]
    else:
        branch_title = st.selectbox("Branche (CBS 85828NED)", ["NONFOOD","FOOD","DH_TOTAAL"], index=0, disabled=not use_retail)
        branch_key = branch_title

# ───────── Macro (buiten de knop) ─────────
def _parse_cbs_period(p: str) -> str:
    if isinstance(p, str) and "MM" in p:
        y = p.split("MM")[0]; m = p.split("MM")[1]
        return f"{y}-{m}"
    return str(p)

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
                        get_retail_index(branch_code_or_title="NONFOOD", months_back=months_back)
except Exception as e:
    retail_series = []
    if use_retail:
        st.info(f"Detailhandelreeks niet beschikbaar: {e}")

# ───────── Helpers ─────────
def add_effective_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    dt_date = pd.to_datetime(d.get("date"), errors="coerce")
    dt_ts   = pd.to_datetime(d.get("timestamp"), errors="coerce")
    d["date_eff"]  = dt_date.fillna(dt_ts)
    d["weekday"]   = d["date_eff"].dt.weekday
    d["ym"]        = d["date_eff"].dt.to_period("M").astype(str)
    d["date_only"] = d["date_eff"].dt.date
    cal = d["date_eff"].dt.isocalendar()
    d["iso_week"]  = cal.week.astype("Int64")   # NA-friendly
    d["iso_year"]  = cal.year.astype("Int64")
    return d

def fetch_hist_kpis_df(shop_ids, period: str) -> pd.DataFrame:
    metrics = ["count_in","conversion_rate","turnover","sales_per_visitor"]
    params = [("data", int(sid)) for sid in shop_ids]
    params += [("data_output", k) for k in metrics]
    params += [("source","shops"), ("period",period), ("step","day")]
    js = api_get_report(params)
    if friendly_error(js, period):
        return pd.DataFrame()
    return normalize_vemcount_response(js, SHOP_NAME_MAP or ID_TO_NAME, kpi_keys=metrics)

def build_weekday_baselines(df: pd.DataFrame) -> dict:
    if df is None or df.empty: return {}
    d = add_effective_date_cols(df)
    out = {}
    for wd, g in d.groupby("weekday"):
        stores = {}
        for sid, gs in g.groupby("shop_id"):
            name = ID_TO_NAME.get(int(sid), f"Shop {sid}")
            spv_series = gs["sales_per_visitor"] if "sales_per_visitor" in gs.columns \
                        else (gs["turnover"]/gs["count_in"].replace(0, pd.NA)).fillna(0)
            conv_ratio = (gs["conversion_rate"]/100.0)
            stores[name] = {
                "visitors":   float(gs["count_in"].mean()),
                "conversion": float(conv_ratio.mean()),
                "spv":        float(spv_series.mean()),
            }
        out[wd] = stores
    return out

def monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
    d = add_effective_date_cols(df)
    g = d.groupby("ym", as_index=False).agg(visitors=("count_in","sum"), turnover=("turnover","sum"))
    d["weighted_conv"] = (d["conversion_rate"]/100.0) * d["count_in"]
    conv = d.groupby("ym", as_index=False).agg(conv_num=("weighted_conv","sum"),
                                              conv_den=("count_in","sum"))
    conv["conversion"] = (conv["conv_num"]/conv["conv_den"]).fillna(0)
    out = g.merge(conv[["ym","conversion"]], on="ym", how="left")
    out["spv"] = (out["turnover"]/out["visitors"]).replace([np.inf],0).fillna(0)
    return out

def mom_last_month(dfm: pd.DataFrame) -> dict:
    if dfm is None or dfm.empty or len(dfm) < 2: return {}
    m = dfm.sort_values("ym").reset_index(drop=True).copy()
    last = m.iloc[-1]; prev = m.iloc[-2]
    def pct(a,b): 
        try: return (float(a)/float(b)-1)*100 if float(b)!=0 else None
        except: return None
    return {
        "ym": last["ym"],
        "turnover_mom":   pct(last["turnover"],  prev["turnover"]),
        "visitors_mom":   pct(last["visitors"],  prev["visitors"]),
        "conversion_mom": pct(last["conversion"],prev["conversion"]),
        "spv_mom":        pct(last["spv"],       prev["spv"]),
        "last_values":    last.to_dict()
    }

# postcode → weercoördinaten (PC2 + regiofallback)
REGION_COORDS = {"Noord NL":(53.219,6.566),"Midden NL":(52.090,5.121),"Zuid NL":(51.441,5.469),
                 "West NL":(52.373,4.900),"Oost NL":(52.223,6.000),"ALL":(52.373,4.900)}
PC2_TO_COORD = {"10":(52.37,4.90),"11":(52.37,4.90),"20":(52.0,4.36),"21":(52.0,4.36),"22":(52.0,4.36),
                "23":(52.39,4.64),"24":(52.39,4.64),"25":(52.39,4.64),"26":(52.39,4.64),"27":(52.08,4.30),
                "28":(52.08,4.30),"29":(51.85,4.28),"30":(51.92,4.48),"31":(51.92,4.48),"32":(51.92,4.48),
                "33":(51.83,4.67),"34":(52.09,5.12),"35":(52.13,5.19),"36":(52.35,5.22),"37":(52.13,5.19),
                "38":(52.51,6.09),"39":(52.51,6.09),"80":(52.52,6.09),"81":(52.38,6.64),"82":(52.50,5.47),
                "83":(52.70,5.75),"90":(53.20,6.57),"91":(53.20,6.57),"92":(53.19,5.79),"93":(53.05,5.67),
                "94":(52.99,6.56),"95":(53.32,6.92),"96":(53.10,6.00),"97":(53.22,6.56),"98":(53.22,6.56),"99":(53.22,6.56),
                "50":(51.44,5.47),"51":(51.44,5.47),"52":(51.59,5.08),"53":(51.59,5.08),"54":(51.59,5.08),
                "55":(51.59,5.08),"56":(51.44,5.47),"57":(51.65,5.05),"58":(51.56,5.09),
                "60":(51.44,6.17),"61":(51.23,5.99),"62":(50.85,5.69),"63":(50.85,5.69),"64":(50.85,5.69)}
PC2_RE = re.compile(r"^\s*(\d{2})")
def _pc2_from_text(txt): 
    m = PC2_RE.match(str(txt)) if txt else None
    return m.group(1) if m else None

def _extract_postcodes_from_df(df: pd.DataFrame) -> list[str]:
    out=[]
    if "shop_name" not in df.columns: return out
    for v in df["shop_name"].dropna().astype(str).head(500):
        pc=None
        s = v.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                j = json.loads(s)
            except Exception:
                try:
                    j = json.loads(s.replace('""','"'))
                except Exception:
                    j = None
            if isinstance(j, dict):
                pc = j.get("postcode")
        if not pc:
            m = re.search(r"\b(\d{4})\s*[A-Z]{0,2}\b", s)
            pc = m.group(1) if m else None
        if pc:
            pc2 = _pc2_from_text(pc)
            if pc2: out.append(pc2)
    return out

def pick_coords_for_weather(region_label: str, df_hist: pd.DataFrame) -> tuple[float,float]:
    pc2_list = _extract_postcodes_from_df(df_hist)
    if pc2_list:
        seen, latlons = set(), []
        for pc2 in pc2_list:
            if pc2 in PC2_TO_COORD and pc2 not in seen:
                latlons.append(PC2_TO_COORD[pc2]); seen.add(pc2)
        if latlons:
            lat = sum(x[0] for x in latlons)/len(latlons)
            lon = sum(x[1] for x in latlons)/len(latlons)
            return lat, lon
    return REGION_COORDS.get(region_label, REGION_COORDS["ALL"])

# kleine helpers
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
    if df is None or df.empty:
        st.warning("Geen historische KPI-data voor deze selectie/periode. Probeer ‘this_year’ of ‘last_year’.")
        st.stop()

    # Debug-tabel: shop_name → nette kolommen; timestamp → date
    ddbg = df.copy()
    def parse_shopname(x):
        if isinstance(x, dict):
            return x.get("name"), x.get("postcode"), x.get("region")
        if isinstance(x, str):
            s = x.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    j = json.loads(s)
                except Exception:
                    try:
                        j = json.loads(s.replace('""','"'))
                    except Exception:
                        j = None
                if isinstance(j, dict):
                    return j.get("name") or j.get("store_name") or s, j.get("postcode"), j.get("region")
            return s, None, None
        return x, None, None

    triples = ddbg["shop_name"].apply(parse_shopname)
    ddbg[["name","postcode","region"]] = pd.DataFrame(triples.tolist(), index=ddbg.index)
    ddbg["date"] = pd.to_datetime(ddbg.get("timestamp"), errors="coerce").dt.date
    show_cols = ["date","shop_id","name","postcode","region","count_in","conversion_rate","turnover","sales_per_visitor"]
    with st.expander("🛠️ Debug — eerste rijen (hist KPI’s)"):
        st.dataframe(ddbg[show_cols].head(15), use_container_width=True)

    # 2) Baselines + regiotrend
    dfW = add_effective_date_cols(df)
    baseline = build_weekday_baselines(dfW)
    dfm = monthly_agg(dfW)

    # Fallback voor samenvatting: als laatste maand 0 is, probeer last_month extra op te halen
    dfm_for_tiles = dfm.copy()
    if dfm_for_tiles.empty or (dfm_for_tiles.sort_values("ym").iloc[-1][["turnover","visitors"]]==0).all():
        df_fb = fetch_hist_kpis_df(shop_ids, "last_month")
        if not df_fb.empty:
            dfm_for_tiles = monthly_agg(add_effective_date_cols(df_fb))

    mom = mom_last_month(dfm_for_tiles)

    # 3) Weer & CCI
    lat, lon = pick_coords_for_weather(region, df)
    forecast = get_daily_forecast(lat, lon, OPENWEATHER_KEY, days_ahead)

    # CCI: gebruik laatste uit reeks; zo niet, fallback op endpoint
    if cci_series:
        last_cci = float(cci_series[-1].get("cci", 0.0))
        last_cci_period = cci_series[-1].get("period","")
    else:
        try:
            cci_info = get_consumer_confidence(CBS_DATASET)
            last_cci = float(cci_info.get("value", 0.0))
            last_cci_period = cci_info.get("period","")
        except Exception:
            last_cci, last_cci_period = 0.0, "n/a"

    def cci_trend_signal(series):
        if not series: return 0.0
        vals = [float(x.get("cci",0)) for x in series[-3:]]
        if len(vals) < 2: return 0.0
        return vals[-1] - vals[0]
    cci_slope = cci_trend_signal(cci_series)

    # 4) Adviesregels (dag/winkel)
    advice = build_advice("Your Company", baseline, forecast, last_cci)

    # 5) Regionale baseline (gemiddelden per weekday)
    baseline_day = {}
    for wd, storemap in baseline.items():
        if not storemap: continue
        visitors = pd.Series([v["visitors"] for v in storemap.values()]).mean()
        spv      = pd.Series([v["spv"]      for v in storemap.values()]).mean()
        baseline_day[wd] = {"visitors": float(visitors), "spv": float(spv)}
    if not baseline_day:
        st.warning("Geen baseline beschikbaar (te weinig dagen). Kies een ruimere periode.")
        st.stop()

    # 6) Prognose komende weken/maand
    cci_factor = 1.0 + 0.01 * (cci_slope/5.0)  # ±2% bij ±10pt/3 mnd
    proj_rows = []
    for f in forecast:
        dte = pd.to_datetime(f["date"])
        wd  = int(dte.weekday())
        base = baseline_day.get(wd, {"visitors":0.0, "spv":0.0})
        base_vis = float(base["visitors"]); base_spv = float(base["spv"]) * cci_factor
        pop  = float(f.get("pop",0.0))
        temp = float(f.get("temp",15.0))
        adj_vis = base_vis * (1 - 0.20*pop) * (1 + 0.01*(temp-15.0))
        adj_turn = adj_vis * base_spv
        cal = dte.isocalendar()
        proj_rows.append({
            "date": dte, "iso_year": int(cal.year), "iso_week": int(cal.week),
            "visitors": adj_vis, "turnover": adj_turn
        })
    proj = pd.DataFrame(proj_rows) if proj_rows else pd.DataFrame(columns=["date","iso_year","iso_week","visitors","turnover"])
    weekly = proj.groupby(["iso_year","iso_week"], as_index=False).agg(visitors=("visitors","sum"),
                                                                      turnover=("turnover","sum"))
    weekly["week_label"] = weekly["iso_year"].astype(str) + "-W" + weekly["iso_week"].astype(str)
    month_total = proj["turnover"].sum()

    # ── Silver-platter regio (robust)
    st.subheader("🔎 Silver-platter samenvatting (regio)")
    if not dfm_for_tiles.empty:
        last = dfm_for_tiles.sort_values("ym").iloc[-1]
        def _safe_metric(val, fmt):
            try: return fmt(val)
            except: return "n.v.t."
        colA, colB, colC, colD = st.columns(4)
        colA.metric(f"Omzet {last['ym']}", _safe_metric(last['turnover'], _eur),
                    _fmt_pct(mom.get("turnover_mom")) if mom else "—")
        colB.metric("Bezoekers", f"{float(last['visitors']):,.0f}".replace(",", "."),
                    _fmt_pct(mom.get("visitors_mom")) if mom else "—")
        colC.metric("Conversie", f"{float(last['conversion'])*100:.2f}%",
                    _fmt_pct(mom.get("conversion_mom")) if mom else "—")
        colD.metric("SPV", f"€{float(last['spv']):.2f}",
                    _fmt_pct(mom.get("spv_mom")) if mom else "—")
    else:
        st.info("Geen maandsamenvatting beschikbaar voor deze periode.")

    # ── Verwachting per week & maand (tekst + kleur)
    st.subheader("🗺️ Verwachting per week & maand (regio)")
    if not weekly.empty:
        wk_lines = [f"- Week {int(r['iso_week'])}: verwacht {_eur(r['turnover'])} omzet en {int(round(r['visitors']))} bezoekers"
                    for _, r in weekly.iterrows()]
        st.markdown("\n".join(wk_lines))

        trend_flag = "flat"
        if len(weekly) >= 2 and weekly.iloc[0]["turnover"] > 0:
            pct = (weekly.iloc[-1]["turnover"] - weekly.iloc[0]["turnover"]) / weekly.iloc[0]["turnover"] * 100
            trend_flag = "up" if pct > 3 else ("down" if pct < -3 else "flat")

        cci_dir = "↗" if cci_slope > 1 else ("↘" if cci_slope < -1 else "→")
        msg = (f"**Conclusie:** De komende weken ogen **opwaarts**. " if trend_flag=="up" else
               f"**Conclusie:** De komende weken ogen **neerwaarts**. " if trend_flag=="down" else
               f"**Conclusie:** De komende weken ogen **vlak**. ")
        msg += f"Voor de komende {days_ahead} dagen verwachten we **{_eur(month_total)}** omzet. "
        msg += f"CCI-trend (3 mnd): **{cci_dir}** — lichte SPV-correctie toegepast."

        if trend_flag=="up":
            st.success(msg)
        elif trend_flag=="down":
            st.error(msg)
        else:
            st.warning(msg)
    else:
        st.info("Geen weekprognose beschikbaar.")

    # ── CCI (duidelijke help)
    cci_help = ("CCI = Consumentenvertrouwen (CBS 83693NED). Rond 0 = neutraal; "
                "boven 0 positiever sentiment (meer kooplust), onder 0 negatiever. "
                "We passen een kleine SPV-correctie toe o.b.v. recente CCI-trend.")
    st.metric("Consumentenvertrouwen (CBS)", f"{last_cci:.1f}", help=cci_help)

    # ── Dag-advies (ingekort)
    with st.expander("📅 Komende dagen — acties & weer", expanded=False):
        for d in advice["days"]:
            w = d.get("weather", {})
            temp = w.get("temp",0.0); pop = int((w.get("pop",0.0) or 0)*100)
            st.write(f"• {d['date']}: {temp:.1f}°C, regen {pop}%")

    # ── CCI vs. Omzet — alleen tonen bij ≥2 overlappende maanden
    st.subheader("📊 CCI vs. Omzet — genormaliseerde trend")
    def _merge_cci_vs_turnover(dfm_in, cci_series_in):
        if not dfm_in or dfm_in.empty or not cci_series_in: return pd.DataFrame()
        df_cci = pd.DataFrame([{"ym": _parse_cbs_period(x.get("period","")), "CCI": float(x.get("cci",0))} for x in cci_series_in])
        df_cci = df_cci.dropna().copy()
        dfm2 = dfm_in[["ym","turnover"]].copy()
        merged = pd.merge(dfm2, df_cci, on="ym", how="inner").sort_values("ym")
        return merged

    merged = _merge_cci_vs_turnover(dfm_for_tiles, cci_series)
    if not merged.empty and len(merged) >= 2:
        merged["Omzet_idx"] = (merged["turnover"]/merged["turnover"].iloc[0])*100.0
        base_cci = merged["CCI"].iloc[0] if merged["CCI"].iloc[0] != 0 else 1.0
        merged["CCI_idx"] = (merged["CCI"]/base_cci)*100.0
        st.line_chart(merged.set_index("ym")[["CCI_idx","Omzet_idx"]].rename(columns={"CCI_idx":"CCI_idx","Omzet_idx":"Omzet_idx"}),
                      use_container_width=True)
        st.caption("Beide reeksen als index (100 = eerste gezamenlijke maand) om samenloop zichtbaar te maken.")
    else:
        st.info("Te weinig overlap tussen CCI-maanden en jouw omzetmaanden om een zinvolle vergelijking te tonen.")

    # ── Detailhandel (optioneel)
    if use_retail and retail_series:
        last_r = retail_series[-1]
        st.metric(f"Detailhandel ({last_r['branch']}) — {last_r['series']}", f"{last_r['retail_value']:.1f}")
    elif use_retail:
        st.info("Geen detailhandelreeks voor deze branche/periode (85828NED).")
