# pages/03_AI_Advisor_Weather_CBS.py
import os
import sys
import json
import pandas as pd
import streamlit as st

# ───────── Mapping & API-wrapper (zoals in je benchmark-tool) ─────────
try:
    from shop_mapping import SHOP_NAME_MAP  # {shop_id: "Store Name" of JSON-string met meta}
except Exception:
    SHOP_NAME_MAP = None

from utils_pfmx import api_get_report, friendly_error

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
from services.advisor import build_advice

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

# ───────── Page setup ─────────
st.set_page_config(page_title="AI Advisor — Weer + CBS", page_icon="🧭", layout="wide")
st.title("🧭 AI Advisor — Weer + CBS")

# ───────── Kleine helpers ─────────
def _eur(x):
    try: return "€{:,.0f}".format(float(x)).replace(",", ".")
    except: return "€0"

def _fmt_pct(x):
    return "—" if (x is None or pd.isna(x)) else f"{float(x):+.1f}%"

def _iso_year_week(dt: pd.Timestamp) -> tuple[int,int]:
    iso = dt.isocalendar()
    return int(iso.year), int(iso.week)

# Gebruik een regio-centroid voor weer (fallback = Amsterdam)
REGION_COORDS = {
    "Noord NL": (53.2, 6.56),
    "Midden NL": (52.1, 5.12),
    "Zuid NL": (51.6, 5.06),
    "Randstad": (52.28, 4.85),
}
DEFAULT_COORDS = (52.37, 4.90)

# ───────── UI Controls ─────────
region = st.selectbox("Regio", options=["ALL"] + list(REGIONS), index=0)
days_ahead = st.slider("Dagen vooruit (weer-forecast)", 7, 30, 14)

st.subheader("Macro-context (CBS)")
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    period_hist = st.selectbox("Historische periode (omzet/traffic)", ["last_month", "this_year", "last_year"], index=0)
with c2:
    use_retail = st.checkbox("Toon detailhandel-index (85828NED)", value=True)
with c3:
    dim_name, branch_items = list_retail_branches("85828NED")
    # default = NONFOOD
    if branch_items:
        title_to_key = {b["title"]: str(b["key"]) for b in branch_items}
        titles = list(title_to_key.keys())
        default_idx = next((i for i, t in enumerate(titles) if "nonfood" in t.lower()), 0)
        branch_title = st.selectbox("Branche (CBS)", titles, index=default_idx)
        branch_key = title_to_key[branch_title]
    else:
        branch_title = st.selectbox("Branche (CBS)", ["DH_NONFOOD","DH_TOTAAL","DH_FOOD"], index=0)
        branch_key = branch_title

# ───────── Macro reeksen (buiten de knop) ─────────
try:
    cci_series = get_cci_series(months_back=24, dataset=CBS_DATASET)
except Exception as e:
    cci_series = []
    st.info(f"CCI niet beschikbaar: {e}")

try:
    retail_series = []
    if use_retail:
        retail_series = get_retail_index(branch_code_or_title=branch_key, months_back=24) or \
                        get_retail_index(branch_code_or_title=branch_title, months_back=24) or \
                        get_retail_index(branch_code_or_title="DH_NONFOOD", months_back=24)
except Exception as e:
    retail_series = []
    if use_retail:
        st.info(f"Detailhandelreeks (85828NED) niet beschikbaar: {e}")

# ───────── Datahulpjes ─────────
def add_effective_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    dt_date = pd.to_datetime(d.get("date"), errors="coerce")
    dt_ts   = pd.to_datetime(d.get("timestamp"), errors="coerce")
    d["date_eff"]  = dt_date.fillna(dt_ts)
    d["weekday"]   = d["date_eff"].dt.weekday
    d["ym"]        = d["date_eff"].dt.to_period("M").astype(str)
    d["date_only"] = d["date_eff"].dt.date
    # ISO week (robust tegen NA)
    iso = d["date_eff"].dt.isocalendar()
    d["iso_week"] = iso.week
    d["iso_year"] = iso.year
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
    """Per weekday per store: gemiddelde visitors/conv/SPV en een paar kwantielen."""
    if df is None or df.empty:
        return {}
    d = add_effective_date_cols(df)
    out = {}
    for wd, g in d.groupby("weekday"):
        stores = {}
        for sid, gs in g.groupby("shop_id"):
            name = ID_TO_NAME.get(int(sid), f"Shop {sid}")
            if "sales_per_visitor" in gs.columns:
                spv_series = gs["sales_per_visitor"]
            else:
                denom = gs["count_in"].replace(0, pd.NA)
                spv_series = (gs["turnover"] / denom).fillna(0)
            stores[name] = {
                "visitors":     float(gs["count_in"].mean()),
                "conversion":   float(gs["conversion_rate"].mean()),  # in %
                "spv":          float(spv_series.mean()),
                "spv_median":   float(spv_series.median()),
                "visitors_p30": float(gs["count_in"].quantile(0.30)),
            }
        out[wd] = stores
    return out

def monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
    d = add_effective_date_cols(df)
    g = d.groupby("ym", as_index=False).agg(
        visitors=("count_in","sum"),
        turnover=("turnover","sum"),
    )
    d["weighted_conv"] = d["conversion_rate"] * d["count_in"]
    conv = d.groupby("ym", as_index=False).agg(conv_num=("weighted_conv","sum"),
                                              conv_den=("count_in","sum"))
    conv["conversion"] = (conv["conv_num"]/conv["conv_den"]).fillna(0)
    out = g.merge(conv[["ym","conversion"]], on="ym", how="left")
    out["spv"] = (out["turnover"]/out["visitors"]).replace([float("inf")], 0).fillna(0)
    return out

def mom_yoy(dfm: pd.DataFrame):
    if dfm is None or dfm.empty:
        return {}
    m = dfm.copy().sort_values("ym").reset_index(drop=True)
    m["ym_dt"] = pd.to_datetime(m["ym"].astype(str) + "-01", errors="coerce")
    m = m[m["ym_dt"].notna()]
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
        "this_label":  last["ym_dt"].strftime("%Y-%m"),
        "prev_label":  prev["ym_dt"].strftime("%Y-%m") if prev is not None else "n.v.t.",
        "visitors":    float(last.get("visitors", 0)),
        "turnover":    float(last.get("turnover", 0)),
        "conversion":  float(last.get("conversion", 0)),  # al in %
        "spv":         float(last.get("spv", 0)),
        "mom": {
            "turnover":   pct(last.get("turnover", 0),   prev.get("turnover", 0))   if prev is not None else None,
            "visitors":   pct(last.get("visitors", 0),   prev.get("visitors", 0))   if prev is not None else None,
            "conversion": pct(last.get("conversion", 0), prev.get("conversion", 0)) if prev is not None else None,
            "spv":        pct(last.get("spv", 0),        prev.get("spv", 0))        if prev is not None else None,
        },
        "yoy": {
            "turnover":   pct(last.get("turnover", 0),   yoy_row.get("turnover", 0))   if yoy_row is not None else None,
            "visitors":   pct(last.get("visitors", 0),   yoy_row.get("visitors", 0))   if yoy_row is not None else None,
            "conversion": pct(last.get("conversion", 0), yoy_row.get("conversion", 0)) if yoy_row is not None else None,
            "spv":        pct(last.get("spv", 0),        yoy_row.get("spv", 0))        if yoy_row is not None else None,
        },
    }

def estimate_weather_uplift(baseline_day: dict, forecast_days: list[dict]) -> dict:
    """
    Baseline bevat 'visitors', 'spv' en 'conv' (in %).
    Omzet = visitors * spv * (conv/100).
    """
    daily = []
    for f in forecast_days:
        wd = pd.to_datetime(f["date"]).weekday()
        base = baseline_day.get(wd, {"visitors": 0.0, "spv": 0.0, "conv": 0.0})
        base_vis = float(base["visitors"]); spv = float(base["spv"]); conv = float(base.get("conv",0.0))/100.0
        pop  = float(f.get("pop", 0.0))
        temp = float(f.get("temp", 15.0))
        adj_vis = base_vis * (1 - 0.20*pop) * (1 + 0.01*(temp-15.0))
        row = {
            "date": f["date"], "weekday": wd, "temp": temp, "pop": pop,
            "base_visitors": base_vis, "adj_visitors": adj_vis, "spv": spv, "conv": conv*100,
            "base_turnover": base_vis * spv * conv, "adj_turnover": adj_vis * spv * conv
        }
        row["delta_turnover"] = row["adj_turnover"] - row["base_turnover"]
        daily.append(row)
    base_total = sum(x["base_turnover"] for x in daily)
    adj_total  = sum(x["adj_turnover"]  for x in daily)
    delta_total = adj_total - base_total
    delta_pct = (delta_total/base_total*100) if base_total else None
    return {"daily": daily, "base_total": base_total, "adj_total": adj_total,
            "delta_total": delta_total, "delta_pct": delta_pct}

def summarize_weeks(wx_daily_rows: list[dict]) -> tuple[pd.DataFrame, str]:
    """Maak weektotalen + korte NL-verklaring van drivers (regen/temp)."""
    if not wx_daily_rows:
        return pd.DataFrame(), "Geen forecast beschikbaar."
    d = pd.DataFrame(wx_daily_rows).copy()
    d["date"] = pd.to_datetime(d["date"])
    d["iso_year"] = d["date"].dt.isocalendar().year.astype(int)
    d["iso_week"] = d["date"].dt.isocalendar().week.astype(int)
    g = (d.groupby(["iso_year","iso_week"], as_index=False)
           .agg(
               expected_turnover=("adj_turnover","sum"),
               baseline_turnover=("base_turnover","sum"),
               expected_visitors=("adj_visitors","sum"),
               pop_mean=("pop","mean"),
               temp_mean=("temp","mean"),
           ))
    g["delta_vs_base"] = g["expected_turnover"] - g["baseline_turnover"]
    g = g.sort_values(["iso_year","iso_week"]).reset_index(drop=True)
    expl = []
    for _, r in g.iterrows():
        driver = []
        if r["pop_mean"] >= 0.6:
            driver.append("veel regen")
        elif r["pop_mean"] <= 0.2:
            driver.append("droog")
        if r["temp_mean"] >= 18:
            driver.append("relatief warm")
        elif r["temp_mean"] <= 10:
            driver.append("koud")
        tone = "boven normaal" if r["delta_vs_base"] >= 0 else "onder normaal"
        wk = f"week {int(r['iso_week'])}"
        dr = (", ".join(driver) or "normaal weer")
        expl.append(f"{wk}: {tone} door {dr}")
    return g, ("; ".join(expl))

def cci_last_and_trend(cci_series_in: list[dict]) -> tuple[float|None, float|None, float]:
    """(laatste_cci, MoM delta, trend_slope laatste 3m)."""
    if not cci_series_in:
        return None, None, 0.0
    c = pd.DataFrame(cci_series_in)
    c["period_dt"] = pd.to_datetime(
        c["period"].astype(str).str.replace("MM","-").str.replace("M","-") + "-01",
        errors="coerce"
    )
    c = c[c["period_dt"].notna()].sort_values("period_dt")
    if c.empty:
        return None, None, 0.0
    last = float(c.iloc[-1]["cci"])
    prev = float(c.iloc[-2]["cci"]) if len(c)>1 else None
    mom = (last - prev) if prev is not None else None
    tail = c.tail(3)
    if len(tail) >= 2:
        slope = (tail["cci"].iloc[-1] - tail["cci"].iloc[0]) / (len(tail)-1)
    else:
        slope = 0.0
    return last, mom, float(slope)

def cci_line_this_year(cci_series_in: list[dict]) -> pd.DataFrame:
    if not cci_series_in: return pd.DataFrame()
    c = pd.DataFrame(cci_series_in)
    c["period_dt"] = pd.to_datetime(
        c["period"].astype(str).str.replace("MM","-").str.replace("M","-") + "-01",
        errors="coerce"
    )
    c = c[c["period_dt"].dt.year == pd.Timestamp.today().year]
    return c[["period_dt","cci"]].sort_values("period_dt")

def merge_cci_vs_turnover_this_year(dfm_in: pd.DataFrame, cci_series_in: list[dict]) -> pd.DataFrame:
    if dfm_in is None or dfm_in.empty or not cci_series_in:
        return pd.DataFrame()
    dm = dfm_in.copy()
    dm["ym_dt"] = pd.to_datetime(dm["ym"].astype(str) + "-01", errors="coerce")
    dm = dm[dm["ym_dt"].dt.year == pd.Timestamp.today().year]
    dm = dm.groupby("ym_dt", as_index=False).agg(turnover=("turnover","sum"))
    if dm.empty:
        return pd.DataFrame()
    c = pd.DataFrame(cci_series_in)
    c["period_dt"] = pd.to_datetime(
        c["period"].astype(str).str.replace("MM","-").str.replace("M","-") + "-01",
        errors="coerce"
    )
    c = c[c["period_dt"].dt.year == pd.Timestamp.today().year][["period_dt","cci"]]
    if c.empty:
        return pd.DataFrame()
    merged = dm.merge(c, left_on="ym_dt", right_on="period_dt", how="inner")
    if merged.empty:
        return pd.DataFrame()
    merged = merged.sort_values("ym_dt").reset_index(drop=True)
    base_t = float(merged["turnover"].iloc[0]) or 1.0
    base_c = float(merged["cci"].iloc[0]) or 1.0
    merged["Omzet_idx"] = merged["turnover"] / base_t * 100.0
    merged["CCI_idx"]   = merged["cci"]      / base_c * 100.0
    return merged[["ym_dt","Omzet_idx","CCI_idx"]]

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

    # Debug-tabel met nette kolommen
    with st.expander("🛠️ Debug — eerste rijen (hist KPI’s)"):
        dshow = df.copy()
        def parse_shop_meta(v):
            try:
                obj = json.loads(v) if isinstance(v, str) else v
                if isinstance(obj, dict):
                    return obj.get("name"), obj.get("postcode"), obj.get("region")
            except Exception:
                pass
            return (v, None, None)
        name, pc, rg = [], [], []
        for v in dshow.get("shop_name", []):
            n, p, r = parse_shop_meta(v)
            name.append(n); pc.append(p); rg.append(r)
        if name:
            dshow["name"] = name; dshow["postcode"] = pc; dshow["region"] = rg
            if "shop_name" in dshow.columns: dshow.drop(columns=["shop_name"], inplace=True)
        # prefer 'timestamp' als 'date'
        if "timestamp" in dshow.columns:
            dshow.rename(columns={"timestamp":"date"}, inplace=True)
        if "date" in dshow.columns and dshow["date"].notna().any():
            dshow["date"] = pd.to_datetime(dshow["date"], errors="coerce").dt.date
        st.dataframe(dshow.head(15), use_container_width=True)

    if df is None or df.empty:
        st.warning("Geen historische KPI-data voor deze selectie/periode. Probeer ‘this_year’ of ‘last_year’.")
        st.stop()

    # 2) Baselines + regiotrend
    df = add_effective_date_cols(df)
    baseline = build_weekday_baselines(df)
    dfm = monthly_agg(df)
    trend = mom_yoy(dfm)

    # 3) Weer + CBS
    # regio-coordinaat:
    lat, lon = REGION_COORDS.get(region, DEFAULT_COORDS)
    forecast = get_daily_forecast(lat, lon, OPENWEATHER_KEY, days_ahead)
    try:
        cci_info = get_consumer_confidence(CBS_DATASET)
        _cci_val = cci_info.get("value", None)
        _cci_period = cci_info.get("period", "")
    except Exception:
        _cci_val, _cci_period = None, ""

    # 4) AI-adviesregels (dag/winkel)
    advice = build_advice("Your Company", baseline, forecast, (_cci_val or 0.0))

    # 5) Regionale baseline incl. conversie
    baseline_day = {}
    for wd, storemap in baseline.items():
        if not storemap:
            continue
        visitors = pd.Series([v["visitors"] for v in storemap.values()]).mean()
        spv      = pd.Series([v["spv"]      for v in storemap.values()]).mean()
        conv     = pd.Series([v["conversion"] for v in storemap.values()]).mean()  # in %
        baseline_day[wd] = {"visitors": float(visitors), "spv": float(spv), "conv": float(conv)}
    if not baseline_day:
        st.warning("Geen baseline beschikbaar (te weinig dagen). Kies een ruimere periode.")
        st.stop()

    # 6) Forecast met weer-correctie → weken + verklaring
    wx = estimate_weather_uplift(baseline_day, forecast)
    wk_df, wk_expl = summarize_weeks(wx["daily"])

    # ── Silver-platter regio
    st.subheader("🔎 Silver-platter samenvatting (regio)")
    if trend:
        colA, colB, colC, colD = st.columns(4)
        colA.metric(f"Omzet {trend.get('this_label','-')}",
                    _eur(trend.get('turnover', 0)),
                    _fmt_pct(trend['mom'].get('turnover')))
        colB.metric(f"Bezoekers {trend.get('this_label','-')}",
                    f"{trend.get('visitors', 0):,.0f}".replace(",", "."),
                    _fmt_pct(trend['mom'].get('visitors')))
        # NIET *100 doen: waarde is al '%'
        colC.metric(f"Conversie {trend.get('this_label','-')}",
                    f"{trend.get('conversion', 0):.2f}%",
                    _fmt_pct(trend['mom'].get('conversion')))
        colD.metric(f"SPV {trend.get('this_label','-')}",
                    f"€{trend.get('spv', 0):.2f}",
                    _fmt_pct(trend['mom'].get('spv')))
        st.caption(
            f"Vergelijken met: {trend.get('prev_label','n.v.t.')}.  "
            + (f"YoY omzet: {trend['yoy']['turnover']:.1f}%  " if trend['yoy']['turnover'] is not None else "")
            + (f"• bezoekers: {trend['yoy']['visitors']:.1f}%  " if trend['yoy']['visitors'] is not None else "")
            + (f"• conversie: {trend['yoy']['conversion']:.1f}%  " if trend['yoy']['conversion'] is not None else "")
            + (f"• SPV: {trend['yoy']['spv']:.1f}%" if trend['yoy']['spv'] is not None else "")
        )

    # ── Verwachting per week & maand
    st.subheader("🗺️ Verwachting per week & maand (regio)")
    if not wk_df.empty:
        rows = []
        for _, r in wk_df.iterrows():
            rows.append(f"• Week {int(r['iso_week'])}: verwacht {_eur(r['expected_turnover'])} omzet en {int(round(r['expected_visitors']))} bezoekers")
        st.markdown("\n".join(rows))
        # CCI richting (op basis van 24m-reeks)
        last_cci, cci_mom, cci_trend_slope = cci_last_and_trend(cci_series)
        direction_down = (wk_df["expected_turnover"].diff().iloc[-1] if len(wk_df)>1 else 0) < 0
        box = st.error if direction_down else st.success
        box(
            f"Conclusie: {'neerwaarts' if direction_down else 'opwaarts'}. "
            f"Voor de komende {len(wx['daily'])} dagen verwachten we in totaal {_eur(wx['adj_total'])} omzet. "
            f"Weercomponent: {wk_expl}. CCI-effect: "
            + ("↗ (lichte SPV-correctie)." if (cci_trend_slope or 0) > 0 else "↘ (lichte SPV-correctie).")
        )
    else:
        st.info("Nog onvoldoende dagen in forecast om weken te tonen.")

    # ── Macro duiding (CCI)
    last_cci, cci_mom, cci_trend_slope = cci_last_and_trend(cci_series)
    st.metric(
        "Consumentenvertrouwen (CBS)",
        f"{last_cci:.1f}" if last_cci is not None else "n.v.t.",
        (f"{cci_mom:+.1f} m/m" if cci_mom is not None else "—"),
        help="CCI > 0 = positiever sentiment; < 0 = voorzichtiger. Delta is t.o.v. vorige maand."
    )
    cci_this_year = cci_line_this_year(cci_series)
    with st.expander("📈 CCI dit jaar (maandelijks)"):
        if cci_this_year.empty:
            st.info("Nog geen CCI-waarnemingen voor dit jaar.")
        else:
            st.line_chart(cci_this_year.set_index("period_dt")["cci"])

    # ── Komende dagen — (optioneel) acties per winkel / dag opvragen
    with st.expander("📅 Komende dagen — acties & weer"):
        for d in advice["days"]:
            st.write(f"• {d['date']}: {d['weather']['temp']:.1f}°C, regen {int(d['weather']['pop']*100)}%")

    # ── CCI vs Omzet — genormaliseerd (dit jaar)
    st.subheader("📊 CCI vs. Omzet — genormaliseerde trend")
    cci_vs_turn = merge_cci_vs_turnover_this_year(dfm, cci_series)
    if cci_vs_turn.empty:
        st.info("Te weinig overlap tussen CCI-maanden en omzetmaanden om een trend te tonen.")
    else:
        st.line_chart(cci_vs_turn.set_index("ym_dt")[["CCI_idx","Omzet_idx"]])
        st.caption("Beide reeksen als index (100 = eerste gezamenlijke maand) om samenloop zichtbaar te maken.")

    # ── (Optioneel) detailhandelreeks
    if retail_series:
        last_r = retail_series[-1]
        st.metric(f"Detailhandel ({last_r['branch']}) — {last_r['series']}", f"{last_r['retail_value']:.1f}")
        with st.expander("🛍️ Detailhandel reeks (CBS 85828NED)"):
            st.line_chart({"Retail": [x["retail_value"] for x in retail_series]})
    elif use_retail:
        st.info("Geen detailhandelreeks gevonden voor deze branche/periode (85828NED).")
