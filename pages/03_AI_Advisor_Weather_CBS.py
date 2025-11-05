# pages/03_AI_Advisor_Weather_CBS.py
import os
import sys
import re
import math
import numpy as np
import pandas as pd
import streamlit as st

# ───────── Mapping & API wrapper (zelfde als benchmark-tool) ─────────
try:
    from shop_mapping import SHOP_NAME_MAP  # {shop_id: "Store Name", ...}
except Exception:
    SHOP_NAME_MAP = None

from utils_pfmx import api_get_report, friendly_error

# ───────── Page setup ─────────
st.set_page_config(page_title="AI Advisor — Weer + CBS", page_icon="🧭", layout="wide")
st.title("🧭 AI Advisor — Weer + CBS (v2)")

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

# ───────── UI Controls ─────────
region = st.selectbox("Regio", options=["ALL"] + list(REGIONS), index=0)
# Lat/Lon verwijderd uit UI; we gebruiken intern een vaste NL-centroid voor weer
NL_LAT, NL_LON = 52.37, 4.90

days_ahead = st.slider("Dagen vooruit", 7, 30, 14)
period_hist = st.selectbox("Historische periode", ["last_month", "this_year", "last_year"], index=0)

st.subheader("Macro-context (CBS)")
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    months_back = st.slider("Maanden terug (CBS)", 6, 36, 18)
with c2:
    use_retail = st.checkbox("Toon detailhandel-index (85828NED)", value=True)
with c3:
    dim_name, branch_items = list_retail_branches("85828NED")
    if branch_items:
        title_to_key = {b["title"]: str(b["key"]) for b in branch_items}
        titles = list(title_to_key.keys())
        # default = NONFOOD waar mogelijk, anders eerste item
        default_idx = next((i for i, t in enumerate(titles) if "nonfood" in t.lower()), 0)
        branch_title = st.selectbox("Branche (CBS)", titles, index=default_idx)
        branch_key = title_to_key[branch_title]
    else:
        branch_title = st.selectbox("Branche (CBS)", ["DH_NONFOOD","DH_TOTAAL","DH_FOOD"], index=0)
        branch_key = branch_title

# ───────── Macro reeksen (buiten de knop) ─────────
def _period_to_ym_str(val: str) -> str:
    """
    Converteer CBS 'Periods' waarden naar 'YYYY-MM'.
    Ondersteunt vormen als '2025MM10', '2025M10', '2025-10', '2025 10'.
    """
    if not isinstance(val, str):
        val = str(val)
    m = re.findall(r"(\d{4}).*?(\d{1,2})", val)
    if m:
        y, mo = m[0]
        return f"{int(y):04d}-{int(mo):02d}"
    return None

try:
    cci_series = get_cci_series(months_back=months_back, dataset=CBS_DATASET)
    # cci_series → [{'period': '2025MM10'|..., 'cci': float}]
    cci_df = pd.DataFrame(cci_series)
    if not cci_df.empty:
        cci_df["ym"] = cci_df["period"].apply(_period_to_ym_str)
        cci_df = cci_df.dropna(subset=["ym"])
    else:
        cci_df = pd.DataFrame(columns=["ym","cci"])
except Exception as e:
    cci_df = pd.DataFrame(columns=["ym","cci"])
    st.info(f"CCI niet beschikbaar: {e}")

try:
    retail_series = []
    if use_retail:
        retail_series = (
            get_retail_index(branch_code_or_title=branch_key,   months_back=months_back) or
            get_retail_index(branch_code_or_title=branch_title, months_back=months_back) or
            get_retail_index(branch_code_or_title="DH_NONFOOD", months_back=months_back) or
            get_retail_index(branch_code_or_title="DH_TOTAAL",  months_back=months_back)
        )
except Exception as e:
    retail_series = []
    if use_retail:
        st.info(f"Detailhandelreeks (85828NED) niet beschikbaar: {e}")

# ───────── Helpers: datum, aggregaties, baseline, forecast ─────────
def add_effective_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    dt_date = pd.to_datetime(d.get("date"), errors="coerce")
    dt_ts   = pd.to_datetime(d.get("timestamp"), errors="coerce")
    d["date_eff"]  = dt_date.fillna(dt_ts)
    d["weekday"]   = d["date_eff"].dt.weekday
    d["ym"]        = d["date_eff"].dt.to_period("M").astype(str)
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
            name = ID_TO_NAME.get(int(sid), f"Shop {sid}")
            if "sales_per_visitor" in gs.columns:
                spv_series = gs["sales_per_visitor"]
            else:
                denom = gs["count_in"].replace(0, pd.NA)
                spv_series = (gs["turnover"] / denom).fillna(0)
            stores[name] = {
                "visitors":     float(gs["count_in"].mean()),
                "conversion":   float(gs["conversion_rate"].mean()),
                "spv":          float(spv_series.mean()),
                "spv_median":   float(spv_series.median()),
                "visitors_p30": float(gs["count_in"].quantile(0.30)),
            }
        out[wd] = stores
    return out

def monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
    d = add_effective_date_cols(df)
    g = d.groupby("ym", as_index=False).agg(
        visitors=("count_in", "sum"),
        turnover=("turnover", "sum"),
    )
    d["weighted_conv"] = d["conversion_rate"] * d["count_in"]
    conv = d.groupby("ym", as_index=False).agg(
        conv_num=("weighted_conv", "sum"),
        conv_den=("count_in", "sum")
    )
    conv["conversion"] = (conv["conv_num"] / conv["conv_den"]).fillna(0)
    out = g.merge(conv[["ym", "conversion"]], on="ym", how="left")
    out["spv"] = (out["turnover"] / out["visitors"]).replace([float("inf")], 0).fillna(0)
    return out

def mom_yoy(dfm: pd.DataFrame):
    if dfm is None or dfm.empty:
        return {}
    m = dfm.copy()
    m["ym_dt"] = pd.to_datetime(m["ym"].astype(str) + "-01", format="%Y-%m-%d", errors="coerce")
    m = m.dropna(subset=["ym_dt"]).sort_values("ym_dt").reset_index(drop=True)
    if m.empty:
        return {}

    last = m.iloc[-1]
    prev = m.iloc[-2] if len(m) > 1 else None
    yoy_dt = last["ym_dt"] - pd.DateOffset(years=1)
    yoy_row = m.loc[m["ym_dt"] == yoy_dt]
    yoy_row = yoy_row.iloc[0] if not yoy_row.empty else None

    def pct(a, b):
        if b in [0, None] or pd.isna(b):
            return None
        try:
            return (float(a) / float(b) - 1) * 100
        except Exception:
            return None

    return {
        "last_ym":   last["ym_dt"].strftime("%Y-%m"),
        "visitors":  float(last.get("visitors", 0)),
        "turnover":  float(last.get("turnover", 0)),
        "conversion":float(last.get("conversion", 0)),
        "spv":       float(last.get("spv", 0)),
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

# ───────── CCI-impact (correlaties en beta’s) ─────────
def compute_cci_impacts(dfm: pd.DataFrame, cci_df: pd.DataFrame):
    """
    Retourneert dict met correlaties en beta's (gevoeligheid) voor visitors/turnover/spv.
    Beta benaderen we met: corr * (std(kpi_pct_change) / std(cci_change))
    """
    if dfm is None or dfm.empty or cci_df is None or cci_df.empty:
        return {"corr": {}, "beta": {}, "last_cci": None, "delta_from_12m_avg": None}

    # Join op ym
    m = dfm.copy()
    m["ym"] = m["ym"].astype(str)
    j = m.merge(cci_df[["ym", "cci"]], on="ym", how="inner").copy()
    if j.empty:
        return {"corr": {}, "beta": {}, "last_cci": None, "delta_from_12m_avg": None}

    # pct changes per maand
    def pct_change(s):
        s = pd.to_numeric(s, errors="coerce")
        return s.pct_change()

    j = j.sort_values("ym")
    kpis = {
        "visitors": j["visitors"],
        "turnover": j["turnover"],
        "spv": j["spv"],
    }
    cci = j["cci"]

    cci_chg = pct_change(cci)
    corr = {}
    beta = {}
    for k, series in kpis.items():
        k_chg = pct_change(series)
        # corr
        c = k_chg.corr(cci_chg)
        corr[k] = float(c) if pd.notna(c) else 0.0
        # beta
        s_k = np.nanstd(k_chg.values)
        s_c = np.nanstd(cci_chg.values)
        b = (c * (s_k / s_c)) if (s_c and not math.isclose(s_c, 0.0)) else 0.0
        beta[k] = float(b if pd.notna(b) else 0.0)

    # delta van laatste CCI t.o.v. 12m gemiddelde (meer stabiel signaal)
    last_cci = cci.iloc[-1] if not cci.empty else None
    cci_12m_avg = cci.iloc[-12:].mean() if len(cci) >= 3 else cci.mean()
    d12 = (last_cci - cci_12m_avg) if (last_cci is not None and pd.notna(last_cci)) else None

    return {"corr": corr, "beta": beta, "last_cci": last_cci, "delta_from_12m_avg": d12}

# ───────── Weer/CCI-forecast ─────────
def estimate_weather_and_cci(baseline_day: dict, forecast_days: list[dict], days_ahead: int, cci_effect: dict):
    """
    Combineer weer (t/m 7 dagen) + CCI-trend (tot 30 dagen).
    - Dagen 1..len(forecast_days): pas weerfactor + CCI toe.
    - Dagen >len(forecast_days)..days_ahead: pas alleen CCI toe (weer neutraal).
    """
    daily = []
    beta = cci_effect.get("beta", {})
    d12 = cci_effect.get("delta_from_12m_avg", 0.0) or 0.0  # CCI delta tov 12m gemiddelde

    # CCI-impuls: omzet ~ beta_turnover * dCCI, visitors ~ beta_visitors * dCCI, spv ~ beta_spv * dCCI
    for i in range(days_ahead):
        if i < len(forecast_days):
            f = forecast_days[i]
            pop  = float(f.get("pop", 0.0))
            temp = float(f.get("temp", 15.0))
            date_str = f.get("date")
        else:
            # neutraal weer, doorrollen kalenderdagen
            pop, temp = 0.3, 15.0
            if daily:
                prev_date = pd.to_datetime(daily[-1]["date"])
                date_str = (prev_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                # als er nog niets is, start vanaf vandaag
                date_str = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

        wd = pd.to_datetime(date_str).weekday()
        base = baseline_day.get(wd, {"visitors": 0.0, "spv": 0.0})
        base_vis = float(base["visitors"]); base_spv = float(base["spv"])

        # Weerfactor (alleen op 1..7)
        vis_adj_weather = base_vis * (1 - 0.20*pop) * (1 + 0.01*(temp-15.0)) if i < len(forecast_days) else base_vis

        # CCI-factor via beta’s
        vis_cc = vis_adj_weather * (1 + (beta.get("visitors", 0.0) * d12))
        spv_cc = base_spv        * (1 + (beta.get("spv", 0.0)      * d12))
        turn_cc = vis_cc * spv_cc

        daily.append({
            "date": date_str,
            "weekday": wd,
            "temp": temp,
            "pop": pop,
            "base_visitors": base_vis,
            "adj_visitors": vis_cc,
            "spv": spv_cc,
            "adj_turnover": turn_cc
        })

    adj_total = sum(x["adj_turnover"] for x in daily)
    return {"daily": daily, "adj_total": adj_total}

# ───────── Formatting helpers ─────────
def _eur(x):
    try: return "€{:,.0f}".format(x).replace(",", ".")
    except: return "–"

def _eur2(x):
    try: return "€{:,.2f}".format(x).replace(",", ".")
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
    with st.expander("🛠️ Debug — eerste rijen (hist KPI’s)"):
        st.write(df.head(15))
    if df is None or df.empty:
        st.warning("Geen historische KPI-data voor deze selectie/periode. Probeer ‘this_year’ of ‘last_year’.")
        st.stop()

    # 2) Baselines + regiotrend (maandelijks)
    df = add_effective_date_cols(df)
    baseline = build_weekday_baselines(df)

    dfm = monthly_agg(df)
    # Guard: drop ongeldige maanden (NaT)
    dfm["_ym_dt_check"] = pd.to_datetime(dfm["ym"].astype(str) + "-01", errors="coerce")
    dfm = dfm.dropna(subset=["_ym_dt_check"]).drop(columns=["_ym_dt_check"])
    trend = mom_yoy(dfm)

    # 3) Weer + CCI
    #   - Weer voor max 7 dagen (OpenWeather)
    forecast7 = get_daily_forecast(NL_LAT, NL_LON, OPENWEATHER_KEY, min(days_ahead, 7))
    #   - CCI laatste waarde + impact
    try:
        cci_info = get_consumer_confidence(CBS_DATASET)
        cci_now = cci_info["value"]; cci_period = cci_info["period"]
    except Exception as e:
        cci_now, cci_period = 0.0, "n/a"
        st.warning(f"Kon actuele CCI niet ophalen (gebruik standaard 0). Details: {e}")

    cci_effect = compute_cci_impacts(dfm, cci_df)

    # 4) AI-adviesregels (dag/winkel) op basis van weer (eerste 7 dagen) + sentiment
    advice = build_advice("Your Company", baseline, forecast7, cci_now)

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

    # 6) 7–30 daagse verwachting met weer (1..7) + CCI (1..N)
    wxcci = estimate_weather_and_cci(baseline_day, forecast7, days_ahead, cci_effect)

    # ── Silver-platter regio
    st.subheader("🔎 Silver-platter samenvatting (regio)")
    if trend:
        colA, colB, colC, colD = st.columns(4)
        colA.metric(f"Omzet {trend.get('last_ym','-')}", _eur(trend.get('turnover', 0)), _fmt_pct(trend['mom'].get('turnover')))
        colB.metric("Bezoekers", f"{trend.get('visitors', 0):,.0f}".replace(",", "."), _fmt_pct(trend['mom'].get('visitors')))
        colC.metric("Conversie", f"{trend.get('conversion', 0)*100:.2f}%", _fmt_pct(trend['mom'].get('conversion')))
        colD.metric("SPV", f"€{trend.get('spv', 0):.2f}", _fmt_pct(trend['mom'].get('spv')))
        st.caption(
            "YoY: "
            + (f"Omzet {trend['yoy']['turnover']:.1f}%  " if trend['yoy']['turnover'] is not None else "")
            + (f"• Bezoekers {trend['yoy']['visitors']:.1f}%  " if trend['yoy']['visitors'] is not None else "")
            + (f"• Conversie {trend['yoy']['conversion']:.1f}%  " if trend['yoy']['conversion'] is not None else "")
            + (f"• SPV {trend['yoy']['spv']:.1f}%" if trend['yoy']['spv'] is not None else "")
        )

    # ── Consumentenvertrouwen → duiding
    last_cci = cci_effect.get("last_cci", cci_now)
    d12 = cci_effect.get("delta_from_12m_avg")
    tone = "positief (kooplustiger)" if (last_cci is not None and last_cci >= 0) else "negatief (voorzichtiger)"
    st.metric("Consumentenvertrouwen (CBS)", f"{last_cci:.1f}" if last_cci is not None else "n.v.t.", help=f"Periode: {cci_period}")
    st.info(
        f"**CCI = {last_cci:.1f if last_cci is not None else 0}** → sentiment {tone}. "
        f"{'↑' if (d12 or 0) > 0 else ('↓' if (d12 or 0) < 0 else '→')} t.o.v. 12m-gem.: "
        f"{(d12 or 0):+.1f} punten. Tool past forecast aan met jouw historische gevoeligheid voor CCI (beta)."
    )

    # ── Komende dagen — verwachting & acties (weer-gedreven; 1..7)
    st.subheader("📅 Komende dagen — verwachting & acties (1–7)")
    for d in advice["days"]:
        with st.expander(f'📆 {d["date"]} — temp {d["weather"]["temp"]:.1f}°C • POP {int(d["weather"]["pop"]*100)}%'):
            for s in d["stores"]:
                st.markdown(f"**{s['store']}**")
                st.write("— Storemanager:", " • ".join(s["store_actions"]))
                st.write("— Regiomanager:", " • ".join(s["regional_actions"]))

    # ── 7–30 dagen: gecombineerde forecast (weer + CCI)
    st.subheader(f"🔮 Verwachting komende {days_ahead} dagen (weer ≤7d + CCI)")
    if wxcci and wxcci["daily"]:
        wx_df = pd.DataFrame(wxcci["daily"]).copy()
        wx_df["date"] = pd.to_datetime(wx_df["date"])
        wx_df = wx_df.sort_values("date")

        # Toon omzet en drivers
        show_cols = ["date","temp","pop","adj_visitors","spv","adj_turnover"]
        nice = wx_df[show_cols].rename(columns={
            "date":"Datum","temp":"Temp (°C)","pop":"Regenkans",
            "adj_visitors":"Verwachte bezoekers","spv":"Verwachte SPV","adj_turnover":"Verwachte omzet"
        })
        nice["Regenkans"] = (nice["Regenkans"]*100).round(0).astype(int).astype(str) + "%"
        nice["Verwachte SPV"] = nice["Verwachte SPV"].map(_eur2)
        nice["Verwachte omzet"] = nice["Verwachte omzet"].map(_eur)

        st.dataframe(nice, use_container_width=True, height=360)
        st.bar_chart(wx_df.set_index("date")["adj_turnover"].rename("Verwachte omzet (dag)"))

        st.success(
            f"Totale verwachte omzet komende {days_ahead} dagen: **{_eur(wxcci['adj_total'])}**. "
            "Dagen 1–7 weersafhankelijk; daarna sentiment-gedreven via CCI-trend.",
            icon="📈"
        )
    else:
        st.info("Geen forecast-gegevens beschikbaar.")

    # ── Macro tiles (context)
    if not cci_df.empty:
        st.metric("CCI (laatste maand)", f"{cci_df.iloc[-1]['cci']:.1f}")
        with st.expander("📈 CCI reeks (CBS)"):
            st.line_chart({"CCI": list(cci_df["cci"])})

    if retail_series:
        last_r = retail_series[-1]
        st.metric(f"Detailhandel ({last_r['branch']}) — {last_r['series']}", f"{last_r['retail_value']:.1f}")
        with st.expander("🛍️ Detailhandel reeks (CBS 85828NED)"):
            st.line_chart({"Retail": [x["retail_value"] for x in retail_series]})
    elif use_retail:
        with st.expander("🛍️ Detailhandel (geen data voor deze selectie) — klik voor opties"):
            st.markdown(
                "- Probeer een andere branche (bijv. **NONFOOD** of **Totaal detailhandel**).\n"
                "- Of zet de toggle **Toon detailhandel-index** uit.\n"
                "- Tip: niet alle branches hebben waarden voor alle maanden."
            )
            try:
                _, avail = list_retail_branches("85828NED")
                if avail:
                    st.write("Beschikbare branches (eerste 25):", [b['title'] for b in avail[:25]])
            except Exception:
                pass
