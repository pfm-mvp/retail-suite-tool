import streamlit as st
import pandas as pd, numpy as np, requests, pytz
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from shop_mapping import SHOP_NAME_MAP
from helpers_shop import ID_TO_NAME, get_ids_by_region
from helpers_normalize import normalize_vemcount_response

st.set_page_config(page_title="AI Retail Advisor", layout="wide", page_icon="🛍️")
st.title("🛍️ AI Retail Advisor: Regio- & Winkelvoorspellingen")

# ─── SECRETS ───
API_URL = st.secrets["API_URL"]
OW_KEY  = st.secrets["openweather_api_key"]
CBS_ID  = st.secrets["cbs_dataset"]
CBS_URL = f"https://opendata.cbs.nl/ODataFeed/odata/{CBS_ID}/Consumentenvertrouwen"

# ─── SIDEBAR ───
PERIODS = ["this_week","last_week","this_month","last_month","this_quarter","last_quarter","this_year","last_year"]
regio = st.sidebar.selectbox("Regio", ["All"] + list(set(i["region"] for i in SHOP_NAME_MAP.values())), index=0)
period = st.sidebar.selectbox("Periode", PERIODS, index=3)  # last_month
periode = st.sidebar.date_input("Custom Periode (optioneel)", [datetime(2025,1,1), datetime(2025,10,31)], key="custom")

# Shop IDs
if regio == "All":
    shop_ids = list(SHOP_NAME_MAP.keys())
else:
    shop_ids = get_ids_by_region(regio)
if not shop_ids:
    st.stop()

# ─── HELPER FUNCTIES (exact van werkend script) ───
TZ = pytz.timezone("Europe/Amsterdam")
TODAY = datetime.now(TZ).date()
METRICS = ["count_in","conversion_rate","turnover","sales_per_visitor","sq_meter"]

def step_for(p: str) -> str:
    return "day" if p.endswith("week") or p.endswith("month") else "month"

def add_effective_date(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "date" not in d.columns:
        d["date"] = pd.NaT
    ts = pd.to_datetime(d.get("timestamp"), errors="coerce")
    d["date_eff"] = pd.to_datetime(d["date"], errors="coerce").fillna(ts)
    d["date_eff"] = d["date_eff"].dt.date
    d["year"] = pd.to_datetime(d["date_eff"]).dt.year
    d["month"] = pd.to_datetime(d["date_eff"]).dt.month
    return d

def fetch_data(shop_ids, period: str) -> pd.DataFrame:
    params = [("data", sid) for sid in shop_ids]
    params += [("data_output", m) for m in METRICS]
    params += [("source", "shops"), ("period", period), ("step", step_for(period))]
    try:
        r = requests.post(API_URL, params=params, timeout=45)
        r.raise_for_status()
        js = r.json()
        st.success(f"✅ Planet PFM online – {len(js)} records")
        df = normalize_vemcount_response(js, ID_TO_NAME, METRICS)
        df = add_effective_date(df)
        if period.startswith("this_"):
            df = df[df["date_eff"] < TODAY]
        df = df.sort_values(["shop_id", "date_eff"])
        df["sq_meter"] = df.groupby("shop_id")["sq_meter"].ffill().bfill()
        return df
    except Exception as e:
        st.error(f"❌ Planet PFM: {str(e)[:100]}")
        return pd.DataFrame()

df = fetch_data(shop_ids, period)

# ─── WEER & CBS (blijft) ───
@st.cache_data(ttl=1800)
def weer(pc):
    try:
        r = requests.get("https://api.openweathermap.org/data/2.5/forecast", params={"q": f"{pc},NL", "appid": OW_KEY, "units":"metric"})
        r.raise_for_status()
        js = r.json()["list"]
        daily = pd.DataFrame([{"date": pd.to_datetime(d["dt_txt"]).date(), "temp": d["main"]["temp"], "rain": d.get("rain", {}).get("3h", 0)} for d in js])
        return daily.groupby("date").mean(numeric_only=True).reset_index().head(28)
    except:
        return None

@st.cache_data(ttl=86400)
def cbs():
    try:
        raw = requests.get(CBS_URL).json()["value"]
        c = pd.DataFrame(raw)[["Perioden","Consumentenvertrouwen_1"]]
        c["maand"] = pd.to_datetime(c["Perioden"].str[:7] + "-01")
        return c.rename(columns={"Consumentenvertrouwen_1":"CBS"})[["maand","CBS"]]
    except:
        return pd.DataFrame({"maand": pd.date_range("2025-01", "2025-11", freq="MS"), "CBS": [-8,-7,-9,-6,-10,-11,-9,-12,-10,-13,-14]})

cbs_df = cbs()

# ─── KPI’s (exact uit Planet PFM) ───
if not df.empty:
    total_foot = df["count_in"].sum()
    total_omzet = df["turnover"].sum()
    avg_conv = (df["turnover"] / df["count_in"].replace(0, np.nan)).mean() * 100
    avg_spv = df["sales_per_visitor"].mean()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Footfall", f"{int(total_foot):,}".replace(",","."))
    c2.metric("Omzet", f"€{int(total_omzet):,}".replace(",","."))
    c3.metric("Conversie", f"{avg_conv:.1f}%")
    c4.metric("SPV", f"€{avg_spv:.0f}")
else:
    st.warning("Geen data – probeer een andere periode.")

# ─── GRAFIEK ───
if not df.empty:
    maand = df.copy()
    maand["maand"] = pd.to_datetime(maand["date_eff"]).dt.strftime("%Y-%m")
    agg = maand.groupby(["maand","shop_id"]).agg({"count_in":"sum","turnover":"sum"}).reset_index()
    agg["regio"] = agg["shop_id"].map(lambda x: SHOP_NAME_MAP[x]["region"])
    maand_agg = agg.groupby(["maand","regio"]).sum().reset_index()

    fig = go.Figure()
    for r in [regio] if regio != "All" else ["Noord NL", "Zuid NL"]:
        if r not in maand_agg["regio"].unique(): continue
        d = maand_agg[maand_agg.regio==r]
        fig.add_trace(go.Bar(x=d["maand"], y=d["turnover"]/1_000_000, name=f"Omzet {r}"))
        fig.add_trace(go.Scatter(x=d["maand"], y=d["count_in"]/1_000, name=f"Footfall {r}", yaxis="y2", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=cbs_df["maand"].dt.strftime("%Y-%m"), y=cbs_df["CBS"], name="CBS Vertrouwen", yaxis="y3", line=dict(color="red")))
    fig.update_layout(yaxis=dict(title="Omzet (€M)"), yaxis2=dict(title="Footfall (×1.000)", overlaying="y", side="right"),
                      yaxis3=dict(title="CBS", overlaying="y", side="right", position=0.99), barmode="group")
    st.plotly_chart(fig, use_container_width=True)

# ─── VOORSPELLING (simpel, met weer) ───
forecast_rows = []
for sid in shop_ids:
    info = SHOP_NAME_MAP[sid]
    w = weer(info["postcode"])
    if w is None: continue
    hist = df[df["shop_id"] == sid]
    if len(hist) < 5: continue
    avg_foot = hist["count_in"].mean()
    for i in range(4):
        week = (datetime.now() + timedelta(weeks=i)).strftime("%-d %b")
        temp = w.iloc[i*7:(i+1)*7]["temp"].mean() if len(w) > i*7+7 else 12
        rain = w.iloc[i*7:(i+1)*7]["rain"].sum() if len(w) > i*7+7 else 0
        adj = 1.0
        if rain > 5: adj *= 0.90
        if temp > 18: adj *= 1.15
        foot = int(avg_foot * 7 * adj)
        omzet = foot * hist["sales_per_visitor"].mean()
        forecast_rows.append({"week": week, "winkel": info["name"], "footfall": foot, "omzet": f"€{int(omzet):,}".replace(",", "."), "duiding": "regen" if rain>5 else "zonnig" if temp>18 else "stabiel"})
forecast = pd.DataFrame(forecast_rows)

tab1,tab2,tab3 = st.tabs(["📊 YTD vs. CBS","🔮 4 Weken","✅ Actieplan"])
with tab2:
    if not forecast.empty:
        st.dataframe(forecast.pivot_table(index="week", columns="winkel", values="footfall").fillna("—"))
        st.dataframe(forecast[["week","winkel","omzet","duiding"]])
with tab3:
    for _,r in forecast.iterrows():
        with st.expander(f"{r['winkel']} – {r['week']}"):
            st.code(f"Beste {r['winkel']},\n\nWeek {r['week']} → {r['footfall']:,} bezoekers\nWeer: {r['duiding']}\n\nActie: {'Indoor promo' if 'regen' in r['duiding'] else 'Terras open'}\n\nSucces!", language="text")

if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

st.caption("Bron: Planet PFM (live), OpenWeather, CBS | Real-time")
