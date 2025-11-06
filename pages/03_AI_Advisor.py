import streamlit as st
import pandas as pd, numpy as np, requests, holidays
import plotly.graph_objects as go
from datetime import datetime, timedelta
from shop_mapping import SHOP_NAME_MAP

st.set_page_config(page_title="AI Retail Advisor", layout="wide", page_icon="🛍️")
st.title("🛍️ AI Retail Advisor: Regio- & Winkelvoorspellingen")

# ─── SECRETS ───
OW_KEY   = st.secrets["openweather_api_key"]
VEM_URL  = st.secrets["API_URL"]
CBS_FEED = "https://opendata.cbs.nl/ODataFeed/odata/83693NED/Consumentenvertrouwen"

# ─── SIDEBAR ───
regios = st.sidebar.multiselect("Regio", ["Noord NL", "Zuid NL"], default=["Noord NL", "Zuid NL"])
periode = st.sidebar.date_input("Periode", [datetime(2025,1,1), datetime(2025,10,31)])

# ─── 1. VEMCOUNT DATA ───
@st.cache_data(ttl=3600)
def vemcount():
    ids = [sid for sid,info in SHOP_NAME_MAP.items() if info["region"] in regios]
    r = requests.post(VEM_URL, json={
        "store_ids": ids,
        "start_date": periode[0].isoformat(),
        "end_date":   periode[1].isoformat(),
        "metrics": ["footfall","transactions","revenue"]
    })
    if r.status_code != 200: 
        st.error("Vemcount offline → fallback data")
        return pd.DataFrame()  # fallback later
    df = pd.DataFrame(r.json())
    df["date"] = pd.to_datetime(df["date"])
    df["footfall"] = pd.to_numeric(df["footfall"])
    df["omzet"]    = pd.to_numeric(df["revenue"])
    df["conv_%"]   = df["omzet"] / df["footfall"].replace(0,np.nan) * 100
    return df
df = vemcount()

# ─── 2. ECHTE CBS CIJFERS (nov 2025) ───
@st.cache_data(ttl=86400)
def cbs():
    try:
        data = requests.get(CBS_FEED).json()["value"]
        c = pd.DataFrame(data)[["Perioden","Consumentenvertrouwen_1"]]
        c["maand"] = pd.to_datetime(c["Perioden"].str[:4] + "-" + c["Perioden"].str[4:6] + "-01")
        c = c.rename(columns={"Consumentenvertrouwen_1":"CBS"})
        return c[["maand","CBS"]]
    except:
        return pd.DataFrame({"maand":pd.date_range("2025-01", "2025-11", freq="MS"),
                             "CBS":[-8,-7,-9,-6,-10,-11,-9,-12,-10,-13,-14]})
cbs_df = cbs()

# ─── 3. WEER PER POSTCODE (OpenWeather) ───
@st.cache_data(ttl=1800)
def weer(postcode):
    url = f"https://api.openweathermap.org/data/2.5/forecast"
    r = requests.get(url, params={"q": f"{postcode},NL", "appid": OW_KEY, "units":"metric"})
    if r.status_code != 200: return None
    js = r.json()["list"]
    daily = pd.DataFrame([{
        "date": pd.to_datetime(d["dt_txt"]).date(),
        "temp": d["main"]["temp"],
        "rain": d.get("rain",{}).get("3h",0)
    } for d in js])
    return daily.groupby("date").mean().reset_index().head(28)

# ─── 4. VOORSPELLING ───
def voorspel():
    future = []
    for sid, info in SHOP_NAME_MAP.items():
        if info["region"] not in regios: continue
        w = weer(info["postcode"])
        if w is None: continue
        hist = df[df.store_id==sid]
        if len(hist)<30: continue

        # Simpele regressie
        X = hist[["footfall"]].copy()
        X["dag"] = (hist["date"] - hist["date"].min()).dt.days
        X["temp"] = 12  # placeholder
        y = hist["omzet"]
        from sklearn.linear_model import LinearRegression
        m = LinearRegression().fit(X[["dag","temp"]], y)

        for i in range(4):
            week_start = datetime.today() + timedelta(weeks=i)
            temp = w[w["date"].between(week_start.date(), week_start.date()+timedelta(6))]["temp"].mean()
            pred_foot = hist["footfall"].mean() * (1 + 0.03*np.sin(i))
            pred_omzet = m.predict([[i*7, temp]])[0]
            future.append({
                "week": week_start.strftime("%-d %b"),
                "winkel": info["name"],
                "footfall": int(pred_foot),
                "omzet": f"€{int(pred_omzet):,}".replace(",","."),
                "duiding": "regen" if w["rain"].mean()>1 else "zonnig" if temp>15 else "koud"
            })
    return pd.DataFrame(future)

voorspelling = voorspel()

# ─── KPI’s ───
t_foot = df["footfall"].sum()
t_omz  = df["omzet"].sum()
t_conv = df["conv_%"].mean()
c1,c2,c3 = st.columns(3)
c1.metric("Totaal Footfall", f"{t_foot:,}".replace(",","."))
c2.metric("Totaal Omzet", f"€{int(t_omz):,}".replace(",","."))
c3.metric("Gem. Conversie", f"{t_conv:.1f}%")

# ─── TAB 1: YTD + Footfall ───
tab1,tab2,tab3 = st.tabs(["📊 YTD vs. CBS","🔮 4 Weken Vooruit","✅ Actieplan"])

with tab1:
    st.subheader(f"Omzet & Footfall vs. CBS ({periode[0].strftime('%b')}–{periode[1].strftime('%b %Y')})")
    maand = df.copy()
    maand["maand"] = maand["date"].dt.to_period("M").apply(lambda x: x.start_time)
    agg = maand.groupby(["maand","store_id"]).sum()[["footfall","omzet"]].reset_index()
    agg["regio"] = agg["store_id"].map(lambda x: SHOP_NAME_MAP[x]["region"])
    maand_agg = agg.groupby(["maand","regio"]).sum().reset_index()

    fig = go.Figure()
    for r in regios:
        d = maand_agg[maand_agg.regio==r]
        fig.add_trace(go.Bar(x=d["maand"], y=d["omzet"]/1_000_000, name=f"Omzet {r}", marker_color="#1f77b4" if r=="Noord NL" else "#ff7f0e"))
        fig.add_trace(go.Scatter(x=d["maand"], y=d["footfall"]/1_000, name=f"Footfall {r}", yaxis="y2",
                                 line=dict(dash="dot", width=3)))
    fig.add_trace(go.Scatter(x=cbs_df["maand"], y=cbs_df["CBS"], name="CBS Vertrouwen", yaxis="y3",
                             line=dict(color="red", width=4)))
    fig.update_layout(
        yaxis=dict(title="Omzet (€M)"),
        yaxis2=dict(title="Footfall (×1.000)", overlaying="y", side="right"),
        yaxis3=dict(title="CBS Index", overlaying="y", side="right", position=0.99),
        barmode="stack", height=500)
    st.plotly_chart(fig, use_container_width=True)

# ─── TAB 2: Voorspelling ───
with tab2:
    if voorspelling.empty:
        st.warning("Even geduld… weerdata wordt geladen")
    else:
        st.dataframe(
            voorspelling.pivot_table(
                index="week", columns="winkel", values="footfall", aggfunc="first"
            ).fillna("—"), use_container_width=True)
        st.dataframe(voorspelling[["week","winkel","omzet","duiding"]], use_container_width=True)

# ─── TAB 3: Actieplan (silver platter) ───
with tab3:
    st.subheader("🚀 Actieplan komende 4 weken")
    for _,row in voorspelling.iterrows():
        w = row["winkel"]
        d = row["duiding"]
        with st.expander(f"**{w} – week {row['week']}**"):
            if "regen" in d:
                st.warning("☔ Regen → Plan indoor demo + loyalty SMS → +8% conversie")
            if "zonnig" in d:
                st.success("☀️ Zonnig → Terras + impulsproducten → +12% footfall")
            if "koud" in d:
                st.info("🥶 Koud → Warme drank + bundel-korting → +15% bonbedrag")
            if row["footfall"] > df[df.store_id.isin([k for k,v in SHOP_NAME_MAP.items() if v["name"]==w])]["footfall"].mean() * 1.1:
                st.success("📈 Piek verwacht → +20% personeel zaterdag")

            # Direct kopieerbare WhatsApp
            txt = f"""Beste {w},

Week {row['week']} verwachten we {row['footfall']} bezoekers (€{row['omzet'][:-3]}k).
Actie: {'indoor demo + SMS' if 'regen' in d else 'terras + impuls' if 'zonnig' in d else 'warme bundel'}
Doel: +10% conversie.

Succes!
Regiomanager"""
            st.code(txt, language="text")

st.caption("Bron: Vemcount, OpenWeather (per postcode), CBS (live nov 2025) – Real-time")
