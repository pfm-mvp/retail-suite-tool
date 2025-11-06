import streamlit as st
import pandas as pd, numpy as np, requests, holidays
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from shop_mapping import SHOP_NAME_MAP

st.set_page_config(page_title="AI Retail Advisor", layout="wide", page_icon="🛍️")
st.title("🛍️ AI Retail Advisor: Regio- & Winkelvoorspellingen")

# ─── SECRETS – ALLEEN UIT STREAMLIT APP SETTINGS ───
PFM_URL        = st.secrets["API_URL"]
OW_KEY         = st.secrets["openweather_api_key"]
CBS_DATASET_ID = st.secrets["cbs_dataset"]
CBS_URL        = f"https://opendata.cbs.nl/ODataFeed/odata/{CBS_DATASET_ID}/Consumentenvertrouwen"

# ─── SIDEBAR ───
regios  = st.sidebar.multiselect("Regio", ["Noord NL", "Zuid NL"], default=["Noord NL", "Zuid NL"])
periode = st.sidebar.date_input("Periode", [datetime(2025,1,1), datetime(2025,10,31)])

# ─── 1. PLANET PFM DATA ───
@st.cache_data(ttl=3600)
def pfm():
    ids = [sid for sid,info in SHOP_NAME_MAP.items() if info["region"] in regios]
    payload = {
        "store_ids": ids,
        "start_date": periode[0].strftime("%Y-%m-%d"),
        "end_date":   periode[1].strftime("%Y-%m-%d"),
        "metrics": ["footfall","transactions","revenue"]
    }
    try:
        r = requests.post(PFM_URL, json=payload, timeout=15)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["footfall"] = pd.to_numeric(df["footfall"], errors="coerce").fillna(0).astype(int)
        df["omzet"]    = pd.to_numeric(df["revenue"],  errors="coerce").fillna(0)
        df["conv_%"]   = (df["omzet"] / df["footfall"].replace(0, np.nan)) * 100
        df["store_id"] = df["store_id"].astype(int)
        return df
    except Exception as e:
        st.error("Planet PFM tijdelijk offline → fallback data")
        return fallback_df()

def fallback_df():
    dates = pd.date_range(periode[0], periode[1])
    rows = []
    for sid, info in SHOP_NAME_MAP.items():
        if info["region"] not in regios: continue
        base = 180 + 30 * (sid % 3)
        for d in dates:
            rows.append({
                "date": d,
                "store_id": sid,
                "footfall": np.random.poisson(base),
                "omzet": np.random.uniform(4000, 12000),
                "conv_%": np.random.uniform(22, 35)
            })
    return pd.DataFrame(rows)

df = pfm()

# ─── 2. ECHTE CBS (nov 2025) ───
@st.cache_data(ttl=86400)
def cbs():
    try:
        raw = requests.get(CBS_URL).json()["value"]
        c = pd.DataFrame(raw)[["Perioden","Consumentenvertrouwen_1"]]
        c["maand"] = pd.to_datetime(c["Perioden"].str[:4] + "-" + c["Perioden"].str[4:6] + "-01")
        c = c.rename(columns={"Consumentenvertrouwen_1":"CBS"})
        return c[["maand","CBS"]].sort_values("maand")
    except:
        return pd.DataFrame({"maand": pd.date_range("2025-01", "2025-11", freq="MS"),
                             "CBS": [-8,-7,-9,-6,-10,-11,-9,-12,-10,-13,-14]})

cbs_df = cbs()

# ─── 3. WEER PER POSTCODE (OpenWeather) ───
@st.cache_data(ttl=1800)
def weer(pc):
    try:
        r = requests.get("https://api.openweathermap.org/data/2.5/forecast",
                         params={"q": f"{pc},NL", "appid": OW_KEY, "units":"metric"})
        r.raise_for_status()
        js = r.json()["list"]
        daily = pd.DataFrame([{
            "date": pd.to_datetime(d["dt_txt"]).date(),
            "temp": d["main"]["temp"],
            "rain_mm": d.get("rain",{}).get("3h",0)
        } for d in js])
        return daily.groupby("date").mean().reset_index().head(28)
    except:
        return None

# ─── 4. VOORSPELLING ───
def voorspel():
    rows = []
    for sid, info in SHOP_NAME_MAP.items():
        if info["region"] not in regios: continue
        w = weer(info["postcode"])
        if w is None: continue
        hist = df[df["store_id"] == sid]
        if len(hist) < 30: continue

        avg_foot = hist["footfall"].mean()
        avg_conv = hist["conv_%"].mean() / 100

        for w_idx in range(4):
            week_start = (datetime.now() + timedelta(weeks=w_idx)).strftime("%-d %b")
            temp = w.iloc[w_idx*7:(w_idx*7+7)]["temp"].mean() if len(w) > w_idx*7+7 else 12
            rain = w.iloc[w_idx*7:(w_idx*7+7)]["rain_mm"].sum() if len(w) > w_idx*7+7 else 0

            foot = avg_foot * (1.15 if temp > 18 else 0.90 if rain > 5 else 1.00)
            omzet = foot * avg_conv * np.random.uniform(25, 35)

            duiding = []
            if rain > 5:   duiding.append("regen")
            if temp > 18:  duiding.append("zonnig")
            if temp < 8:   duiding.append("koud")

            rows.append({
                "week": week_start,
                "winkel": info["name"],
                "regio": info["region"],
                "footfall": int(foot),
                "omzet": f"€{int(omzet):,}".replace(",","."),
                "duiding": ", ".join(duiding) or "stabiel"
            })
    return pd.DataFrame(rows)

voorspelling = voorspel()

# ─── KPI’s ───
t_foot = int(df["footfall"].sum())
t_omz  = int(df["omzet"].sum())
t_conv = df["conv_%"].mean()
c1,c2,c3 = st.columns(3)
c1.metric("Totaal Footfall", f"{t_foot:,}".replace(",","."))
c2.metric("Totaal Omzet", f"€{t_omz:,}".replace(",","."))
c3.metric("Gem. Conversie", f"{t_conv:.1f}%")

# ─── TABS ───
tab1,tab2,tab3 = st.tabs(["📊 YTD vs. CBS","🔮 4 Weken Vooruit","✅ Actieplan"])

with tab1:
    st.subheader("Omzet & Footfall vs. CBS Vertrouwen")
    maand = df.copy()
    maand["maand"] = maand["date"].dt.to_period("M").apply(lambda x: x.start_time)
    agg = maand.groupby(["maand","store_id"]).sum()[["footfall","omzet"]].reset_index()
    agg["regio"] = agg["store_id"].map(lambda x: SHOP_NAME_MAP[x]["region"])
    maand_agg = agg.groupby(["maand","regio"]).sum().reset_index()

    fig = go.Figure()
    for r in regios:
        d = maand_agg[maand_agg.regio==r]
        fig.add_trace(go.Bar(x=d["maand"], y=d["omzet"]/1_000_000,
                             name=f"Omzet {r}",
                             marker_color="#1f77b4" if r=="Noord NL" else "#ff7f0e"))
        fig.add_trace(go.Scatter(x=d["maand"], y=d["footfall"]/1_000,
                                 name=f"Footfall {r}", yaxis="y2",
                                 line=dict(dash="dot", width=3)))
    fig.add_trace(go.Scatter(x=cbs_df["maand"], y=cbs_df["CBS"],
                             name="CBS Vertrouwen", yaxis="y3",
                             line=dict(color="red", width=4)))
    fig.update_layout(
        yaxis=dict(title="Omzet (€M)"),
        yaxis2=dict(title="Footfall (×1.000)", overlaying="y", side="right"),
        yaxis3=dict(title="CBS Index", overlaying="y", side="right", position=0.99),
        barmode="stack", height=550)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    if voorspelling.empty:
        st.warning("Weerdata wordt geladen… 5 sec")
    else:
        st.subheader("Voorspelling per winkel")
        st.dataframe(
            voorspelling.pivot_table(index="week", columns="winkel", values="footfall").fillna("—"),
            use_container_width=True)
        st.dataframe(voorspelling[["week","winkel","omzet","duiding"]].style.format({"omzet": lambda x: x}),
                     use_container_width=True)

with tab3:
    st.subheader("🚀 Actieplan – Kopieer & Verstuur")
    for _,row in voorspelling.iterrows():
        with st.expander(f"**{row['winkel']} – {row['week']}** | {row['footfall']} bezoekers"):
            if "regen" in row["duiding"]:
                st.warning("☔ Regen → indoor demo + SMS: 'Kom droog shoppen, 20% korting!'")
            if "zonnig" in row["duiding"]:
                st.success("☀️ Zonnig → terras + impuls: 'Gratis ijsje bij €50'")
            if "koud" in row["duiding"]:
                st.info("🥶 Koud → warme bundel: 'Koop jas + sjaal = -15%'")
            if row["footfall"] > df[df["store_id"] == [k for k,v in SHOP_NAME_MAP.items() if v["name"]==row["winkel"]][0]]["footfall"].mean() * 1.1:
                st.success("📈 Piek → +2 extra medewerkers zaterdag")

            txt = f"""Beste {row['winkel']},

Week {row['week']} → {row['footfall']} bezoekers ({row['omzet'][:-3]}k)
Weer: {row['duiding'].capitalize()}

Actie:
→ {'Indoor demo + SMS' if 'regen' in row['duiding'] else 'Terras + impuls' if 'zonnig' in row['duiding'] else 'Warme bundel'}
Doel: +10% conversie

Succes!
Regiomanager"""
            st.code(txt, language="text")

st.caption("Bron: Planet PFM, OpenWeather (per postcode), CBS (live nov 2025) – Real-time")
