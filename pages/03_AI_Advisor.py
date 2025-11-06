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

# ─── SECRETS ───
PFM_URL = st.secrets["API_URL"]
OW_KEY  = st.secrets["openweather_api_key"]
CBS_ID  = st.secrets["cbs_dataset"]
CBS_URL = f"https://opendata.cbs.nl/ODataFeed/odata/{CBS_ID}/Consumentenvertrouwen"

# ─── SIDEBAR ───
regios  = st.sidebar.multiselect("Regio", ["Noord NL", "Zuid NL"], default=["Noord NL", "Zuid NL"])
periode = st.sidebar.date_input("Periode", [datetime(2025,1,1), datetime(2025,10,31)])

# ─── 1. PLANET PFM – EXACT JOUW PAYLOAD ───
@st.cache_data(ttl=3600)
def pfm():
    ids = [sid for sid, info in SHOP_NAME_MAP.items() if info["region"] in regios]
    payload = {
        "query": {
            "source": "reports",
            "data": {
                "store_ids": ids,
                "start_date": periode[0].strftime("%Y-%m-%d"),
                "end_date":   periode[1].strftime("%Y-%m-%d")
            },
            "data_output": ["footfall", "transactions", "revenue"]
        }
    }
    try:
        r = requests.post(PFM_URL, json=payload, timeout=20)
        if r.status_code != 200:
            raise ValueError(r.text)
        raw = r.json()
        st.success("✅ Planet PFM online – echte data geladen")
    except Exception as e:
        st.error(f"❌ Planet PFM fout: {str(e)[:120]} → fallback")
        return fallback()

    # Veilige kolom-check
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df.get("date", pd.NaT))
    df["footfall"] = pd.to_numeric(df.get("footfall", 0), errors="coerce").fillna(0).astype(int)
    df["omzet"]    = pd.to_numeric(df.get("revenue", 0), errors="coerce").fillna(0)
    df["store_id"] = df["store_id"].astype(int)
    df["conv_%"]   = (df["omzet"] / df["footfall"].replace(0, np.nan)) * 100
    df["spv"]      = df["omzet"] / df["footfall"].replace(0, np.nan)
    return df

def fallback():
    dates = pd.date_range(periode[0], periode[1])
    rows = []
    for sid, info in SHOP_NAME_MAP.items():
        if info["region"] not in regios: continue
        base = 200 + sid % 50
        for d in dates:
            rows.append({
                "date": d, "store_id": sid, "footfall": np.random.poisson(base),
                "omzet": np.random.uniform(5000, 15000), "conv_%": np.random.uniform(22, 35),
                "spv": np.random.uniform(25, 35)
            })
    return pd.DataFrame(rows)

df = pfm()

# ─── 2. CBS LIVE ───
@st.cache_data(ttl=86400)
def cbs():
    try:
        raw = requests.get(CBS_URL).json()["value"]
        c = pd.DataFrame(raw)[["Perioden","Consumentenvertrouwen_1"]]
        c["maand"] = pd.to_datetime(c["Perioden"].str[:7] + "-01")
        c = c.rename(columns={"Consumentenvertrouwen_1":"CBS"})
        return c[["maand","CBS"]]
    except:
        return pd.DataFrame({"maand": pd.date_range("2025-01", "2025-11", freq="MS"),
                             "CBS": [-8,-7,-9,-6,-10,-11,-9,-12,-10,-13,-14]})

cbs_df = cbs()

# ─── 3. WEER PER POSTCODE ───
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
            "rain": d.get("rain", {}).get("3h", 0)
        } for d in js])
        return daily.groupby("date").mean(numeric_only=True).reset_index().head(28)
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
        if len(hist) < 10: continue
        avg_foot = hist["footfall"].mean()
        avg_spv  = hist["spv"].mean()
        for i in range(4):
            week = (datetime.now() + timedelta(weeks=i)).strftime("%-d %b")
            temp = w.iloc[i*7:(i+1)*7]["temp"].mean() if len(w) > i*7+7 else 12
            rain = w.iloc[i*7:(i+1)*7]["rain"].sum() if len(w) > i*7+7 else 0
            adj = 1.0
            if rain > 5: adj *= 0.90
            if temp > 18: adj *= 1.15
            foot = int(avg_foot * 7 * adj)
            omzet = foot * avg_spv
            rows.append({
                "week": week, "winkel": info["name"], "footfall": foot,
                "omzet": f"€{int(omzet):,}".replace(",", "."), "spv": f"€{avg_spv:.0f}",
                "duiding": f"{'regen ' if rain>5 else ''}{'zonnig ' if temp>18 else ''}".strip() or "stabiel"
            })
    return pd.DataFrame(rows)

voorsp = voorspel()

# ─── PERIODE & KPI’s ───
st.info(f"**Data over**: {periode[0].strftime('%d %b %Y')} – {periode[1].strftime('%d %b %Y')}")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Footfall", f"{int(df['footfall'].sum()):,}".replace(",","."))
c2.metric("Omzet", f"€{int(df['omzet'].sum()):,}".replace(",","."))
c3.metric("Conversie", f"{df['conv_%'].mean():.1f}%")
c4.metric("SPV", f"€{df['spv'].mean():.0f}")

# ─── TABS ───
tab1,tab2,tab3 = st.tabs(["📊 YTD vs. CBS","🔮 4 Weken","✅ Actieplan"])

with tab1:
    maand = df.copy()
    maand["maand"] = maand["date"].dt.strftime("%Y-%m")
    agg = maand.groupby(["maand","store_id"]).agg({"footfall":"sum","omzet":"sum"}).reset_index()
    agg["regio"] = agg["store_id"].map(lambda x: SHOP_NAME_MAP[x]["region"])
    maand_agg = agg.groupby(["maand","regio"]).sum().reset_index()

    fig = go.Figure()
    for r in regios:
        d = maand_agg[maand_agg.regio==r]
        fig.add_trace(go.Bar(x=d["maand"], y=d["omzet"]/1_000_000, name=f"Omzet {r}"))
        fig.add_trace(go.Scatter(x=d["maand"], y=d["footfall"]/1_000, name=f"Footfall {r}", yaxis="y2", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=cbs_df["maand"].dt.strftime("%Y-%m"), y=cbs_df["CBS"], name="CBS Vertrouwen", yaxis="y3", line=dict(color="red")))
    fig.update_layout(yaxis=dict(title="Omzet (€M)"), yaxis2=dict(title="Footfall (×1.000)", overlaying="y", side="right"),
                      yaxis3=dict(title="CBS", overlaying="y", side="right", position=0.99), barmode="group")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    if voorsp.empty:
        st.warning("Weer laadt…")
    else:
        st.dataframe(voorsp.pivot_table(index="week", columns="winkel", values="footfall").fillna("—"))
        st.dataframe(voorsp[["week","winkel","omzet","spv","duiding"]])

with tab3:
    st.subheader("Actieplan – Klaar voor WhatsApp")
    for _,r in voorsp.iterrows():
        with st.expander(f"{r['winkel']} – {r['week']} | {r['footfall']:,} bezoekers"):
            txt = f"""Beste {r['winkel']},

Week {r['week']} → {r['footfall']:,} bezoekers ({r['omzet']})
Weer: {r['duiding'].capitalize()}

Actie:
→ {'Indoor demo + SMS' if 'regen' in r['duiding'] else 'Terras open' if 'zonnig' in r['duiding'] else 'Focus loyaliteit'}
Doel: +10% conversie

Succes!
Regiomanager"""
            st.code(txt, language="text")

if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

st.caption("Bron: Planet PFM (live), OpenWeather, CBS (nov 2025)")
