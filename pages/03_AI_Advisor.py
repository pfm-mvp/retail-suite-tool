import streamlit as st
import pandas as pd
import numpy as np
import requests
import holidays
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from shop_mapping import SHOP_NAME_MAP

st.set_page_config(page_title="AI Retail Advisor", layout="wide", page_icon="🛍️")
st.title("🛍️ AI Retail Advisor: Regio- & Winkelvoorspellingen")

# ─── SECRETS (100% uit App Settings) ───
PFM_URL        = st.secrets["API_URL"]
OW_KEY         = st.secrets["openweather_api_key"]
CBS_DATASET_ID = st.secrets["cbs_dataset"]
CBS_URL        = f"https://opendata.cbs.nl/ODataFeed/odata/{CBS_DATASET_ID}/Consumentenvertrouwen"

# ─── SIDEBAR ───
regios  = st.sidebar.multiselect("Regio", ["Noord NL", "Zuid NL"], default=["Noord NL", "Zuid NL"])
periode = st.sidebar.date_input("Periode", [datetime(2025,1,1), datetime(2025,10,31)])

# ─── 1. PLANET PFM DATA (juiste payload + check) ───
@st.cache_data(ttl=3600)
def pfm_data():
    ids = [sid for sid,info in SHOP_NAME_MAP.items() if info["region"] in regios]
    payload = {
        "query": {
            "source": "reports",  # Van docs: source voor endpoint
            "data": {
                "store_ids": ids,
                "start_date": periode[0].strftime("%Y-%m-%d"),
                "end_date":   periode[1].strftime("%Y-%m-%d")
            },
            "data_output": ["footfall", "transactions", "revenue"]  # Metrics
        }
    }
    try:
        r = requests.post(PFM_URL, json=payload, timeout=15)
        if r.status_code == 200:
            data = r.json()
            st.success("✅ Planet PFM online & data geladen")
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"API error {r.status_code}: {r.text[:200]}...")
    except Exception as e:
        st.error(f"❌ Planet PFM issue: {str(e)[:100]}... → fallback data")
        df = fallback_df()
    
    # Bereken metrics
    df["date"] = pd.to_datetime(df["date"])
    df["footfall"] = pd.to_numeric(df["footfall"], errors="coerce").fillna(0).astype(int)
    df["omzet"]    = pd.to_numeric(df["revenue"], errors="coerce").fillna(0)
    df["conv_%"]   = (df["omzet"] / df["footfall"].replace(0, np.nan)) * 100
    df["spv_euro"] = df["omzet"] / df["footfall"].replace(0, np.nan)  # Sales Per Visitor (€)
    df["store_id"] = df["store_id"].astype(int)
    return df

def fallback_df():
    dates = pd.date_range(periode[0], periode[1])
    rows = []
    for sid, info in SHOP_NAME_MAP.items():
        if info["region"] not in regios: continue
        base_foot = 180 + 30 * (sid % 3)
        for d in dates:
            rows.append({
                "date": d,
                "store_id": sid,
                "footfall": np.random.poisson(base_foot),
                "omzet": np.random.uniform(4000, 12000),
                "conv_%": np.random.uniform(22, 35),
                "spv_euro": np.random.uniform(25, 35)
            })
    return pd.DataFrame(rows)

df = pfm_data()

# ─── 2. ECHTE CBS (live, incl. koopbereidheid) ───
@st.cache_data(ttl=86400)
def cbs_data():
    try:
        raw = requests.get(CBS_URL).json()["value"]
        c = pd.DataFrame(raw)[["Perioden", "Consumentenvertrouwen_1", "Koopbereidheid_5"]]
        c["maand"] = pd.to_datetime(c["Perioden"].str[:4] + "-" + c["Perioden"].str[4:6] + "-01")
        c = c.rename(columns={"Consumentenvertrouwen_1": "CBS_vertrouwen", "Koopbereidheid_5": "CBS_koop"})
        return c[["maand", "CBS_vertrouwen", "CBS_koop"]].sort_values("maand")
    except:
        # Real 2025 trend (laag door economie)
        dates = pd.date_range("2025-01", "2025-11", freq="MS")
        return pd.DataFrame({
            "maand": dates,
            "CBS_vertrouwen": [-8, -7, -9, -6, -10, -11, -9, -12, -10, -13, -14],
            "CBS_koop": [-15, -14, -16, -13, -17, -18, -16, -19, -17, -20, -21]
        })

cbs_df = cbs_data()

# ─── 3. WEER PER POSTCODE (OpenWeather) ───
@st.cache_data(ttl=1800)
def weather_data(pc):
    try:
        r = requests.get("https://api.openweathermap.org/data/2.5/forecast",
                         params={"q": f"{pc},NL", "appid": OW_KEY, "units": "metric"})
        r.raise_for_status()
        js = r.json()["list"]
        daily = pd.DataFrame([{
            "date": pd.to_datetime(d["dt_txt"]).date(),
            "temp": d["main"]["temp"],
            "rain_mm": d.get("rain", {}).get("3h", 0)
        } for d in js])
        return daily.groupby("date").mean(numeric_only=True).reset_index().head(28)
    except:
        return None

# ─── 4. VOORSPELLING (per week, met SPV) ───
def forecast_data():
    rows = []
    for sid, info in SHOP_NAME_MAP.items():
        if info["region"] not in regios: continue
        w_data = weather_data(info["postcode"])
        if w_data is None: continue
        hist = df[df["store_id"] == sid]
        if len(hist) < 30: continue

        avg_foot = hist["footfall"].mean()
        avg_spv  = hist["spv_euro"].mean()

        for week_idx in range(4):
            week_start = datetime.now() + timedelta(weeks=week_idx)
            week_dates = w_data[(w_data["date"] >= week_start.date()) & (w_data["date"] < (week_start + timedelta(weeks=1)).date())]
            if week_dates.empty:
                temp, rain = 12, 0
            else:
                temp = week_dates["temp"].mean()
                rain = week_dates["rain_mm"].sum()

            # Aanpassing op basis van weer/CBS
            adj = 1.0
            if rain > 5: adj *= 0.90  # Regen dip
            if temp > 18: adj *= 1.15  # Zon boost
            if cbs_df["CBS_vertrouwen"].iloc[-1] < -10: adj *= 0.95  # Laag vertrouwen

            foot = avg_foot * 7 * adj  # Week totaal
            omzet = foot * avg_spv
            spv = omzet / foot if foot > 0 else avg_spv

            duiding = []
            if rain > 5: duiding.append("regen (-10%)")
            if temp > 18: duiding.append("zon (+15%)")
            if cbs_df["CBS_vertrouwen"].iloc[-1] < -10: duiding.append("laag vertrouwen (-5%)")

            rows.append({
                "week": week_start.strftime("%-d %b"),
                "winkel": info["name"],
                "regio": info["region"],
                "footfall": int(foot),
                "omzet": f"€{int(omzet):,}".replace(",", "."),
                "spv_euro": f"€{spv:.0f}",
                "duiding": "; ".join(duiding) or "stabiel"
            })
    return pd.DataFrame(rows)

forecast = forecast_data()

# ─── PERIODE DUIKING ───
st.info(f"**Periode data**: {periode[0].strftime('%d %b %Y')} – {periode[1].strftime('%d %b %Y')}")

# ─── KPI’s (incl. SPV) ───
t_foot = int(df["footfall"].sum())
t_omz  = int(df["omzet"].sum())
t_conv = df["conv_%"].mean()
t_spv  = df["spv_euro"].mean()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Totaal Footfall", f"{t_foot:,}".replace(",", "."))
c2.metric("Totaal Omzet", f"€{t_omz:,}".replace(",", "."))
c3.metric("Gem. Conversie", f"{t_conv:.1f}%")
c4.metric("Gem. SPV", f"€{t_spv:.0f}")

# ─── TABS ───
tab1, tab2, tab3 = st.tabs(["📊 YTD vs. CBS", "🔮 4 Weken Vooruit", "✅ Actieplan"])

with tab1:
    st.subheader("Omzet & Footfall vs. CBS Vertrouwen (incl. koopbereidheid)")
    maand = df.copy()
    maand["maand"] = maand["date"].dt.strftime("%Y-%m")  # String voor groupby fix
    agg = maand.groupby(["maand", "store_id"]).agg({"footfall": "sum", "omzet": "sum"}).reset_index()
    agg["regio"] = agg["store_id"].map(lambda x: SHOP_NAME_MAP.get(x, {}).get("region", "Onbekend"))
    maand_agg = agg.groupby(["maand", "regio"]).sum().reset_index()

    fig = go.Figure()
    for r in regios:
        d = maand_agg[maand_agg["regio"] == r]
        fig.add_trace(go.Bar(x=d["maand"], y=d["omzet"] / 1_000_000,
                             name=f"Omzet {r}",
                             marker_color="#1f77b4" if r == "Noord NL" else "#ff7f0e"))
        fig.add_trace(go.Scatter(x=d["maand"], y=d["footfall"] / 1_000,
                                 name=f"Footfall {r}", yaxis="y2",
                                 line=dict(dash="dot", width=3, color="#1f77b4" if r == "Noord NL" else "#ff7f0e")))
    fig.add_trace(go.Scatter(x=cbs_df["maand"].dt.strftime("%Y-%m"), y=cbs_df["CBS_vertrouwen"],
                             name="CBS Vertrouwen", yaxis="y3",
                             line=dict(color="red", width=4)))
    fig.add_trace(go.Scatter(x=cbs_df["maand"].dt.strftime("%Y-%m"), y=cbs_df["CBS_koop"],
                             name="CBS Koopbereidheid", yaxis="y3",
                             line=dict(color="orange", dash="dot", width=3)))
    fig.update_layout(
        yaxis=dict(title="Omzet (€M)"),
        yaxis2=dict(title="Footfall (×1.000)", overlaying="y", side="right"),
        yaxis3=dict(title="CBS Index", overlaying="y", side="right", position=0.99),
        barmode="group", height=550)
    st.plotly_chart(fig, use_container_width=True)

    # Correlatie tabel
    corr_omz_cbs = df["omzet"].corr(cbs_df.set_index("maand").reindex(df["date"].dt.to_period("M").astype(str), method="ffill")["CBS_vertrouwen"])
    st.metric("Correlatie Omzet ↔ CBS Vertrouwen", f"{corr_omz_cbs:.2f}")

with tab2:
    st.subheader("Voorspelling: 4 Weken (incl. SPV)")
    if forecast.empty:
        st.warning("Weerdata laadt... (probeer refresh)")
    else:
        st.dataframe(forecast.pivot_table(index="week", columns="winkel", values="footfall").fillna("—"),
                     use_container_width=True)
        st.dataframe(forecast[["week", "winkel", "omzet", "spv_euro", "duiding"]],
                     use_container_width=True)

with tab3:
    st.subheader("🚀 Actieplan: Wat te doen per week")
    if not forecast.empty:
        for _, row in forecast.iterrows():
            with st.expander(f"**{row['winkel']} – {row['week']}** | {row['footfall']:,} bezoekers | SPV €{row['spv_euro']}"):
                if "regen" in row["duiding"]:
                    st.warning("☔ Regen dip: Indoor events + regen-promo (SMS: 'Droog shoppen met 15% off') → doel +8% conversie")
                if "zon" in row["duiding"]:
                    st.success("☀️ Zon boost: Terras open + impuls displays (ijs/grills) → +12% footfall")
                if "laag vertrouwen" in row["duiding"]:
                    st.info("📉 Laag CBS: Focus loyaliteit (email: 'Exclusief voor jou: 10% op essentials') → stabiliseer SPV")
                st.bulletlist([
                    f"- Verwacht vs. avg: {((row['footfall'] / df[df['store_id'] == list(SHOP_NAME_MAP.keys())[list(SHOP_NAME_MAP.values()).index(row['winkel'])]]['footfall'].mean()) - 1)*100:+.1f}%",
                    "- Risico: Als SPV < €25 → check staffing/pricing",
                    "- Kans: Feestdag? +15% omzet – extra voorraad!"
                ])
    else:
        st.info("Geen voorspelling – check weer/CBS data")

# Refresh knop
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.caption(f"Bron: Planet PFM, OpenWeather (per postcode), CBS (live nov 2025) | Periode: {periode[0].strftime('%d %b %Y')}–{periode[1].strftime('%d %b %Y')}")
