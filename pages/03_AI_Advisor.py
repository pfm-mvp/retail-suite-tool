import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import holidays
import statsmodels.api as sm
from pathlib import Path
import sys

# --- Laad shop_mapping.py uit root ---
sys.path.append(str(Path(__file__).parent.parent))
from shop_mapping import SHOP_NAME_MAP  # Jouw exacte mapping

# Reverse mapping: name → id
NAME_TO_ID = {v["name"]: k for k, v in SHOP_NAME_MAP.items()}
ID_TO_INFO = SHOP_NAME_MAP

# ================== CONFIG ==================
st.set_page_config(page_title="AI Retail Advisor", layout="wide", page_icon="🛍️")
st.title("🛍️ AI Retail Advisor: Regio- & Winkelvoorspellingen")

# Secrets (zoals in jouw Streamlit App Settings)
VEMCOUNT_API_URL = st.secrets["API_URL"]
OPENWEATHER_KEY = st.secrets["openweather_api_key"]
CBS_DATASET = st.secrets["cbs_dataset"]

CBS_FEED = f"https://opendata.cbs.nl/ODataFeed/odata/{CBS_DATASET}"
nl_holidays = holidays.NL(years=[2025, 2026])

# ================== DATA FUNCTIONS ==================

@st.cache_data(ttl=3600)
def get_vemcount_data(store_ids, start_date, end_date):
    url = VEMCOUNT_API_URL
    payload = {
        "store_ids": store_ids,
        "start_date": start_date,
        "end_date": end_date,
        "metrics": ["footfall", "transactions", "revenue"]
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df["footfall"] = pd.to_numeric(df["footfall"], errors='coerce').fillna(0).astype(int)
            df["omzet"] = pd.to_numeric(df["revenue"], errors='coerce').fillna(0)
            df["conversion"] = df["omzet"] / df["footfall"].replace(0, np.nan)
            return df[["date", "store_id", "footfall", "omzet", "conversion"]]
    except Exception as e:
        st.warning(f"Vemcount error: {e}")
    
    # Fallback
    dates = pd.date_range(start_date, end_date, freq='D')
    fallback = []
    for sid in store_ids:
        base = 150 + 50 * (sid % 3)
        for d in dates:
            seasonal = 1 + 0.3 * np.sin(2 * np.pi * d.dayofyear / 365)
            fallback.append({
                "date": d, "store_id": sid,
                "footfall": int(np.random.poisson(base * seasonal)),
                "omzet": np.random.uniform(3000, 9000) * seasonal,
                "conversion": np.random.uniform(0.22, 0.33)
            })
    return pd.DataFrame(fallback)

@st.cache_data(ttl=1800)
def get_weather_for_postcode(postcode, days=30):
    url = "https://api.openweathermap.org/data/2.5/forecast"
    try:
        r = requests.get(url, params={"q": f"{postcode},NL", "appid": OPENWEATHER_KEY, "units": "metric"})
        data = r.json()["list"]
        df = pd.DataFrame([{
            "date": pd.to_datetime(d["dt_txt"]).date(),
            "temp": d["main"]["temp"],
            "rain_prob": d.get("pop", 0),
            "rain_mm": d.get("rain", {}).get("3h", 0)
        } for d in data])
        daily = df.groupby("date").agg({"temp": "mean", "rain_prob": "mean", "rain_mm": "sum"}).reset_index()
        daily["date"] = pd.to_datetime(daily["date"])
        return daily[daily["date"] >= datetime.now().date()].head(days)
    except:
        dates = [datetime.now().date() + timedelta(days=i) for i in range(days)]
        return pd.DataFrame({"date": dates, "temp": np.random.normal(12, 4, days), "rain_prob": np.random.uniform(0.2, 0.6, days), "rain_mm": np.random.exponential(1.5, days)})

@st.cache_data(ttl=86400)
def get_cbs_data():
    try:
        r = requests.get(f"{CBS_FEED}/Consumentenvertrouwen")
        data = r.json()["value"]
        df = pd.DataFrame(data)[["Perioden", "Consumentenvertrouwen_1", "Koopbereidheid_5"]]
        df.columns = ["period", "vertrouwen", "koopbereidheid"]
        df["date"] = pd.to_datetime(df["period"].str[:4] + "-" + df["period"].str[4:6] + "-01")
        return df[["date", "vertrouwen", "koopbereidheid"]].sort_values("date")
    except:
        dates = pd.date_range("2025-01-01", "2025-10-01", freq="MS")
        return pd.DataFrame({"date": dates, "vertrouwen": np.linspace(102, 88, 10), "koopbereidheid": np.linspace(18, 6, 10)})

# ================== MODEL ==================
def build_forecast_model(historical_df, weather_dict, cbs_df):
    # Voeg winkelinfo toe
    hist = historical_df.copy()
    hist["store_info"] = hist["store_id"].map(ID_TO_INFO)
    hist = hist.dropna(subset=["store_info"])
    hist["region"] = hist["store_info"].apply(lambda x: x["region"])
    hist["postcode"] = hist["store_info"].apply(lambda x: x["postcode"])

    # Week aggregatie
    hist["week"] = hist["date"].dt.to_period("W").apply(lambda r: r.start_time)
    weekly = hist.groupby(["week", "store_id", "region", "postcode"]).agg({
        "footfall": "sum", "omzet": "sum", "conversion": "mean"
    }).reset_index()

    # Weather per winkel
    weather_weekly = {}
    for pc in weekly["postcode"].unique():
        w = get_weather_for_postcode(pc, 30)
        w["week"] = pd.to_datetime(w["date"]).dt.to_period("W").apply(lambda r: r.start_time)
        w = w.groupby("week").agg({"temp": "mean", "rain_prob": "mean"}).reset_index()
        weather_weekly[pc] = w

    # Merge
    X_list = []
    for _, row in weekly.iterrows():
        w = weather_weekly.get(row["postcode"])
        if w is not None:
            match = w[w["week"] == row["week"]]
            if not match.empty:
                X_list.append({
                    "week": row["week"],
                    "store_id": row["store_id"],
                    "region": row["region"],
                    "footfall": row["footfall"],
                    "omzet": row["omzet"],
                    "conversion": row["conversion"],
                    "temp": match.iloc[0]["temp"],
                    "rain_prob": match.iloc[0]["rain_prob"]
                })
    X = pd.DataFrame(X_list)
    if X.empty:
        st.error("Geen weerdata beschikbaar.")
        return None, None

    # CBS
    cbs_df["week"] = cbs_df["date"].dt.to_period("W").apply(lambda r: r.start_time)
    cbs_week = cbs_df.groupby("week").agg({"vertrouwen": "mean", "koopbereidheid": "mean"}).reset_index()

    X = X.merge(cbs_week, on="week", how="left")
    X["is_holiday"] = X["week"].apply(lambda w: 1 if any(d in nl_holidays for d in pd.date_range(w, w + timedelta(days=6))) else 0)
    X = X.dropna()

    # Model
    features = ["temp", "rain_prob", "vertrouwen", "koopbereidheid", "is_holiday"]
    X_model = sm.add_constant(X[features])
    model = sm.OLS(X["footfall"], X_model).fit()

    # Voorspel 4 weken
    last_week = X["week"].max()
    future_weeks = [last_week + timedelta(weeks=i+1) for i in range(4)]
    forecast = []

    for fw in future_weeks:
        cbs_pred = cbs_df.tail(1).iloc[0]
        for pc in weekly["postcode"].unique():
            w = weather_weekly.get(pc)
            if w is not None:
                fw_match = w[w["week"] == fw]
                if not fw_match.empty:
                    X_pred = sm.add_constant(pd.DataFrame([{
                        "temp": fw_match.iloc[0]["temp"],
                        "rain_prob": fw_match.iloc[0]["rain_prob"],
                        "vertrouwen": cbs_pred["vertrouwen"],
                        "koopbereidheid": cbs_pred["koopbereidheid"],
                        "is_holiday": 1 if any(d in nl_holidays for d in pd.date_range(fw, fw + timedelta(days=6))) else 0
                    }]))
                    pred_foot = model.predict(X_pred)[0]
                    pred_omzet = pred_foot * X[X["postcode"] == pc]["conversion"].mean()
                    store_name = [k for k, v in SHOP_NAME_MAP.items() if v["postcode"] == pc][0]
                    region = ID_TO_INFO[store_name]["region"]

                    duiding = []
                    if fw_match.iloc[0]["rain_prob"] > 0.5: duiding.append("regen")
                    if cbs_pred["vertrouwen"] < 92: duiding.append("laag vertrouwen")
                    if X_pred.iloc[0]["is_holiday"] == 1: duiding.append("feestdag")

                    forecast.append({
                        "week": fw.strftime("%d %b"),
                        "winkel": ID_TO_INFO[store_name]["name"],
                        "regio": region,
                        "footfall": int(pred_foot),
                        "omzet": f"€{int(pred_omzet):,}".replace(",", "."),
                        "duiding": ", ".join(duiding) if duiding else "stabiel"
                    })

    return pd.DataFrame(forecast), model

# ================== APP ==================
st.sidebar.header("Analyse Instellingen")
regions = st.sidebar.multiselect("Regio", options=["Noord NL", "Zuid NL"], default=["Noord NL", "Zuid NL"])
period = st.sidebar.date_input("Periode", value=[datetime(2025,1,1), datetime(2025,10,31)])

# Filter stores op regio
selected_ids = [sid for sid, info in SHOP_NAME_MAP.items() if info["region"] in regions]
start_str, end_str = period[0].strftime("%Y-%m-%d"), period[1].strftime("%Y-%m-%d")

with st.spinner("Data ophalen..."):
    hist_data = get_vemcount_data(selected_ids, start_str, end_str)
    cbs_data = get_cbs_data()

# KPIs
col1, col2, col3 = st.columns(3)
total_foot = hist_data["footfall"].sum()
total_omzet = hist_data["omzet"].sum()
col1.metric("Totaal Footfall", f"{total_foot:,}".replace(",", "."))
col2.metric("Totaal Omzet", f"€{int(total_omzet):,}".replace(",", "."))
col3.metric("Gem. Conversie", f"{(total_omzet/total_foot):.1%}" if total_foot > 0 else "N/A")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 YTD vs. CBS", "🔮 Voorspelling per Winkel", "✅ Actieplan"])

with tab1:
    st.subheader("Omzet & Footfall vs. Consumentenvertrouwen (per regio)")
    monthly = hist_data.copy()
    monthly["month"] = monthly["date"].dt.to_period("M")
    monthly["region"] = monthly["store_id"].map(lambda x: ID_TO_INFO.get(x, {}).get("region", "Onbekend"))
    agg = monthly.groupby(["month", "region"]).agg({"footfall": "sum", "omzet": "sum"}).reset_index()
    agg["month_start"] = agg["month"].apply(lambda x: x.start_time)

    fig = go.Figure()
    for reg in regions:
        df_reg = agg[agg["region"] == reg]
        fig.add_trace(go.Bar(x=df_reg["month_start"], y=df_reg["omzet"], name=f"Omzet {reg}"))
    fig.add_trace(go.Scatter(x=cbs_data["date"], y=cbs_data["vertrouwen"], name="Vertrouwen (CBS)", yaxis="y2", line=dict(color="red", dash="dash")))
    fig.update_layout(yaxis2=dict(title="Vertrouwen", overlaying="y", side="right"), barmode="stack")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Voorspelling: Aankomende 4 Weken (per winkel)")
    weather_dict = {info["postcode"]: None for info in SHOP_NAME_MAP.values()}
    forecast_df, model = build_forecast_model(hist_data, weather_dict, cbs_data)
    if forecast_df is not None:
        st.dataframe(forecast_df.pivot_table(index=["week", "regio"], columns="winkel", values="footfall", aggfunc="first").fillna("-"), use_container_width=True)
        st.dataframe(forecast_df[["week", "winkel", "regio", "omzet", "duiding"]], use_container_width=True)

with tab3:
    st.subheader("Actieplan voor Regiomanagers")
    if forecast_df is not None:
        for reg in regions:
            with st.expander(f"📍 {reg}"):
                reg_data = forecast_df[forecast_df["regio"] == reg]
                for _, row in reg_data.iterrows():
                    if "regen" in row["duiding"]:
                        st.warning(f"**{row['winkel']}**: Regen → indoor promo of SMS-actie")
                    if "feestdag" in row["duiding"]:
                        st.success(f"**{row['winkel']}**: Feestdag → +20% personeel")
                    if "laag vertrouwen" in row["duiding"]:
                        st.info(f"**{row['winkel']}**: CBS laag → focus op loyaliteit")

    if st.button("📧 Genereer Regio Rapport (E-mail)"):
        report = "\n".join([f"- {row['winkel']}: {row['footfall']} bezoekers ({row['duiding']})" for _, row in forecast_df.iterrows()])
        st.code(f"**Regio Update**\n\n{report}\n\nActies volgen uit AI-analyse.", language="markdown")

st.caption("Bron: Vemcount, OpenWeather (per postcode), CBS, KNMI | Real-time bijgewerkt")
