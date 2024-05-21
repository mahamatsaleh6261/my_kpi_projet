import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Configuration du thème
st.set_page_config(
    page_title="Analyse des KPI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Titre de l'application
st.title("📊 Analyse des KPI pour une entreprise de vente en ligne")


# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv("atomic_data.csv")
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"])
    df["Revenue"] = df["Quantity"] * df["Unit Price"]
    return df


df = load_data()

# Sidebar pour la navigation
st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Choisissez l'analyse",
    (
        "Santé financière",
        "Chiffre d'affaires par produit",
        "Moyen de paiement le plus utilisé",
        "Ventes par pays",
        "Tendance des ventes",
        "Prédiction pour Mai 2024",
    ),
)

# Objectif 1 : Santé financière
if option == "Santé financière":
    st.subheader("💰 Santé financière")
    revenue = df["Revenue"].sum()
    st.metric(label="Revenu total", value=f"${revenue:,.2f}")
    cost = df["Quantity"].sum() * df["Unit Price"].mean()
    profit_margin = (revenue - cost) / revenue
    st.metric(label="Marge bénéficiaire", value=f"{profit_margin:.2%}")

# Objectif 2 : Chiffre d'affaires par produit
elif option == "Chiffre d'affaires par produit":
    st.subheader("🛍️ Chiffre d'affaires par produit")
    product_sales = (
        df.groupby("Product Name")["Revenue"].sum().sort_values(ascending=False)
    )
    st.bar_chart(product_sales)
    st.dataframe(product_sales)

# Objectif 3 : Moyen de paiement le plus utilisé
elif option == "Moyen de paiement le plus utilisé":
    st.subheader("💳 Moyen de paiement le plus utilisé")
    payment_methods = df["Payment Method"].value_counts()
    st.bar_chart(payment_methods)
    st.dataframe(payment_methods)

# Objectif 4 : Ventes par pays
elif option == "Ventes par pays":
    st.subheader("🌍 Ventes par pays")
    country_sales = df.groupby("Country")["Revenue"].sum().sort_values(ascending=False)
    st.bar_chart(country_sales, use_container_width=True)
    st.dataframe(country_sales)

# Objectif 5 : Tendance des ventes en fonction du temps
elif option == "Tendance des ventes":
    st.subheader("📈 Tendance des ventes")
    sales_trend = df.groupby(df["Transaction Date"])["Revenue"].sum()
    st.line_chart(sales_trend)
    st.dataframe(sales_trend)

# Objectif 6 : Prédiction du chiffre d'affaires pour Mai 2024
elif option == "Prédiction pour Mai 2024":
    st.subheader("🔮 Prédiction du chiffre d'affaires pour Mai 2024")

    # Préparer les données pour Prophet
    df_prophet = df[["Transaction Date", "Revenue"]].rename(
        columns={"Transaction Date": "ds", "Revenue": "y"}
    )

    # Initialiser et entraîner le modèle Prophet
    model = Prophet()
    model.fit(df_prophet)

    # Faire des prévisions pour les 12 prochains mois
    future = model.make_future_dataframe(periods=12, freq="M")
    forecast = model.predict(future)

    # Filtrer les prévisions pour le mois de Mai 2024
    forecast_may_2024 = forecast[
        (forecast["ds"] >= "2024-05-01") & (forecast["ds"] < "2024-06-01")
    ]
    total_revenue_may_2024 = forecast_may_2024["yhat"].sum()

    # Afficher la prédiction
    st.metric(
        label="Prédiction du chiffre d'affaires pour Mai 2024",
        value=f"${total_revenue_may_2024:,.2f}",
    )

    # Visualiser les prévisions
    st.write("Prévisions des ventes pour les 12 prochains mois")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)
