# pages/02_Metric_Outlier.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 Metric Outlier Detection")

address = st.text_input("Enter address for Metric Outlier Detection:")

if address:
    try:
        response = requests.get(f"http://localhost:3000/metrics/{address}")
        data = response.json()
    except Exception as e:
        st.error(f"API call failed: {e}")
        st.stop()

    df = pd.DataFrame(data)
    if df.empty:
        st.warning("No data found for this address.")
    else:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

        # Convert numeric columns
        numeric_cols = [
            'total_maker', 'total_volume', 'maker_buy', 'total_buy',
            'maker_sell', 'total_sell', 'total_supply', 'total_transfer',
            'total_transfer_amount', 'total_mint', 'total_mint_amount',
            'total_burn', 'total_burn_amount'
        ]
        for col in numeric_cols:
            df[col] = df[col].astype(float)

        # Isolation Forest
        X = df[numeric_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
        model.fit(X_scaled)

        scores = model.decision_function(X_scaled)
        threshold = np.percentile(scores, 5)
        df['anomaly'] = (scores <= threshold).astype(int)

        st.subheader(f"Found {df['anomaly'].sum()} outliers")

        # Scatter plot
        fig, ax = plt.subplots()
        ax.scatter(df['total_buy'], df['total_sell'], c=df['anomaly'], cmap='coolwarm', s=50)
        ax.set_xlabel('total_buy')
        ax.set_ylabel('total_sell')
        st.pyplot(fig)

        # Pairplot (first 3 numeric columns)
        sns.pairplot(df, vars=numeric_cols[:3], hue='anomaly', palette='coolwarm')
        st.pyplot(plt)
