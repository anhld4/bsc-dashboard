# pages/02_Metric_Outlier.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

st.title("📊 Metric Outlier Detection with Explanation")

address = st.text_input("Enter address for Metric Outlier Detection:")

if address:
    try:
        response = requests.get(f"http://51.79.251.8:6969/metrics/{address}")
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

        # -----------------------
        # Isolation Forest
        # -----------------------
        X = df[numeric_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
        model.fit(X_scaled)

        scores = model.decision_function(X_scaled)
        threshold = np.percentile(scores, 5)
        df['anomaly'] = (scores <= threshold).astype(int)
        df['score'] = scores

        # -----------------------
        # Explain why outlier
        # -----------------------
        mean = X.mean(axis=0)
        std = X.std(axis=0)

        def explain_outlier(row):
            reasons = []
            for i, col in enumerate(numeric_cols):
                if row[col] > mean[i] + 2*std[i]:
                    reasons.append(f"{col} high ({row[col]:.2f})")
                elif row[col] < mean[i] - 2*std[i]:
                    reasons.append(f"{col} low ({row[col]:.2f})")
            return ", ".join(reasons) if reasons else "N/A"

        df['reason'] = df.apply(lambda row: explain_outlier(row) if row['anomaly']==1 else "", axis=1)

        st.subheader(f"Found {df['anomaly'].sum()} outliers")
        st.dataframe(df[['id', 'anomaly', 'score', 'reason'] + numeric_cols])

        # -----------------------
        # Feature line plots with outlier highlight
        # -----------------------
        st.subheader("Feature Trends with Outliers Highlighted")
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(10,3))
            ax.plot(df.index, df[col], label=col, marker='o')
            # Highlight outliers
            outlier_idx = df.index[df['anomaly']==1]
            ax.scatter(outlier_idx, df.loc[outlier_idx, col], color='red', label='Outlier', zorder=5)
            ax.set_title(col)
            ax.set_xlabel("Record Index")
            ax.set_ylabel(col)
            ax.legend()
            st.pyplot(fig)
