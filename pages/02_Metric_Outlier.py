# pages/02_Metric_Outlier.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import seaborn as sns
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
        df['score'] = scores  # lưu score để tham khảo

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
        # Scatter plot
        # -----------------------
        st.subheader("Scatter plot with outliers highlighted")
        fig, ax = plt.subplots()
        ax.scatter(df['total_buy'], df['total_sell'], c=df['anomaly'], cmap='coolwarm', s=50)
        ax.set_xlabel('total_buy')
        ax.set_ylabel('total_sell')
        # Annotate outliers
        for i, row in df[df['anomaly']==1].iterrows():
            ax.annotate(row['id'][:6], (row['total_buy'], row['total_sell']), fontsize=8, color='red')
        st.pyplot(fig)

        # -----------------------
        # Boxplot for each numeric column
        # -----------------------
        st.subheader("Boxplot for each numeric column with outliers")
        for col in numeric_cols:
            fig, ax = plt.subplots()
            ax.boxplot(df[col], vert=True)
            # Highlight outliers
            outlier_values = df[df['anomaly']==1][col]
            ax.scatter(np.where(df['anomaly']==1)[0]+1, outlier_values, color='red', label='Outlier')
            ax.set_title(col)
            st.pyplot(fig)
