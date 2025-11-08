# pages/02_Metric_Outlier.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------
# Helper: Format số đẹp
# --------------------------
def human_format(num):
    if num is None or pd.isna(num):
        return "-"
    try:
        num = float(num)
    except ValueError:
        return str(num)
    for unit in ['', 'K', 'M', 'B', 'T']:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}P"


st.title("📊 Metric Outlier Detection")

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

        # --------------------------
        # Chuẩn bị dữ liệu
        # --------------------------
        numeric_cols = [
            'total_maker', 'total_volume', 'maker_buy', 'total_buy',
            'maker_sell', 'total_sell', 'total_supply', 'total_transfer',
            'total_transfer_amount', 'total_mint', 'total_mint_amount',
            'total_burn', 'total_burn_amount'
        ]
        for col in numeric_cols:
            df[col] = df[col].astype(float)

        X = df[numeric_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
        model.fit(X_scaled)

        scores = model.decision_function(X_scaled)
        threshold = np.percentile(scores, 5)
        df['anomaly'] = (scores <= threshold).astype(int)

        st.subheader(f"⚠️ Found {df['anomaly'].sum()} outliers")

        # --------------------------
        # Giải thích tại sao là outlier
        # --------------------------
        outliers = df[df['anomaly'] == 1]
        if not outliers.empty:
            st.markdown("### 💡 Why these points are outliers:")
            feature_means = df[numeric_cols].mean()
            explanations = []
            for idx, row in outliers.iterrows():
                diffs = []
                for col in numeric_cols:
                    mean_val = feature_means[col]
                    val = row[col]
                    if mean_val == 0:
                        continue
                    diff_ratio = abs(val - mean_val) / abs(mean_val)
                    if diff_ratio > 1.0:  # lớn hơn 100% trung bình
                        diffs.append(f"{col} ({human_format(val)} vs avg {human_format(mean_val)})")
                if diffs:
                    explanations.append(f"- **ID {row['id']}** deviates strongly in: " + ", ".join(diffs))
            if explanations:
                st.markdown("\n".join(explanations))
            else:
                st.info("No strong feature deviations detected among outliers.")
        else:
            st.success("✅ No outliers detected for this address.")

        # --------------------------
        # Chọn feature để phân tích
        # --------------------------
        st.markdown("---")
        st.markdown("### 🎛️ Select feature to visualize")
        selected_feature = st.selectbox("Choose a feature", numeric_cols)

        if selected_feature:
            st.markdown(f"### 📈 {selected_feature} trend (highlight outliers)")
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=df.index,
                y=df[selected_feature],
                mode='lines+markers',
                name='Normal',
                marker=dict(color='blue'),
                text=[f"{selected_feature}: {human_format(v)}" for v in df[selected_feature]],
                hoverinfo='text'
            ))
            fig_line.add_trace(go.Scatter(
                x=df[df['anomaly'] == 1].index,
                y=df[df['anomaly'] == 1][selected_feature],
                mode='markers',
                name='Outlier',
                marker=dict(color='red', size=10, symbol='x'),
                text=[f"OUTLIER {selected_feature}: {human_format(v)}" for v in df[df['anomaly'] == 1][selected_feature]],
                hoverinfo='text'
            ))
            fig_line.update_layout(
                title=f"{selected_feature} over time",
                xaxis_title="Record index",
                yaxis_title=selected_feature,
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig_line, use_container_width=True)

            # --------------------------
            # Histogram phân phối
            # --------------------------
            st.markdown(f"### 📊 Distribution of {selected_feature}")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=df[selected_feature],
                nbinsx=30,
                name='All data',
                marker_color='blue',
                opacity=0.6
            ))
            fig_hist.add_trace(go.Histogram(
                x=df[df['anomaly'] == 1][selected_feature],
                nbinsx=30,
                name='Outliers',
                marker_color='red',
                opacity=0.8
            ))
            fig_hist.update_layout(
                barmode='overlay',
                xaxis_title=selected_feature,
                yaxis_title="Count",
                height=400
            )
            st.plotly_chart(fig_hist, use_container_width=True)
