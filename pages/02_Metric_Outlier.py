import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go

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

# --------------------------
# Streamlit UI
# --------------------------
st.title("📊 Metric Outlier Detection")

# Nhập địa chỉ
address = st.text_input("Enter address for Metric Outlier Detection:", key="address_input")

# Hàm callback để xóa input
def clear_input():
    st.session_state.address_input = ""

# Hai nút: Submit và Clear
col1, col2 = st.columns(2)
submit = col1.button("🚀 Submit")
clear = col2.button("🧹 Clear", on_click=clear_input)

# Nếu nhấn Clear thì reset
if clear:
    st.session_state.clear()

# Chỉ chạy khi nhấn Submit
if submit and address:
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

        numeric_cols = [
        'total_maker', 'total_volume', 'maker_buy', 'total_buy',
        'maker_sell', 'total_sell', 'total_supply', 'total_transfer',
        'total_transfer_amount'
        ]

        # Chuyển sang float
        for col in numeric_cols:
            df[col] = df[col].astype(float)

        # --------------------------
        # 🧮 Tính Z-score cho từng feature
        # --------------------------
        for col in numeric_cols:
            mean_val = df[col].mean()
            std_val = df[col].std(ddof=0)
            if std_val == 0:
                df[f"{col}_zscore"] = 0
            else:
                df[f"{col}_zscore"] = (df[col] - mean_val) / std_val

        # --------------------------
        # 🤖 Isolation Forest dùng Z-score
        # --------------------------
        zscore_cols = [f"{col}_zscore" for col in numeric_cols]
        X = df[zscore_cols].fillna(0)

        model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
        model.fit(X)

        scores = model.decision_function(X)
        threshold = np.percentile(scores, 1)
        df['anomaly'] = (scores <= threshold).astype(int)

        st.subheader(f"⚠️ Found {df['anomaly'].sum()} outliers")

        # --------------------------
        # Giải thích tại sao là outlier (hiển thị created_date)
        # --------------------------
        # Chuyển created_date về datetime trước khi lặp
        df['created_date'] = pd.to_datetime(df['created_date'])

        outliers = df[df['anomaly'] == 1]
        if not outliers.empty:
            st.markdown("### 💡 Why these points are outliers:")
            feature_means = df[numeric_cols].mean()
            explanations = []

            for idx, row in outliers.iterrows():
                diffs = []
                z_threshold = 2.5  # ngưỡng |Z| > 2.5 coi là bất thường
                for col in numeric_cols:
                    z_col = f"{col}_zscore"
                    z_val = row[z_col]
                    if abs(z_val) > z_threshold:
                        diffs.append(f"{col} (Z={z_val:.2f})")

                if diffs:
                    created_str = pd.to_datetime(row['created_date']).strftime("%Y-%m-%dT%H:%M")
                    explanations.append(f"- **{created_str}** : " + ", ".join(diffs))

            if explanations:
                st.markdown("\n".join(explanations))
            else:
                st.info("No strong feature deviations detected among outliers.")
        else:
            st.success("✅ No outliers detected for this address.")

        # --------------------------
        # Biểu đồ line các feature + highlight outlier
        # --------------------------
        st.markdown("### 📈 Feature trends with outliers highlighted")
        for col in numeric_cols:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines+markers',
                name='Normal',
                marker=dict(color='blue'),
                text=[f"{col}: {human_format(v)}" for v in df[col]],
                hoverinfo='text'
            ))
            fig.add_trace(go.Scatter(
                x=df[df['anomaly'] == 1].index,
                y=df[df['anomaly'] == 1][col],
                mode='markers',
                name='Outlier',
                marker=dict(color='red', size=10, symbol='x'),
                text=[f"OUTLIER {col}: {human_format(v)}" for v in df[df['anomaly'] == 1][col]],
                hoverinfo='text'
            ))
            fig.update_layout(
                title=f"{col} over time",
                xaxis_title="Index (record order)",
                yaxis_title=col,
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

# Khi chưa nhấn submit
elif not submit and not clear:
    st.info("👆 Enter an address and press **Submit** to start analysis.")
