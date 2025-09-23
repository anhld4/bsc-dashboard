import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone

def human_format(num):
    for unit in ['', 'K', 'M', 'B', 'T']:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}P"

st.set_page_config(page_title="Token Concentration Dashboard", layout="wide")

# --------------------------
# Helpers
# --------------------------
def format_pct(x):
    try:
        return f"{x*100:.2f}%"
    except:
        return x

@st.cache_data(ttl=60)  # cache 60s, tránh gọi API liên tục
def call_api(api_base, token, timeRange=None, range_min=None):
    """
    Gọi API GET /analyze/:address hoặc POST tuỳ endpoint của bạn.
    Trả về dict JSON hoặc raise exception.
    """
    # chuẩn hóa URL: nếu api_base ends with /, remove
    base = api_base.rstrip("/")
    url = f"{base.rstrip('/')}/concentration/{token}"

    params = {}
    if timeRange:
        params["timeRange"] = int(timeRange)
    if range_min:
        params["range"] = int(range_min)

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

# --------------------------
# Sidebar: config
# --------------------------
st.sidebar.title("⚙️ Config")
api_base = st.sidebar.text_input("API base URL", value="http://localhost:6969")
default_token = st.sidebar.text_input("Default token address (optional)", value="")
time_range = st.sidebar.number_input("Time range (minutes)", min_value=1, value=360, step=10)
range_min = st.sidebar.number_input("Bucket size (minutes)", min_value=1, value=10, step=1)

# --------------------------
# Main UI
# --------------------------
st.title("📊 Token Concentration & Wash Trading Dashboard")
st.markdown("Nhập token address và nhấn `Analyze` — ứng dụng sẽ gọi API và hiển thị kết quả trực quan.")

col1, col2 = st.columns([3, 1])

with col1:
    token = st.text_input("Token address", value=default_token)
with col2:
    analyze_btn = st.button("🔍 Analyze", type="primary")

# Optional quick examples
st.markdown("**Example:** `0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c` (WBNB)")

# --------------------------
# When press analyze
# --------------------------
if analyze_btn:
    if not token:
        st.error("Vui lòng nhập token address")
    else:
        with st.spinner("Đang gọi API và phân tích..."):
            try:
                data = call_api(api_base, token, timeRange=time_range, range_min=range_min)
            except Exception as e:
                st.exception(f"API call failed: {e}")
                st.stop()

        # ---------- Summary / Risk ----------
        st.subheader("📌 Summary")
        risk = data.get("riskLevel")
        # risk may be string or object {riskLevel, riskReason}
        if isinstance(risk, dict):
            level = risk.get("riskLevel", "Unknown")
            reasons = risk.get("riskReason", [])
        else:
            level = risk or "Unknown"
            reasons = []

        colA, colB, colC = st.columns(3)
        colA.metric("Total volume (before)", f"{human_format(data.get('totalVolumeBefore'))}")
        colB.metric("Total volume (after wash filter)", f"{human_format(data.get('totalVolumeAfter'))}")
        colC.metric("Unique makers (after)", data.get("uniqueMakersAfter"))

        # Risk badge + reasons
        if level.lower() == "high":
            st.error(f"⚠️ Risk Level: {level}")
        elif level.lower() == "medium":
            st.warning(f"🟠 Risk Level: {level}")
        else:
            st.success(f"🟢 Risk Level: {level}")

        if reasons:
            st.markdown("**Reasons:**")
            for r in reasons:
                st.write(f"- {r}")

        # ---------- Top Makers ----------
        st.subheader("🏦 Top Makers (after wash filtering)")
        top_df = pd.DataFrame(data.get("topMakers", []))
        if not top_df.empty:
            top_df["share_pct"] = top_df["share"].apply(lambda x: f"{x*100:.2f}%")
            st.dataframe(top_df[["maker", "volume", "share_pct"]], use_container_width=True)
        else:
            st.write("No top makers data")

        # ---------- Wash makers ----------
        st.subheader("🚨 Wash trading suspects")
        wash_df = pd.DataFrame(data.get("washMakers", []))
        if not wash_df.empty:
            wash_df = wash_df.assign(washScore=lambda df: df["washScore"].round(4))
            st.dataframe(wash_df[["maker", "buy", "sell", "total", "washScore"]], use_container_width=True)
        else:
            st.success("No wash trading detected")

        # ---------- Volume buckets & charts ----------
        buckets = data.get("volumeBuckets", [])
        if buckets:
            vol_df = pd.DataFrame(buckets)
            vol_df["start"] = pd.to_datetime(vol_df["start"])
            vol_df["end"] = pd.to_datetime(vol_df["end"])
            vol_df = vol_df.sort_values("start")

            st.subheader("📈 Volume (buckets)")
            fig = px.bar(vol_df, x="start", y="volume", hover_data=["zscore"], labels={"start":"Start (UTC)"})
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📉 Z-score across buckets")
            fig2 = px.line(vol_df, x="start", y="zscore", markers=True)
            # highlight threshold
            fig2.add_hline(y=3, line_dash="dash", line_color="red", annotation_text="z=3", annotation_position="top left")
            fig2.add_hline(y=-3, line_dash="dash", line_color="red", annotation_text="z=-3", annotation_position="bottom left")
            st.plotly_chart(fig2, use_container_width=True)

            # Show anomalous buckets
            anomalies = vol_df[vol_df["zscore"].abs() > 3].copy()
            if not anomalies.empty:
                st.subheader("⚠️ Anomalous buckets (|z| > 3)")
                anomalies["zscore"] = anomalies["zscore"].round(3)
                anomalies["start"] = anomalies["start"].dt.strftime("%Y-%m-%d %H:%M:%S")
                anomalies["end"] = anomalies["end"].dt.strftime("%Y-%m-%d %H:%M:%S")
                st.dataframe(anomalies[["start","end","volume","zscore"]], use_container_width=True)
            else:
                st.info("No extreme anomalies (|z|>3) detected")

            # Allow download CSV
            csv = vol_df.to_csv(index=False)
            st.download_button("⬇️ Download bucket CSV", csv, file_name=f"{token}_buckets.csv", mime="text/csv")
        else:
            st.info("No volume bucket data")

        # ---------- Footer stats ----------
        st.markdown("---")
        st.write(f"minZscore: {data.get('minZscore')}, maxZscore: {data.get('maxZscore')}")
