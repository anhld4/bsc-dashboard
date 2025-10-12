import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone
import numpy as np

def human_format(num):
    for unit in ['', 'K', 'M', 'B', 'T']:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}P"

st.set_page_config(page_title="BSC Dashboard", layout="wide")

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

    resp = requests.get(url, params=params, timeout=90) # Thoi gian timeout = 90 giay
    resp.raise_for_status()
    return resp.json()

# --------------------------
# Sidebar: config
# --------------------------
st.sidebar.title("⚙️ Config")
api_base = st.sidebar.text_input("API base URL", value="http://51.79.251.8:6969")
default_token = st.sidebar.text_input("Default token address (optional)", value="")
time_range = st.sidebar.number_input("Time range (minutes)", min_value=1, value=4320, step=10)
range_min = st.sidebar.number_input("Bucket size (minutes)", min_value=1, value=90, step=1)

# --------------------------
# Main UI
# --------------------------
st.title("📊 BSC Dashboard")
st.markdown("Nhập token address và nhấn `Analyze`.")

col1, col2, col3 = st.columns([3, 1, 1])

# Khởi tạo state cho input nếu chưa có
if "token_input" not in st.session_state:
    st.session_state.token_input = default_token

col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    token = st.text_input("Token address", key="token_input")

with col2:
    analyze_btn = st.button("🔍 Analyze", type="primary")

with col3:
    clear_btn = st.button("❌ Clear", on_click=lambda: st.session_state.update(token_input=""))

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

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Total volume (before)", f"{human_format(data.get('totalVolumeBefore'))}")
        colB.metric("Total volume (after wash filter)", f"{human_format(data.get('totalVolumeAfter'))}")
        colC.metric("Unique makers (before)", data.get("uniqueMakersBefore"))
        colD.metric("Unique makers (after)", data.get("uniqueMakersAfter"))

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

            st.subheader("📉 Z-score of Total Volume")
            fig2 = px.line(vol_df, x="start", y="zscore", markers=True)
            # highlight threshold
            max_z = data.get("maxZscore", 0)
            min_z = data.get("minZscore", 0)
            fig2.add_hline(y=max_z, line_dash="dash", line_color="red", annotation_text=f"z={max_z:.2f}", annotation_position="top left")
            fig2.add_hline(y=min_z, line_dash="dash", line_color="red", annotation_text=f"z={min_z:.2f}", annotation_position="bottom left")
            st.plotly_chart(fig2, use_container_width=True)

        else:
            st.info("No volume bucket data")

        # ---------- Netflow buckets & chart ----------
        netflowBuckets = data.get("netflowBuckets", [])
        if netflowBuckets:
            nf_df = pd.DataFrame(netflowBuckets)
            nf_df["start"] = pd.to_datetime(nf_df["start"])
            nf_df["end"] = pd.to_datetime(nf_df["end"])
            nf_df = nf_df.sort_values("start")

            st.subheader("🔄 Netflow (Buy - Sell per bucket)")
            fig3 = px.bar(
                nf_df,
                x="start",
                y="netflow",
                labels={"start": "Start (UTC)", "netflow": "Netflow"},
                color=nf_df["netflow"].apply(lambda x: "Positive" if x >= 0 else "Negative"),
                color_discrete_map={"Positive": "green", "Negative": "red"}
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No netflow data")

        # Lấy danh sách netflow từ buckets
        netflows = [b["netflow"] for b in netflowBuckets]
        # Đếm số bucket có netflow dương
        count_positive = sum(1 for x in netflows if x > 0)

        # Tổng số bucket
        count_total = len(netflows)

        # Tránh chia 0
        if count_total > 0:
            ratio = (count_positive / count_total) * 100
        else:
            ratio = 0

        st.metric("Positive Netflow Buckets (%)", f"{ratio:.2f}%")

        # ---------- Z-score Netflow ----------
        if netflowBuckets:
            nf_df = pd.DataFrame(netflowBuckets)
            nf_df["start"] = pd.to_datetime(nf_df["start"])
            nf_df = nf_df.sort_values("start")

            st.subheader("📊 Z-score of Netflow")
            fig4 = px.line(
                nf_df,
                x="start",
                y="zscore",
                markers=True,
                labels={"start": "Start (UTC)", "zscore": "Z-score (Netflow)"}
            )
            # highlight min/max zscore
            min_nf_z = data.get("minNetflowZ", 0)
            max_nf_z = data.get("maxNetflowZ", 0)
            if min_nf_z is not None:
                fig4.add_hline(
                    y=min_nf_z,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"min z={min_nf_z:.2f}",
                    annotation_position="bottom left"
                )
            if max_nf_z is not None:
                fig4.add_hline(
                    y=max_nf_z,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"max z={max_nf_z:.2f}",
                    annotation_position="top left"
                )

            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No netflow zscore data")

        # ---------- Footer stats ----------
        st.markdown("---")
        # st.write(f"minZscore: {data.get('minZscore')}, maxZscore: {data.get('maxZscore')}")
