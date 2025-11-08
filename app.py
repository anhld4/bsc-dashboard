import streamlit as st

st.set_page_config(page_title="BSC Multi-Page Dashboard", layout="wide")
st.title("🏗️ BSC Dashboard - Multi Page App")

st.markdown("""
Use the sidebar to switch between pages:

- **BSC Dashboard** → Original dashboard
- **Metric Outlier Detection** → New page to detect outliers per address
""")
