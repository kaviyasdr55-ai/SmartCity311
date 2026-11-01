import streamlit as st
import plotly.express as px
from dashboard.common import *

setup_page("🏙️ SmartCity311 – Overview", "City Complaint Analytics & Predictive Service Intelligence")

try:
    df = get_data()
except Exception as e:
    st.error(f"Dataset loading failed: {e}")
    st.stop()

total = len(df)
closed = int((df["resolution_status"] == "Closed").sum())
opened = int((df["resolution_status"] == "Open").sum())
rate = closed / total * 100 if total else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Complaints", f"{total:,}")
c2.metric("Closed Complaints", f"{closed:,}")
c3.metric("Open Complaints", f"{opened:,}")
c4.metric("Closure Rate", f"{rate:.2f}%")

st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Complaint Status")
    status = df["resolution_status"].value_counts().rename_axis("Status").reset_index(name="Count")
    fig = px.pie(status, names="Status", values="Count", hole=.5)
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Top 10 Complaint Categories")
    top = df["complaint_type"].value_counts().head(10).sort_values().reset_index()
    top.columns = ["Complaint Type", "Count"]
    fig = px.bar(top, x="Count", y="Complaint Type", orientation="h",
                 template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Borough-wise Complaints")
borough = df["borough"].fillna("Unspecified").value_counts().reset_index()
borough.columns = ["Borough", "Complaints"]
fig = px.bar(borough, x="Borough", y="Complaints", color="Complaints",
             template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)
