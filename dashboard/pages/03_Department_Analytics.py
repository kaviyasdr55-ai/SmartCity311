import streamlit as st
import plotly.express as px
from dashboard.common import *

setup_page("🏢 Department Analytics", "Department workload, closure rates and resolution performance")

try:
    df = get_data()
except Exception as e:
    st.error(f"Dataset loading failed: {e}")
    st.stop()

workload = department_workload(df)
performance = department_resolution(df)

st.subheader("Department Workload")
st.dataframe(workload, use_container_width=True, hide_index=True)

fig = px.bar(
    workload.head(12), x="total_complaints", y="agency_name",
    orientation="h", color="closure_percentage",
    labels={"total_complaints":"Total Complaints","agency_name":"Department"},
    template="plotly_dark", title="Top Departments by Complaint Volume"
)
fig.update_layout(yaxis={"categoryorder":"total ascending"})
st.plotly_chart(fig, use_container_width=True)

st.subheader("Department Closure Rate")
fig = px.bar(
    workload.sort_values("closure_percentage"),
    x="closure_percentage", y="agency_name", orientation="h",
    template="plotly_dark", title="Closure Percentage by Department"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Resolution Time Performance")
st.caption("Only departments with at least 30 closed records are shown. These are observed historical durations, not official service-level targets.")
st.dataframe(performance, use_container_width=True, hide_index=True)

if not performance.empty:
    fig = px.bar(
        performance, x="average_resolution_hours", y="agency_name",
        orientation="h", template="plotly_dark",
        title="Average Resolution Time (Hours)"
    )
    fig.update_layout(yaxis={"categoryorder":"total ascending"})
    st.plotly_chart(fig, use_container_width=True)

st.info("An open complaint is not automatically overdue. This dataset analysis does not establish official department SLA violations.")
