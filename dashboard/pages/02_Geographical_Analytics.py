import streamlit as st
import plotly.express as px
from dashboard.common import *

setup_page("📍 Geographical Analytics", "Complaint distribution and exploratory location clustering")

try:
    df = get_data()
except Exception as e:
    st.error(f"Dataset loading failed: {e}")
    st.stop()

geo = df.dropna(subset=["latitude", "longitude"]).copy()
geo = geo[
    geo["latitude"].between(40.4, 41.0) &
    geo["longitude"].between(-74.3, -73.6)
]

st.metric("Records with usable NYC coordinates", f"{len(geo):,}")
st.caption("The coordinate bounds are a rough exploratory filter, not an official NYC boundary.")

if geo.empty:
    st.warning("No valid geographic coordinates were found.")
    st.stop()

st.subheader("Complaint Hotspot Map")
sample_size = min(5000, len(geo))
sample = geo.sample(sample_size, random_state=42) if len(geo) > sample_size else geo

fig = px.scatter_mapbox(
    sample, lat="latitude", lon="longitude",
    color="complaint_type",
    hover_name="complaint_type",
    hover_data=["borough", "agency_name"],
    zoom=9, height=650,
    mapbox_style="open-street-map",
    title="Sampled Complaint Locations"
)
fig.update_layout(template="plotly_dark", margin={"r":0,"t":40,"l":0,"b":0})
st.plotly_chart(fig, use_container_width=True)

st.subheader("Borough Distribution")
borough = geo["borough"].fillna("Unspecified").value_counts().reset_index()
borough.columns = ["Borough", "Complaints"]
fig = px.bar(borough, x="Borough", y="Complaints", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Complaint Coordinates")
st.dataframe(sample[["complaint_type","borough","latitude","longitude"]].head(100),
             use_container_width=True)
