import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime

st.set_page_config(
    page_title="SmartCity311 | Intelligence Dashboard",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = os.getcwd()

DATA_PATH = os.path.join(BASE, "data", "processed", "nyc_311_cleaned")
MODEL_DIR = os.path.join(BASE, "models")
REPORT_DIR = os.path.join(BASE, "reports")

# -------------------------------
# CUSTOM DESIGN
# -------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background-color: #111827;
}
.metric-card {
    background: #172235;
    border: 1px solid #26364d;
    border-radius: 12px;
    padding: 18px;
}
h1, h2, h3 {
    color: #f8fafc !important;
}
p, label {
    color: #cbd5e1;
}
div[data-testid="stMetric"] {
    background: #172235;
    border: 1px solid #26364d;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------
# DATA LOADING
# -------------------------------

@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)

@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load cleaned dataset: {e}")
    st.info("Check that data/processed/nyc_311_cleaned exists.")
    st.stop()

resolution_model = load_model(
    os.path.join(MODEL_DIR, "best_resolution_model.joblib")
)

delay_model = load_model(
    os.path.join(MODEL_DIR, "best_delay_model.joblib")
)

top_complaints = load_csv(
    os.path.join(REPORT_DIR, "analytics", "top_complaints.csv")
)
resolution_report = load_csv(
    os.path.join(REPORT_DIR, "analytics", "resolution_time_by_type.csv")
)
borough_report = load_csv(
    os.path.join(REPORT_DIR, "analytics", "borough_analysis.csv")
)
department_report = load_csv(
    os.path.join(REPORT_DIR, "department", "department_workload.csv")
)
department_performance = load_csv(
    os.path.join(REPORT_DIR, "department", "department_resolution_performance.csv")
)
cluster_report = load_csv(
    os.path.join(REPORT_DIR, "geographical", "cluster_summary.csv")
)
resolution_comparison = load_csv(
    os.path.join(REPORT_DIR, "ml", "model_comparison.csv")
)
delay_comparison = load_csv(
    os.path.join(REPORT_DIR, "ml", "delay_model_comparison.csv")
)

# -------------------------------
# HEADER
# -------------------------------

st.title("🏙️ SmartCity311")
st.markdown("### Complaint Analytics & Predictive Service Intelligence")
st.caption("NYC 311 Open Data | Big Data Analytics | Machine Learning")

st.sidebar.title("SMARTCITY311")
st.sidebar.caption("AI-powered city complaint intelligence")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Geographical Analytics",
        "Department Performance",
        "AI Resolution Prediction",
        "AI Delay Detection",
        "Model Evaluation"
    ]
)

st.sidebar.divider()
st.sidebar.write("Dataset records:", f"{len(df):,}")
st.sidebar.write("Resolution model:", "Available" if resolution_model else "Missing")
st.sidebar.write("Delay model:", "Available" if delay_model else "Missing")
st.sidebar.caption("Academic project demonstration")


# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

def safe_col(frame, col):
    return col in frame.columns

def chart_theme(fig, height=380):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="#0b1120",
        plot_bgcolor="#111827",
        font_color="#e5e7eb",
        margin=dict(l=20, r=20, t=45, b=20)
    )
    return fig

def metric(label, value, help_text=None):
    st.metric(label, value, help=help_text)

def make_input():
    defaults = {
        "agency": "NYPD",
        "agency_name": "New York City Police Department",
        "complaint_type": "Illegal Parking",
        "borough": "BROOKLYN",
        "complaint_hour": 12,
        "complaint_weekday": 2,
        "complaint_month": 10
    }

    input_values = {}

    categorical = [
        ("agency", "Agency"),
        ("agency_name", "Department"),
        ("complaint_type", "Complaint Type"),
        ("borough", "Borough")
    ]

    for key, label in categorical:
        if key in df.columns:
            choices = sorted(
                df[key].dropna().astype(str).unique().tolist()
            )
        else:
            choices = [defaults[key]]

        if not choices:
            choices = [defaults[key]]

        default = defaults[key]
        index = choices.index(default) if default in choices else 0

        input_values[key] = st.selectbox(
            label, choices, index=index, key="input_" + key
        )

    input_values["complaint_hour"] = st.slider(
        "Complaint Hour", 0, 23, 12
    )
    input_values["complaint_weekday"] = st.selectbox(
        "Weekday (0 = Monday)", list(range(7)), index=2
    )
    input_values["complaint_month"] = st.selectbox(
        "Month", list(range(1, 13)), index=9
    )

    return pd.DataFrame([input_values])


# -------------------------------
# PAGE 1: OVERVIEW
# -------------------------------

if page == "Overview":

    st.subheader("City Complaint Overview")

    total = len(df)
    open_count = (
        (df["resolution_status"] == "Open").sum()
        if "resolution_status" in df.columns else 0
    )
    closed_count = (
        (df["resolution_status"] == "Closed").sum()
        if "resolution_status" in df.columns else 0
    )

    closure_rate = closed_count / total * 100 if total else 0

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric("Total Complaints", f"{total:,}")
    with c2:
        metric("Closed Complaints", f"{closed_count:,}")
    with c3:
        metric("Open Complaints", f"{open_count:,}")
    with c4:
        metric("Closure Rate", f"{closure_rate:.2f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown("#### Complaint Status")

        status_data = pd.DataFrame({
            "Status": ["Closed", "Open"],
            "Count": [closed_count, open_count]
        })

        fig = px.pie(
            status_data,
            names="Status",
            values="Count",
            hole=0.55
        )
        st.plotly_chart(chart_theme(fig), use_container_width=True)

    with right:
        st.markdown("#### Top Complaint Categories")

        if not top_complaints.empty:
            xcol = "complaint_type"
            ycol = "total_complaints"

            if xcol in top_complaints.columns and ycol in top_complaints.columns:
                data = top_complaints.head(10).sort_values(ycol)
                fig = px.bar(
                    data,
                    x=ycol,
                    y=xcol,
                    orientation="h",
                    labels={
                        xcol: "Complaint Type",
                        ycol: "Number of Complaints"
                    }
                )
                st.plotly_chart(chart_theme(fig), use_container_width=True)
            else:
                st.dataframe(top_complaints, use_container_width=True)
        else:
            st.info("Top complaints report is unavailable.")

    st.markdown("#### Borough-wise Complaints")

    if not borough_report.empty:
        st.dataframe(borough_report, use_container_width=True)
    elif safe_col(df, "borough"):
        borough_data = (
            df.groupby("borough")
            .size()
            .reset_index(name="complaints")
            .sort_values("complaints", ascending=False)
        )
        fig = px.bar(borough_data, x="borough", y="complaints")
        st.plotly_chart(chart_theme(fig), use_container_width=True)

    st.caption("Data source: NYC 311 Open Data. This dashboard analyzes the collected sample.")


# -------------------------------
# PAGE 2: GEOGRAPHICAL ANALYTICS
# -------------------------------

elif page == "Geographical Analytics":

    st.subheader("Geographical Complaint Intelligence")

    if "latitude" not in df.columns or "longitude" not in df.columns:
        st.warning("Coordinate columns are unavailable.")
    else:
        geo = df.dropna(subset=["latitude", "longitude"]).copy()

        geo = geo[
            geo["latitude"].between(40.4, 41.0)
            & geo["longitude"].between(-74.3, -73.6)
        ]

        st.write(f"Complaints with usable NYC coordinates: {len(geo):,}")

        if not geo.empty:
            sample = geo.sample(
                min(4000, len(geo)),
                random_state=42
            )

            color_col = (
                "complaint_type"
                if "complaint_type" in sample.columns else None
            )

            fig = px.scatter_mapbox(
                sample,
                lat="latitude",
                lon="longitude",
                color=color_col,
                hover_name="complaint_type" if color_col else None,
                zoom=9,
                height=600,
                mapbox_style="open-street-map"
            )

            fig.update_layout(
                mapbox_style="open-street-map",
                margin=dict(l=0, r=0, t=0, b=0)
            )

            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Exploratory K-Means Clusters")

    if not cluster_report.empty:
        st.dataframe(cluster_report, use_container_width=True)

        if {
            "center_latitude", "center_longitude", "total_complaints"
        }.issubset(cluster_report.columns):
            fig = px.scatter(
                cluster_report,
                x="center_longitude",
                y="center_latitude",
                size="total_complaints",
                color="cluster_id",
                hover_data=["total_complaints"],
                title="Cluster Centres"
            )
            st.plotly_chart(chart_theme(fig), use_container_width=True)
    else:
        st.info("Cluster report not available.")

    st.caption(
        "Clusters are exploratory complaint-density groups, "
        "not official risk or severity zones."
    )


# -------------------------------
# PAGE 3: DEPARTMENT PERFORMANCE
# -------------------------------

elif page == "Department Performance":

    st.subheader("Department Workload & Performance")

    if not department_report.empty:

        st.markdown("#### Department Workload")

        st.dataframe(
            department_report,
            use_container_width=True,
            hide_index=True
        )

        if {
            "agency_name", "total_complaints", "open_complaints"
        }.issubset(department_report.columns):

            fig = px.bar(
                department_report.sort_values(
                    "total_complaints", ascending=False
                ).head(12),
                x="agency_name",
                y=["open_complaints", "closed_complaints"],
                barmode="group",
                title="Open vs Closed Complaints by Department"
            )
            st.plotly_chart(chart_theme(fig, 450), use_container_width=True)

    else:
        st.warning("Department workload report is unavailable.")

    st.divider()
    st.markdown("#### Resolution Time Performance")

    if not department_performance.empty:
        st.dataframe(
            department_performance,
            use_container_width=True,
            hide_index=True
        )

        if {
            "agency_name", "average_resolution_hours"
        }.issubset(department_performance.columns):

            fig = px.bar(
                department_performance.sort_values(
                    "average_resolution_hours", ascending=False
                ),
                x="agency_name",
                y="average_resolution_hours",
                title="Average Resolution Time by Department"
            )
            st.plotly_chart(chart_theme(fig, 450), use_container_width=True)

    else:
        st.info("Department resolution performance report unavailable.")

    st.warning(
        "An open complaint is not automatically overdue. "
        "Official department service-level deadlines are not included."
    )


# -------------------------------
# PAGE 4: RESOLUTION PREDICTION
# -------------------------------

elif page == "AI Resolution Prediction":

    st.subheader("AI-Based Resolution Time Prediction")

    st.write(
        "Enter complaint details to estimate the expected resolution duration."
    )

    if resolution_model is None:
        st.error("Resolution model file not found.")
    else:
        with st.form("resolution_form"):
            input_df = make_input()
            submitted = st.form_submit_button(
                "Predict Resolution Time",
                use_container_width=True
            )

        if submitted:
            try:
                prediction = float(resolution_model.predict(input_df)[0])

                st.divider()
                st.success("Prediction completed!")

                a, b = st.columns(2)

                with a:
                    st.metric(
                        "Estimated Resolution",
                        f"{max(0, prediction):.2f} hours"
                    )

                with b:
                    st.metric(
                        "Estimated Days",
                        f"{max(0, prediction)/24:.2f} days"
                    )

                st.caption(
                    "This is a model estimate, not a guaranteed completion time."
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")


# -------------------------------
# PAGE 5: DELAY DETECTION
# -------------------------------

elif page == "AI Delay Detection":

    st.subheader("AI-Based Complaint Delay Detection")

    st.write(
        "Predict whether a complaint may take more than 24 hours "
        "to resolve using the trained classification model."
    )

    if delay_model is None:
        st.error("Delay classification model file not found.")
    else:
        with st.form("delay_form"):
            input_df = make_input()
            submitted = st.form_submit_button(
                "Detect Potential Delay",
                use_container_width=True
            )

        if submitted:
            try:
                prediction = int(delay_model.predict(input_df)[0])
                probabilities = delay_model.predict_proba(input_df)[0]

                delayed_probability = float(probabilities[1]) * 100

                st.divider()

                left, right = st.columns(2)

                with left:
                    if prediction == 1:
                        st.error("Potential Delay Detected")
                    else:
                        st.success("Not Classified as Delayed")

                with right:
                    st.metric(
                        "Model Delay Probability",
                        f"{delayed_probability:.2f}%"
                    )

                st.progress(
                    min(max(delayed_probability / 100, 0.0), 1.0)
                )

                st.caption(
                    "Project-defined classification threshold: "
                    "resolution duration greater than 24 hours. "
                    "This is not an official NYC SLA."
                )

            except Exception as e:
                st.error(f"Delay prediction failed: {e}")


# -------------------------------
# PAGE 6: MODEL EVALUATION
# -------------------------------

elif page == "Model Evaluation":

    st.subheader("Machine Learning Model Evaluation")

    st.markdown("#### Resolution Time Prediction")

    if not resolution_comparison.empty:
        st.dataframe(
            resolution_comparison,
            use_container_width=True,
            hide_index=True
        )

        if "R2" in resolution_comparison.columns:
            fig = px.bar(
                resolution_comparison,
                x="model",
                y="R2",
                title="Resolution Prediction - R² Comparison"
            )
            st.plotly_chart(chart_theme(fig), use_container_width=True)
    else:
        st.info("Resolution model report unavailable.")

    st.divider()

    st.markdown("#### Delay Classification")

    if not delay_comparison.empty:
        st.dataframe(
            delay_comparison,
            use_container_width=True,
            hide_index=True
        )

        if "f1_score" in delay_comparison.columns:
            fig = px.bar(
                delay_comparison,
                x="model",
                y=["accuracy", "precision", "recall", "f1_score"],
                barmode="group",
                title="Delay Classification - Model Comparison"
            )
            st.plotly_chart(chart_theme(fig), use_container_width=True)
    else:
        st.info("Delay model report unavailable.")

    st.warning(
        "Evaluation uses a random train/test split from the collected sample. "
        "Temporal and external-city validation are still required."
    )


# -------------------------------
# FOOTER
# -------------------------------

st.divider()
st.caption(
    "SmartCity311 | Big Data Analytics Academic Project | "
    "NYC 311 Open Data | Machine Learning Demonstration"
)
