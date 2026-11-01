import streamlit as st
import pandas as pd
from dashboard.common import *

setup_page("🤖 AI Prediction", "Predict estimated resolution duration and identify potential long-resolution cases")

try:
    df = get_data()
except Exception as e:
    st.error(f"Dataset loading failed: {e}")
    st.stop()

resolution_model = get_model("best_resolution_model.joblib")
delay_model = get_model("best_delay_model.joblib")

if resolution_model is None:
    st.error("Resolution prediction model not found. Check models/best_resolution_model.joblib.")
if delay_model is None:
    st.error("Delay classification model not found. Check models/best_delay_model.joblib.")

agency_names = sorted(df["agency_name"].dropna().unique().tolist())
agency_codes = sorted(df["agency"].dropna().unique().tolist())
complaints = sorted(df["complaint_type"].dropna().unique().tolist())
boroughs = sorted(df["borough"].dropna().unique().tolist())

st.subheader("Enter Complaint Details")

with st.form("prediction_form"):
    a, b = st.columns(2)
    with a:
        agency_name = st.selectbox("Department", agency_names)
        complaint_type = st.selectbox("Complaint Type", complaints)
        borough = st.selectbox("Borough", boroughs)
    with b:
        hour = st.slider("Complaint Hour", 0, 23, 12)
        weekday = st.selectbox("Weekday (0 = Monday)", list(range(7)), index=2)
        month = st.selectbox("Month", list(range(1, 13)), index=10)
    submitted = st.form_submit_button("Run AI Prediction", use_container_width=True)

if submitted:
    agency_match = df.loc[df["agency_name"] == agency_name, "agency"].dropna()
    agency = agency_match.mode().iloc[0] if not agency_match.empty else (agency_codes[0] if agency_codes else "")

    st.divider()
    if resolution_model is not None:
        features = get_model_features("model_features.json")
        X = prediction_row(features, agency, agency_name, complaint_type,
                           borough, hour, weekday, month)
        try:
            prediction = float(resolution_model.predict(X)[0])
            st.metric("Estimated Resolution Time", f"{max(0, prediction):.2f} hours")
            st.caption("This is a model estimate based on historical data, not a guaranteed completion time.")
        except Exception as e:
            st.error(f"Resolution prediction failed: {e}")
            st.write("Input columns sent to model:", list(X.columns))

    if delay_model is not None:
        features = get_model_features("delay_model_features.json")
        X_delay = prediction_row(features, agency, agency_name, complaint_type,
                                 borough, hour, weekday, month)
        try:
            prediction = int(delay_model.predict(X_delay)[0])
            probability = None
            if hasattr(delay_model, "predict_proba"):
                probability = float(delay_model.predict_proba(X_delay)[0][1])
            if prediction == 1:
                st.warning("Model classification: Potential long-resolution case")
            else:
                st.success("Model classification: Not classified as a long-resolution case")
            if probability is not None:
                st.metric("Predicted Long-Resolution Probability", f"{probability*100:.2f}%")
            st.caption("Project-defined classification: resolution time greater than 24 hours. This is not an official NYC SLA.")
        except Exception as e:
            st.error(f"Delay classification failed: {e}")
            st.write("Input columns sent to model:", list(X_delay.columns))

st.divider()
st.subheader("Model Information")
st.write("Resolution model: Random Forest Regressor (selected in the earlier model comparison).")
st.write("Delay model: trained to classify closed complaints with resolution duration above the project's 24-hour threshold.")
