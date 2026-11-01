import os
import json
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "nyc_311_cleaned"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

@st.cache_data
def get_data():
    return pd.read_parquet(DATA_PATH)

@st.cache_data
def get_report(relative_path):
    path = REPORTS / relative_path
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_resource
def get_model(filename):
    path = MODELS / filename
    if path.exists():
        return joblib.load(path)
    return None

def setup_page(title, subtitle):
    st.set_page_config(page_title=title, page_icon="🏙️", layout="wide")
    st.markdown("""
    <style>
    .stApp {background:#0b1120;color:#e5e7eb;}
    [data-testid="stSidebar"] {background:#111827;}
    h1,h2,h3 {color:#f8fafc!important;}
    div[data-testid="stMetric"] {
        background:#172235;border:1px solid #26364d;
        padding:15px;border-radius:10px;
    }
    </style>
    """, unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)

def get_model_features(filename):
    path = MODELS / filename
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                for key in ("features", "feature_names", "all_features"):
                    if isinstance(obj.get(key), list):
                        return obj[key]
                result = []
                for key in ("categorical_features", "numeric_features"):
                    result.extend(obj.get(key, []))
                if result:
                    return result
        except Exception:
            pass
    return []

def department_workload(df):
    result = (
        df.groupby("agency_name", dropna=False)
        .agg(
            total_complaints=("unique_key", "count"),
            closed_complaints=("resolution_status", lambda x: (x == "Closed").sum()),
            open_complaints=("resolution_status", lambda x: (x == "Open").sum())
        )
        .reset_index()
    )
    result["closure_percentage"] = (
        result["closed_complaints"] / result["total_complaints"] * 100
    ).round(2)
    return result.sort_values("total_complaints", ascending=False)

def department_resolution(df):
    closed = df[
        (df["resolution_status"] == "Closed") &
        df["resolution_hours"].notna() &
        (df["resolution_hours"] >= 0)
    ]
    result = (
        closed.groupby("agency_name")
        .agg(
            closed_sample_count=("resolution_hours", "count"),
            average_resolution_hours=("resolution_hours", "mean"),
            median_resolution_hours=("resolution_hours", "median"),
            p90_resolution_hours=("resolution_hours", lambda x: x.quantile(.90))
        )
        .reset_index()
    )
    result = result[result["closed_sample_count"] >= 30]
    return result.sort_values("average_resolution_hours", ascending=False).round(2)

def prediction_row(features, agency, agency_name, complaint_type, borough,
                   hour, weekday, month):
    values = {
        "agency": agency,
        "agency_name": agency_name,
        "complaint_type": complaint_type,
        "borough": borough,
        "created_hour": hour,
        "complaint_hour": hour,
        "created_weekday": weekday,
        "complaint_weekday": weekday,
        "created_month": month,
        "complaint_month": month
    }
    if not features:
        features = list(values.keys())
    return pd.DataFrame([{feature: values.get(feature, 0) for feature in features}])
