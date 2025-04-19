import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.express as px
from catboost import CatBoostClassifier

# -------------------- Streamlit Page Config --------------------
st.set_page_config(page_title="LineGuard AI", layout="wide")

# -------------------- Firebase Configuration --------------------
FIREBASE_URL = "https://lineguard-ai-default-rtdb.firebaseio.com/.json"  # Use proper REST URL for real-time DB

# -------------------- Load Trained CatBoost Model --------------------
@st.cache_resource()
def load_model():
    model = CatBoostClassifier()
    model.load_model("C:/Users/Hp/Downloads/catboost_fault_detection_model (1).cbm")
    return model

model = load_model()

# -------------------- Fetch Real-Time Data from Firebase --------------------
def fetch_firebase_data():
    response = requests.get(FIREBASE_URL)
    if response.status_code == 200:
        data = response.json()
        if data:
            df = pd.DataFrame(data.values())
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
            df = df.sort_values("timestamp")
            return df
    return pd.DataFrame()

# -------------------- Fault Severity Logic --------------------
def classify_fault_severity(row):
    if row["fault_distance"] > 6:
        return "Critical 🚨"
    elif row["fault_distance"] > 4:
        return "Warning ⚠️"
    else:
        return "Normal ✅"

# -------------------- Fault Prediction Pipeline --------------------
def predict_fault(df):
    features = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc", "fault_distance"]
    df["fault_type"] = model.predict(df[features])
    df["fault_severity"] = df.apply(classify_fault_severity, axis=1)
    return df

# -------------------- Header --------------------
st.title("⚡ LineGuard AI")
st.markdown("**Effortless Fault Detection and Insights for Your Power Lines.**")

# -------------------- Sidebar: Data Fetching --------------------
st.sidebar.header("📡 Live Data Feed")
if st.sidebar.button("Fetch Latest Data"):
    sensor_data = fetch_firebase_data()
    if not sensor_data.empty:
        sensor_data = predict_fault(sensor_data)
        st.sidebar.success("✅ Data Fetched Successfully")
    else:
        st.sidebar.warning("⚠️ No Data Available!")
else:
    sensor_data = fetch_firebase_data()
    if not sensor_data.empty:
        sensor_data = predict_fault(sensor_data)

# -------------------- Fault Location Visualization --------------------
st.subheader("📍 Fault Location (7-meter Line)")
if not sensor_data.empty:
    fig = px.scatter(
        sensor_data,
        x="fault_distance",
        y=[1] * len(sensor_data),
        color="fault_severity",
        color_discrete_map={
            "Critical 🚨": "red",
            "Warning ⚠️": "orange",
            "Normal ✅": "green"
        },
        labels={"fault_distance": "Distance (meters)"},
        title="Fault Location on Transmission Line"
    )
    fig.update_yaxes(visible=False, showticklabels=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Live fault location data not available yet.")

# -------------------- Real-Time Voltage and Current Monitoring --------------------
st.subheader("📊 Voltage and Current Monitoring")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔋 Voltage Trends (Va, Vb, Vc)")
    if not sensor_data.empty:
        fig_v = px.line(sensor_data, x="timestamp", y=["Va", "Vb", "Vc"], title="Voltage Over Time")
        st.plotly_chart(fig_v, use_container_width=True)
    else:
        st.info("No voltage data available yet.")

with col2:
    st.markdown("### ⚡ Current Trends (Ia, Ib, Ic)")
    if not sensor_data.empty:
        fig_i = px.line(sensor_data, x="timestamp", y=["Ia", "Ib", "Ic"], title="Current Over Time")
        st.plotly_chart(fig_i, use_container_width=True)
    else:
        st.info("No current data available yet.")

# -------------------- Fault Insights Table --------------------
st.subheader("📌 Fault Insights")
if not sensor_data.empty:
    insights = sensor_data[["timestamp", "fault_distance", "fault_type", "fault_severity"]]
    insights = insights.sort_values(by="timestamp", ascending=False)
    st.dataframe(insights)
else:
    st.warning("No fault insights to show.")

# -------------------- File Upload for Batch Prediction --------------------
st.subheader("📂 Upload CSV for Batch Predictions")
uploaded_file = st.file_uploader("Upload your CSV file with voltage, current, and distance data:", type=["csv"])
if uploaded_file:
    uploaded_df = pd.read_csv(uploaded_file)
    if all(col in uploaded_df.columns for col in ["Ia", "Ib", "Ic", "Va", "Vb", "Vc", "fault_distance"]):
        predicted_df = predict_fault(uploaded_df)
        st.success("✅ Predictions Complete")
        st.dataframe(predicted_df[["fault_distance", "fault_type", "fault_severity"]])
    else:
        st.error("Uploaded file is missing required columns.")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("📌 Developed as a part of **Smart Transmission Line Fault Detection System**")
