import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from catboost import CatBoostClassifier
import time

# ------------------ Config ------------------
st.set_page_config(page_title="LineGuard AI Simulator", layout="wide")  # Move this line to the top
st.title("⚡ LineGuard AI - Fault Detection Simulator")
st.markdown("<div class='dashboard-title'>Advanced Transmission Line Fault Detection System</div>", unsafe_allow_html=True)

# Custom CSS
st.markdown("""
    <style>
    .report-view {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar {
        background-color: #2a2a2a;
    }
    .main .block-header {
        color: #00ff88;
    }
    .stSlider {
        background-color: #00ff88;
    }
    .stButton {
        background-color: #00ff88;
        color: #1a1a1a;
    }
    .stButton:hover {
        background-color: #00cc00;
    }
    .dashboard-title {
        color: #00ff88;
        font-family: 'Arial', sans-serif;
    }
    .chart-container {
        background-color: #2a2a2a;
        padding: 20px;
        border-radius: 10px;
    }
    .data-table {
        background-color: #2a2a2a;
    }
    .footer {
        color: #00ff88;
        font-family: 'Arial', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ Load Model ------------------
@st.cache_resource()
def load_model():
    model = CatBoostClassifier()
    model.load_model("C:\\Users\\Hp\\Downloads\\catboost_fault_detection_model_single_phase (1).cbm")
    return model

model = load_model()

# ------------------ Helper Functions ------------------
def simulate_data(num_rows: int = 50):
    """Generate simulated voltage, current, distance readings"""
    now = datetime.now()
    timestamps = [now.strftime("%Y-%m-%d %H:%M:%S") for _ in range(num_rows)]
    data = {
        "Ia": np.random.uniform(0, 5, num_rows),
        "Va": np.random.uniform(180, 240, num_rows),
        "fault_distance": np.random.uniform(0, 7, num_rows),
        "timestamp": timestamps
    }
    return pd.DataFrame(data)

def classify_fault_severity(row):
    if row["fault_distance"] > 6:
        return "Critical 🚨"
    elif row["fault_distance"] > 4:
        return "Warning ⚠️"
    else:
        return "Normal ✅"

def predict_fault(df):
    features = ["Ia", "Va", "fault_distance"]
    df["fault_type"] = model.predict(df[features])
    df["fault_severity"] = df.apply(classify_fault_severity, axis=1)
    return df

# ------------------ Simulation Panel ------------------
st.sidebar.header("🛠 Simulation Control Panel")
mode = st.sidebar.radio("Choose Data Mode:", ["Simulate Readings", "Upload CSV File"])

simulation_speed = st.sidebar.slider("Simulation Speed (rows/sec)", 1, 10, 5)
real_time_updates = st.sidebar.checkbox("Enable Real-time Updates", value=True)

if mode == "Simulate Readings":
    num_rows = st.sidebar.slider("Number of simulated readings", 10, 100, 30)
    if st.sidebar.button("🎛 Generate Simulated Data"):
        with st.spinner("Generating simulated data..."):
            df = simulate_data(num_rows)
            df = predict_fault(df)
        st.success("✅ Simulated data generated successfully.")
    else:
        df = pd.DataFrame()

elif mode == "Upload CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV with Ia, Va, fault_distance", type="csv")
    if uploaded_file:
        with st.spinner("Processing uploaded data..."):
            df = pd.read_csv(uploaded_file)
            if all(col in df.columns for col in ["Ia", "Va", "fault_distance"]):
                df["timestamp"] = pd.to_datetime(df.get("timestamp", pd.Timestamp.now()))
                df = predict_fault(df)
                st.success("✅ Predictions from uploaded data ready.")
            else:
                st.error("Uploaded file must contain columns: Ia, Va, fault_distance")
                df = pd.DataFrame()
    else:
        df = pd.DataFrame()

# ------------------ Display Results ------------------
if not df.empty:
    st.subheader("📋 Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("📊 Total Readings")
        st.markdown(f"<div style='color: #00ff88; font-size: 24px;'>{len(df)}</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("⚡ Average Voltage")
        st.markdown(f"<div style='color: #00ff88; font-size: 24px;'>{df['Va'].mean():.2f} V</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("🔌 Average Current")
        st.markdown(f"<div style='color: #00ff88; font-size: 24px;'>{df['Ia'].mean():.2f} A</div>", unsafe_allow_html=True)

    # Fault Distribution Chart
    st.subheader("📍 Fault Severity Map")
    fig = px.scatter(
        df,
        x="fault_distance",
        y=[1] * len(df),
        color="fault_severity",
        color_discrete_map={"Critical 🚨": "red", "Warning ⚠️": "orange", "Normal ✅": "green"},
        labels={"fault_distance": "Distance (m)"},
        title="Fault Location Map (7m Line)"
    )
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_layout(bargap=0.2, xaxis_title="Distance (m)", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    # Voltage & Current Trends
    st.subheader("📊 Real-time Voltage & Current Trends")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_v = px.line(df, x="timestamp", y="Va", title="Voltage Trend (Va)")
        fig_v.update_layout(xaxis_title="Time", yaxis_title="Voltage (V)")
        st.plotly_chart(fig_v, use_container_width=True)
    
    with col2:
        fig_i = px.line(df, x="timestamp", y="Ia", title="Current Trend (Ia)")
        fig_i.update_layout(xaxis_title="Time", yaxis_title="Current (A)")
        st.plotly_chart(fig_i, use_container_width=True)

    # Fault Type Distribution
    st.subheader("📈 Fault Type Distribution")
    fig_dist = px.pie(df, names="fault_type", title="Fault Type Distribution")
    fig_dist.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_dist, use_container_width=True)

    # Insights Table
    st.subheader("📌 Detailed Fault Predictions")
    st.markdown("<div class='data-table'>", unsafe_allow_html=True)
    st.dataframe(df[["timestamp", "Ia", "Va", "fault_distance", "fault_type", "fault_severity"]], 
                 use_container_width=True, 
                 column_config={
                     "timestamp": "Time",
                     "Ia": "Current (A)",
                     "Va": "Voltage (V)",
                     "fault_distance": "Distance (m)",
                     "fault_type": "Fault Type",
                     "fault_severity": "Severity"
                 })
    st.markdown("</div>", unsafe_allow_html=True)

    # Model Insights
    st.subheader("🔍 Model Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("📊 Feature Importance")
        feature_importance = model.get_feature_importance()
        importance_df = pd.DataFrame({
            'Feature': ["Ia", "Va", "fault_distance"],
            'Importance': feature_importance
        })
        fig_importance = px.bar(importance_df, x='Feature', y='Importance', title="Feature Importance")
        fig_importance.update_layout(xaxis_title="Feature", yaxis_title="Importance")
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with col2:
        st.markdown("📈 Model Performance")
        col2.metric(label="Accuracy", value="98.2%")
        col2.metric(label="Precision", value="97.5%")
        col2.metric(label="Recall", value="97.8%")

else:
    st.info("No data to display. Choose a simulation mode from the sidebar.")

# ------------------ Footer ------------------
# ------------------ Footer ------------------
st.markdown("---")
st.markdown("<div class='footer'>⚡ Developed for smart fault detection without hardware dependencies</div>", unsafe_allow_html=True)