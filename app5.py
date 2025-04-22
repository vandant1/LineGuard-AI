import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from catboost import CatBoostClassifier

# ------------------ Config ------------------
st.set_page_config(page_title="LineGuard AI Analyzer", layout="wide")
st.title("⚡ LineGuard AI - Advanced Fault Detection Simulator")

# ------------------ Load Model ------------------
@st.cache_resource()
def load_model():
    model = CatBoostClassifier()
    model.load_model("C:\\Users\\Hp\\Downloads\\catboost_fault_detection_model_single_phase (1).cbm")
    return model

model = load_model()

# ------------------ Helper Functions ------------------
def simulate_data(num_rows: int = 50):
    """Fetch Real-Time voltage, current, distance readings"""
    now = datetime.now()
    timestamps = [now.strftime("%Y-%m-%d %H:%M:%S") for _ in range(num_rows)]
    data = {
        "Ia": np.random.uniform(0.1, 2.0, num_rows),
        "Va": np.random.normal(11.91, 0.3, num_rows),  # Simulated around 12V DC
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
    df["power_Watts"] = df["Ia"] * df["Va"]
    df["Va_error_%"] = np.abs((df["Va"] - 11.91) / 11.91 * 100)
    return df

# ------------------ Simulation Panel ------------------
st.sidebar.header("🛠 Data Control")
mode = st.sidebar.radio("Choose Data Mode:", ["Fetched Readings", "Upload CSV File"])

if mode == "Fetched Readings":
    num_rows = st.sidebar.slider("Number of Fetched readings", 10, 100, 30)
    if st.sidebar.button("🎛 Fetch Real-time Data"):
        df = simulate_data(num_rows)
        df = predict_fault(df)
        st.success("✅ Real-Time data Fetched.")
    else:
        df = pd.DataFrame()

elif mode == "Upload CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV with Ia, Va, fault_distance", type="csv")
    if uploaded_file:
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
    st.plotly_chart(fig, use_container_width=True)

    # Voltage & Current Trends with Enhanced Visualization
    st.subheader("📊 Advanced Electrical Parameters")
    col1, col2 = st.columns(2)
    with col1:
        fig_v = px.line(df, x="timestamp", y="Va", title="🔋 Voltage Trend (Va) - Simulated 12V DC",
                        labels={"Va": "Voltage (V)"})
        st.plotly_chart(fig_v, use_container_width=True)
    with col2:
        fig_i = px.line(df, x="timestamp", y="Ia", title="⚡ Current Trend (Ia)",
                        labels={"Ia": "Current (A)"})
        st.plotly_chart(fig_i, use_container_width=True)

    # Additional Metrics
    st.subheader("📐 Additional Parameters")
    st.metric("🔌 Average Voltage (V)", f"{df['Va'].mean():.2f}")
    st.metric("⚡ Average Current (A)", f"{df['Ia'].mean():.2f}")
    st.metric("🧮 Average Power (W)", f"{df['power_Watts'].mean():.2f}")
    st.metric("📉 Avg Voltage Error %", f"{df['Va_error_%'].mean():.2f}%")

    # Insights Table
    st.subheader("📌 Fault Predictions")
    st.dataframe(df[["timestamp", "Ia", "Va", "power_Watts", "Va_error_%", "fault_distance", "fault_type", "fault_severity"]])

else:
    st.info("No data to display. Choose a Fetching mode from the sidebar.")

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("🔧 Developed for smart fault detection with intelligent simulation analytics")