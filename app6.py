import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from datetime import datetime, timedelta
from catboost import CatBoostClassifier

# ------------------ Config ------------------
st.set_page_config(page_title="LineGuard AI Analyzer", layout="wide")
st.title("⚡ LineGuard AI - Advanced Fault Detection Simulator")

# ------------------ Load Model ------------------
@st.cache_resource()
def load_model():
    model = CatBoostClassifier()
    model.load_model("C:\\2k25\\LineGuard AI\\models\\catboost_fault_detection_model_single_phase (1).cbm")
    return model

model = load_model()

# ------------------ Helper Functions ------------------
def simulate_data(num_rows: int = 50):
    """Fetch Real-Time voltage, current, distance readings"""
    now = datetime.now()
    timestamps = pd.date_range(start=now, periods=num_rows, freq="5s")
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

# ------------------ Sidebar UI ------------------
st.sidebar.header("🛠 Data Control")

# ESP32 Live Simulation Info
st.sidebar.markdown("### 📡 ESP32 Live Feed")
st.sidebar.success("🟢 ESP32 Connected")
st.sidebar.markdown("**MAC Address:** `30:AE:A4:9F:8B:C1`")
st.sidebar.markdown("**IP Address:** `192.168.1.42`")
st.sidebar.markdown("**Uptime:** `0d 03h 14m 27s`")
last_ping = datetime.now() - timedelta(seconds=np.random.randint(1, 10))
st.sidebar.markdown(f"**⏱ Last Sensor Ping:** `{last_ping.strftime('%H:%M:%S')}`")

# Data Mode
mode = st.sidebar.radio("Choose Data Mode:", ["Fetched Readings", "Upload CSV File"])

if mode == "Fetched Readings":
    num_rows = st.sidebar.slider("Number of Fetched readings", 10, 100, 30)
    if st.sidebar.button("🎛 Fetch Real-time Data"):
        with st.spinner("Fetching data from sensors..."):
            time.sleep(1.5)
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

    # Voltage & Current Trends
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

    # ESP32 Logs (Fake Console)
    with st.expander("📜 ESP32 Serial Logs"):
        st.text("""
[INFO] Boot completed at 0x0001
[DATA] Voltage(Va) = {:.2f} | Current(Ia) = {:.2f} | Distance = {:.2f}m
[INFO] Fault Status: {}
[DATA] Transmitting data to Firebase...
[INFO] Firebase Push: success ✅
""".format(df.iloc[-1]['Va'], df.iloc[-1]['Ia'], df.iloc[-1]['fault_distance'], df.iloc[-1]['fault_severity']))

    # Advanced Metrics
    st.subheader("📐 System Performance Metrics")
    st.metric("🔌 Avg Voltage (V)", f"{df['Va'].mean():.2f}", delta=f"{df['Va'].std():.2f}")
    st.metric("⚡ Avg Current (A)", f"{df['Ia'].mean():.2f}", delta=f"{df['Ia'].std():.2f}")
    st.metric("🧮 Avg Power (W)", f"{df['power_Watts'].mean():.2f}", delta=f"{df['power_Watts'].std():.2f}")
    st.metric("📉 Voltage Error %", f"{df['Va_error_%'].mean():.2f}%", delta=f"±{df['Va_error_%'].std():.2f}%")

    # Predictions Table
    st.subheader("📌 Fault Predictions")
    st.dataframe(df[["timestamp", "Ia", "Va", "power_Watts", "Va_error_%", "fault_distance", "fault_type", "fault_severity"]])
else:
    st.info("No data to display. Choose a Fetching mode from the sidebar.")

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("🔧 Developed to emulate real-world fault monitoring on a 7-meter transmission line using LineGuard AI Analyzer.")