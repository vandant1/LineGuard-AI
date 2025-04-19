import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from catboost import CatBoostClassifier
from datetime import datetime
import time

# -------------------- Streamlit Configuration --------------------
st.set_page_config(
    page_title="LineGuard AI - Power Line Monitoring",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# -------------------- Firebase Configuration --------------------
@st.cache_resource
def get_firebase_config():
    return {
        "FIREBASE_URL": "https://console.firebase.google.com/u/0/project/lineguard-ai/database/lineguard-ai-default-rtdb/data/~2F"  # Use Streamlit secrets
        "REFRESH_INTERVAL": 5  # Seconds between data refreshes
    }

firebase_config = get_firebase_config()

# -------------------- Load Trained Model --------------------
@st.cache_resource(ttl=3600)
def load_model():
    model = CatBoostClassifier()
    model.load_model("catboost_fault_detection_model.cbm")  # Update path
    return model

model = load_model()

# -------------------- Data Handling --------------------
@st.cache_data(ttl=firebase_config["REFRESH_INTERVAL"], show_spinner=False)
def fetch_firebase_data():
    try:
        response = requests.get(firebase_config["FIREBASE_URL"], timeout=5)
        response.raise_for_status()
        data = response.json()
        if data:
            df = pd.DataFrame(data.values())
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp", ascending=False)
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

def process_data(df):
    if not df.empty:
        features = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc", "fault_distance"]
        df["fault_type"] = model.predict(df[features])
        df["fault_severity"] = df.apply(classify_fault_severity, axis=1)
        df["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return df

# -------------------- Severity Classification --------------------
def classify_fault_severity(row):
    distance = row["fault_distance"]
    if distance > 6:
        return {"text": "Critical", "level": 3, "icon": "🚨"}
    elif distance > 4:
        return {"text": "Warning", "level": 2, "icon": "⚠️"}
    else:
        return {"text": "Normal", "level": 1, "icon": "✅"}

# -------------------- UI Components --------------------
def display_system_status(df):
    if not df.empty:
        latest = df.iloc[0]
        status = classify_fault_severity(latest)
        
        with st.status_container:
            cols = st.columns(4)
            cols[0].metric("System Status", f"{status['icon']} {status['text']}", 
                         help="Overall system health status")
            cols[1].metric("Latest Fault Distance", f"{latest['fault_distance']:.2f} m",
                         delta_color="off", help="Distance from substation to fault location")
            cols[2].metric("Data Freshness", latest["last_updated"],
                         help="Last received sensor data timestamp")
            cols[3].metric("Update Frequency", f"{firebase_config['REFRESH_INTERVAL']}s",
                         help="Data refresh interval")

def create_fault_gauge(value):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fault Distance (m)", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 10], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 4], 'color': "green"},
                {'range': [4, 6], 'color': "orange"},
                {'range': [6, 10], 'color': "red"}]
        }
    ))
    fig.update_layout(height=300, margin=dict(t=0, b=0))
    return fig

# -------------------- Main App --------------------
def main():
    # ----- Header Section -----
    st.title("⚡ LineGuard AI - Power Line Monitoring System")
    st.markdown("""
    **Real-time Transmission Line Fault Detection and Predictive Analytics Platform**  
    *Empowering Grid Operators with AI-Driven Insights*
    """)
    
    # Real-time Data Updates
    data_placeholder = st.empty()
    df = process_data(fetch_firebase_data())
    
    # ----- System Status Dashboard -----
    display_system_status(df)
    
    # ----- Main Columns -----
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # ----- Fault Visualization -----
        st.subheader("🔍 Fault Location Analysis")
        if not df.empty:
            tab1, tab2 = st.tabs(["📈 Spatial Distribution", "📊 Severity Analysis"])
            
            with tab1:
                fig = px.scatter(
                    df.head(20),
                    x="fault_distance",
                    y="fault_type",
                    color="fault_severity",
                    size="fault_distance",
                    hover_data=["timestamp"],
                    labels={"fault_distance": "Distance (meters)", "fault_type": "Fault Type"},
                    title="Fault Distribution Pattern Analysis"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.plotly_chart(create_fault_gauge(df.iloc[0]["fault_distance"]), 
                              use_container_width=True)

    with col2:
        # ----- Alert System -----
        st.subheader("🚨 Active Alerts")
        if not df.empty:
            critical_alerts = df[df["fault_severity"].apply(lambda x: x['level'] >= 2)]
            if not critical_alerts.empty:
                for _, alert in critical_alerts.head(3).iterrows():
                    st.error(f"""
                    **{alert['fault_severity']['icon']} {alert['fault_severity']['text']} Alert**  
                    *{alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}*  
                    Fault Type: {alert['fault_type']}  
                    Distance: {alert['fault_distance']:.2f}m
                    """)
            else:
                st.success("No active alerts - System operating normally")

    # ----- Real-time Monitoring -----
    st.subheader("📈 Real-time Power Line Metrics")
    if not df.empty:
        fig = px.line(df.sort_values("timestamp"),
                    x="timestamp", 
                    y=["Va", "Vb", "Vc", "Ia", "Ib", "Ic"],
                    title="Multi-Parameter Time Series Analysis",
                    labels={"value": "Magnitude", "variable": "Parameter"},
                    height=500)
        st.plotly_chart(fig, use_container_width=True)

    # ----- Historical Analysis -----
    st.subheader("📚 Historical Data Insights")
    if not df.empty:
        with st.expander("View Historical Records"):
            st.dataframe(
                df[["timestamp", "fault_distance", "fault_type", 
                    "fault_severity", "Va", "Vb", "Vc"]].sort_values("timestamp", ascending=False),
                column_config={
                    "timestamp": "Timestamp",
                    "fault_distance": st.column_config.NumberColumn(
                        "Distance (m)", format="%.2f m"
                    ),
                    "fault_severity": "Severity Level"
                },
                height=300
            )

    # ----- File Upload Section -----
    st.subheader("📤 Batch Analysis Module")
    uploaded_file = st.file_uploader("Upload historical data for analysis", type=["csv"])
    if uploaded_file:
        with st.spinner("Analyzing data..."):
            uploaded_df = pd.read_csv(uploaded_file)
            processed_df = process_data(uploaded_df)
            
            st.success("Analysis Complete")
            st.download_button(
                label="Download Results",
                data=processed_df.to_csv(index=False),
                file_name="fault_analysis_report.csv",
                mime="text/csv"
            )

    # ----- Footer -----
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center">
        <p>🔒 ISO 27001 Certified | 🛡️ GDPR Compliant</p>
        <p>Developed by [Your Organization] | 📧 support@lineguard.ai | 📞 +1 (555) 123-4567</p>
        <p style="font-size: 0.8em">System Version 2.1.0 | Last Updated: 2024-03-15</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()