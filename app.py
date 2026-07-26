import streamlit as st
import pandas as pd
import pickle
import os

# 1. Page Configuration & Layout
st.set_page_config(page_title="Syrian Climate Early-Warning AI", page_icon="🌾", layout="wide")

st.title("🛰️ LIVE 2026 Syrian Climate & Early-Warning AI System")
st.markdown("### Active Humanitarian Monitoring Dashboard | SDG 13 (Climate Action) & SDG 2 (Zero Hunger)")

# 2. File Verification Safeguards
if not os.path.exists("drought_model.pkl"):
    st.error("❌ Component Missing: 'drought_model.pkl' not found. Please run your Jupyter training cell to save your model.")
elif not os.path.exists("live_2026_forecast.csv"):
    st.error("❌ Component Missing: 'live_2026_forecast.csv' not found. Please run 'python live_pipeline.py' first to stream active data.")
else:
    # 3. Load Datasets & Saved Model Artifacts
    with open("drought_model.pkl", "rb") as file:
        model = pickle.load(file)
        
    live_data = pd.read_csv("live_2026_forecast.csv")
    live_data['date'] = pd.to_datetime(live_data['date'])
    
    # 4. Live Machine Learning Inference Engine
    # Isolate the exact feature matrices your model was trained on
    features = live_data[['max_temp', 'precipitation', 'root_zone_moisture', 'deep_soil_moisture', 'rolling_rain_14d']]
    
    # Generate live predictive probability percentages for upcoming drought risk anomalies
    live_data['drought_probability_pct'] = model.predict_proba(features)[:, 1] * 100
    
    # 5. Dashboard Visualizations (KPI Section)
    highest_risk = live_data['drought_probability_pct'].max()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="🚨 Max Predicted Crisis Risk (Next 7 Days)", 
            value=f"{highest_risk:.1f}%",
            delta="CRITICAL ALERT" if highest_risk > 50 else "STABLE CONDITIONS",
            delta_color="inverse"
        )
    with col2:
        current_temp = live_data['max_temp'].iloc[0]
        st.metric(label="🌡️ Current Max Temperature (Al-Hasakah)", value=f"{current_temp:.1f} °C")
    with col3:
        current_moisture = live_data['root_zone_moisture'].iloc[0]
        st.metric(label="💧 Current Crop Root Zone Moisture", value=f"{current_moisture:.3f} m³/m³")

    # 6. Interactive Forecast Horizon Timeline Chart
    st.subheader("🗓️ 7-Day Predictive Risk Horizon Timeline (2026)")
    chart_df = live_data.set_index('date')[['drought_probability_pct', 'max_temp']]
    # Rename columns on the fly for cleaner UI labels inside the chart
    chart_df.columns = ['Drought Probability (%)', 'Max Temperature (°C)']
    st.line_chart(chart_df)
    
    # 7. Raw Operational Data Grid Layout
    st.subheader("📋 Live Operational Data Feed (Current Week Matrix)")
    st.markdown("This structured dataset contains real-time telemetry variables and engineered features currently parsed by the predictive engine.")
    
    # Display data formatted for humanitarian decision-makers
    st.dataframe(
        live_data.style.format({
            'max_temp': '{:.1f}°C',
            'precipitation': '{:.2f}mm',
            'root_zone_moisture': '{:.3f}',
            'deep_soil_moisture': '{:.3f}',
            'rolling_rain_14d': '{:.2f}mm',
            'drought_probability_pct': '{:.1f}%'
        }), 
        use_container_width=True
    )
