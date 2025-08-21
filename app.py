import streamlit as st
import numpy as np
import joblib

# Load Mobile Price Prediction Model and Scaler
model = joblib.load('mobile_price_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Mobile Price Prediction App")

# Input fields for mobile features
Price = st.number_input("Price (USD)", 500, 5000, 2000)
weight = st.number_input("Weight (grams)",0,1000,150)
resolution = st.number_input("Resolution (inches)", 1.0, 15.0, 5.0)
ppi = st.number_input("PPI", 100, 1000, 400)
cpu_core = st.number_input("CPU Cores", 1, 16, 4)
cpu_freq = st.number_input("CPU Frequency (GHz)", 0.1, 5.0, 2.0)
internal_mem = st.number_input("Internal Memory (GB)", 1, 512, 32)
ram = st.number_input("RAM (GB)", 1, 32, 4)
rear_cam = st.number_input("Rear Camera (MP)", 0, 64, 12)
front_cam = st.number_input("Front Camera (MP)", 0, 32, 8)
battery = st.number_input("Battery (mAh)", 500,10000, 3000)
thickness = st.number_input("Thickness (mm)", 3.0, 20.0, 8.0)


mobile_input = np.array([[Price,weight, resolution, ppi, cpu_core, cpu_freq, internal_mem, ram, rear_cam, front_cam, battery, thickness]])

if st.button("🔮 Predict Sale"):
    scaled = scaler.transform(mobile_input) 
    prediction = model.predict(scaled)
    st.success(f"📈 Predicted Sale: {int(prediction[0])} USD")

