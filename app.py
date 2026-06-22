import streamlit as st
import joblib
import numpy as np
import os

model_path = os.path.join(os.path.dirname(__file__), "housing_model.pkl")
try:
    model = joblib.load(model_path)
except Exception as e:
    model = None
    st.error(f"Failed to load model: {e}")

st.title("California Housing Price Predictor")

MedInc = st.number_input("Median Income")
HouseAge = st.number_input("House Age")
AveRooms = st.number_input("Average Rooms")
AveBedrms = st.number_input("Average Bedrooms")
Population = st.number_input("Population")
AveOccup = st.number_input("Average Occupancy")
Latitude = st.number_input("Latitude")
Longitude = st.number_input("Longitude")

if st.button("Predict"):
    if model is None:
        st.error("Model not loaded. Check model file and restart the app.")
    else:
        data = np.array([[
            MedInc,
            HouseAge,
            AveRooms,
            AveBedrms,
            Population,
            AveOccup,
            Latitude,
            Longitude
        ]])

        prediction = model.predict(data)

        st.success(
            f"Predicted House Value: ${prediction[0]*100000:.2f}"
        )