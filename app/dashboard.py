import streamlit as st
from joblib import load, dump
import numpy as np
import pandas as pd

# Load the saved model and scaler
rf_model = load('models/rf_wildfire_model.joblib')
scaler = load('models/scaler.joblib')

st.title('Wildfire Risk Prediction Dashboard')

st.write("Input environmental and situational data to predict wildfire risk levels.")

# ---- User Input Form ----
with st.form("input_form"):
    latitude = st.number_input('Latitude', min_value=-90.0, max_value=90.0, value=40.0)
    longitude = st.number_input('Longitude', min_value=-180.0, max_value=180.0, value=-120.0)
    discovery_doy = st.slider('Discovery Day of Year', min_value=1, max_value=366, value=150)
    discovery_hour = st.slider('Discovery Hour (0-23)', min_value=0, max_value=23, value=14)

    state = st.selectbox('State (Encoded)', list(range(0, 51)))  # Replace with actual label encoding mapping
    owner_descr = st.selectbox('Owner Description (Encoded)', list(range(0, 10)))  # Replace with actual mapping
    season = st.selectbox('Season (Encoded: Winter=0, Spring=1, Summer=2, Fall=3)', [0, 1, 2, 3])
    cause_simple = st.selectbox('Cause Simple (Encoded: Human=0, Natural=1, Unknown=2)', [0, 1, 2])

    submit = st.form_submit_button('Predict Wildfire Risk')

# ---- Process & Predict ----
if submit:
    # Assemble features into a NumPy array (adjust order according to your model)
    input_features = np.array([
        latitude, longitude, discovery_doy, discovery_hour,
        state, owner_descr, season, cause_simple
    ]).reshape(1, -1)

    # Scale features (assuming the scaler was fit on the same set)
    scaled_features = scaler.transform(input_features)

    # Predict
    prediction = rf_model.predict(scaled_features)[0]

    # Decode prediction
    risk_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
    risk_level = risk_mapping.get(prediction, 'Unknown')

    st.success(f'### Predicted Wildfire Risk Level: {risk_level}')

    st.write("**Details:**")
    st.write(f"Raw Prediction: {prediction}")
    st.write(f"Input Features: {input_features}")