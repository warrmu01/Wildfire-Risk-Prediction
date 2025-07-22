import streamlit as st
from joblib import load
import numpy as np

# Load model, scaler, and encoders
rf_model = load('models/rf_wildfire_model.joblib')
scaler = load('models/scaler.joblib')
state_encoder = load('models/state_encoder.joblib')
owner_encoder = load('models/owner_descr_encoder.joblib')
cause_descr_encoder = load('models/stat_cause_descr_encoder.joblib')

st.title('🔥 Wildfire Risk Prediction Dashboard')
st.markdown("Predict wildfire **risk levels** based on environmental and situational data.")

st.header("📋 Input Features")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        latitude = st.number_input('Latitude', min_value=-90.0, max_value=90.0, value=40.0)
        longitude = st.number_input('Longitude', min_value=-180.0, max_value=180.0, value=-120.0)
        discovery_doy = st.slider('Discovery Day of Year', min_value=1, max_value=366, value=150)
        discovery_hour = st.slider('Discovery Hour (0-23)', min_value=0, max_value=23, value=14)

    with col2:
        state_name = st.selectbox('State', state_encoder.classes_)
        owner_descr_name = st.selectbox('Owner Description', owner_encoder.classes_)
        season = st.selectbox('Season', ['Winter', 'Spring', 'Summer', 'Fall'])
        stat_cause_descr_name = st.selectbox('Stat Cause Description', cause_descr_encoder.classes_)
        cause_simple = st.selectbox('Cause Simple', ['Human', 'Natural', 'Unknown'])

    submit = st.form_submit_button('🚀 Predict Wildfire Risk')

if submit:
    # --- Encode categorical inputs ---
    state = state_encoder.transform([state_name])[0]
    owner_descr = owner_encoder.transform([owner_descr_name])[0]
    stat_cause_descr = cause_descr_encoder.transform([stat_cause_descr_name])[0]
    season_encoded = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}[season]
    cause_simple_encoded = {'Human': 0, 'Natural': 1, 'Unknown': 2}[cause_simple]

    # --- Scale numerical features ---
    numerical_features = np.array([
        latitude, longitude, discovery_doy, discovery_hour
    ]).reshape(1, -1)

    scaled_numerical = scaler.transform(numerical_features)

    # --- Combine scaled numerical + categorical features ---
    categorical_features = np.array([
        state, owner_descr, season_encoded, stat_cause_descr, cause_simple_encoded
    ]).reshape(1, -1)

    final_features = np.hstack((scaled_numerical, categorical_features))

    # --- Make prediction ---
    st.write("Final input to model:", final_features)

    prediction = rf_model.predict(final_features)[0]
    risk_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
    risk_level = risk_mapping.get(prediction, 'Unknown')
    risk_colors = {'Low': '🟩', 'Medium': '🟨', 'High': '🟥'}


    st.success(f"### {risk_colors.get(risk_level)} Predicted Wildfire Risk Level: **{risk_level}**")

    with st.expander("See Prediction Details"):
        st.write(f"**Raw Prediction Class:** {prediction}")
        st.write(f"**Scaled Numerical Features:** {scaled_numerical}")
        st.write(f"**Categorical Features:** {categorical_features}")
        st.write(f"**Final Feature Array:** {final_features}")

    probs = rf_model.predict_proba(final_features)[0]
    risk_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
    st.write("Class Probabilities:")
    for i, prob in enumerate(probs):
        st.write(f"{risk_mapping[i]}: {prob:.2f}")
