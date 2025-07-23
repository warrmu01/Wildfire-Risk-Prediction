import streamlit as st
from joblib import load
import numpy as np

# Load model, scaler, and encoders
rf_model = load('models/rf_wildfire_model.joblib')
scaler = load('models/scaler.joblib')
state_encoder = load('models/state_encoder.joblib')
owner_encoder = load('models/owner_descr_encoder.joblib')
cause_descr_encoder = load('models/stat_cause_descr_encoder.joblib')

state_bounds = {
    'AL': {'lat': (30.2, 35.0), 'lon': (-88.5, -84.8)},
    'AK': {'lat': (51.2, 71.4), 'lon': (-179.1, -129.9)},
    'AZ': {'lat': (31.3, 37.0), 'lon': (-114.8, -109.0)},
    'AR': {'lat': (33.0, 36.5), 'lon': (-94.6, -89.6)},
    'CA': {'lat': (32.5, 42.0), 'lon': (-124.5, -114.1)},
    'CO': {'lat': (36.9, 41.0), 'lon': (-109.1, -102.0)},
    'CT': {'lat': (40.9, 42.1), 'lon': (-73.7, -71.8)},
    'DE': {'lat': (38.4, 39.9), 'lon': (-75.8, -75.0)},
    'FL': {'lat': (24.5, 31.0), 'lon': (-87.6, -80.0)},
    'GA': {'lat': (30.4, 35.0), 'lon': (-85.6, -80.8)},
    'HI': {'lat': (18.9, 20.6), 'lon': (-160.3, -154.8)},
    'ID': {'lat': (42.0, 49.0), 'lon': (-117.2, -111.0)},
    'IL': {'lat': (36.9, 42.5), 'lon': (-91.5, -87.0)},
    'IN': {'lat': (37.8, 41.8), 'lon': (-88.1, -84.8)},
    'IA': {'lat': (40.4, 43.5), 'lon': (-96.6, -90.1)},
    'KS': {'lat': (36.9, 40.0), 'lon': (-102.1, -94.6)},
    'KY': {'lat': (36.5, 39.2), 'lon': (-89.6, -82.0)},
    'LA': {'lat': (28.9, 33.0), 'lon': (-94.0, -88.8)},
    'ME': {'lat': (43.1, 47.5), 'lon': (-71.1, -66.9)},
    'MD': {'lat': (37.9, 39.7), 'lon': (-79.5, -75.0)},
    'MA': {'lat': (41.2, 42.9), 'lon': (-73.5, -69.9)},
    'MI': {'lat': (41.7, 48.3), 'lon': (-90.4, -82.4)},
    'MN': {'lat': (43.5, 49.4), 'lon': (-97.2, -89.5)},
    'MS': {'lat': (30.2, 35.0), 'lon': (-91.7, -88.0)},
    'MO': {'lat': (36.0, 40.6), 'lon': (-95.8, -89.1)},
    'MT': {'lat': (44.4, 49.1), 'lon': (-116.1, -104.0)},
    'NE': {'lat': (39.9, 43.0), 'lon': (-104.1, -95.3)},
    'NV': {'lat': (35.0, 42.0), 'lon': (-120.0, -114.0)},
    'NH': {'lat': (42.7, 45.3), 'lon': (-72.6, -70.6)},
    'NJ': {'lat': (38.9, 41.4), 'lon': (-75.6, -73.9)},
    'NM': {'lat': (31.3, 37.0), 'lon': (-109.1, -103.0)},
    'NY': {'lat': (40.5, 45.1), 'lon': (-79.8, -71.8)},
    'NC': {'lat': (33.8, 36.6), 'lon': (-84.3, -75.5)},
    'ND': {'lat': (45.9, 49.0), 'lon': (-104.1, -96.6)},
    'OH': {'lat': (38.4, 41.9), 'lon': (-84.8, -80.5)},
    'OK': {'lat': (33.6, 37.0), 'lon': (-103.0, -94.4)},
    'OR': {'lat': (41.9, 46.3), 'lon': (-124.6, -116.5)},
    'PA': {'lat': (39.7, 42.3), 'lon': (-80.5, -74.7)},
    'RI': {'lat': (41.1, 42.0), 'lon': (-71.9, -71.1)},
    'SC': {'lat': (32.0, 35.2), 'lon': (-83.4, -78.5)},
    'SD': {'lat': (42.5, 45.9), 'lon': (-104.1, -96.4)},
    'TN': {'lat': (35.0, 36.7), 'lon': (-90.3, -81.6)},
    'TX': {'lat': (25.8, 36.5), 'lon': (-106.6, -93.5)},
    'UT': {'lat': (36.9, 42.0), 'lon': (-114.1, -109.0)},
    'VT': {'lat': (42.7, 45.0), 'lon': (-73.4, -71.5)},
    'VA': {'lat': (36.5, 39.5), 'lon': (-83.7, -75.2)},
    'WA': {'lat': (45.5, 49.1), 'lon': (-124.8, -116.9)},
    'WV': {'lat': (37.2, 40.6), 'lon': (-82.6, -77.7)},
    'WI': {'lat': (42.5, 47.3), 'lon': (-92.9, -86.2)},
    'WY': {'lat': (41.0, 45.0), 'lon': (-111.1, -104.0)},
}


st.title('🔥 Wildfire Risk Prediction Dashboard')
st.markdown("Predict wildfire **risk levels** based on environmental and situational data.")

st.header("📋 Input Features")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col2:
        state_name = st.selectbox('State', state_encoder.classes_)
        bounds = state_bounds.get(state_name, {'lat': (-90, 90), 'lon': (-180, 180)})
        owner_descr_name = st.selectbox('Owner Description', owner_encoder.classes_)
        season = st.selectbox('Season', ['Winter', 'Spring', 'Summer', 'Fall'])
        stat_cause_descr_name = st.selectbox('Stat Cause Description', cause_descr_encoder.classes_)
        cause_simple = st.selectbox('Cause Simple', ['Human', 'Natural', 'Unknown'])

    with col1:
        latitude = st.number_input(
            'Latitude', min_value=bounds['lat'][0], max_value=bounds['lat'][1],
            value=(bounds['lat'][0] + bounds['lat'][1]) / 2, step=0.01
        )
        longitude = st.number_input(
            'Longitude', min_value=bounds['lon'][0], max_value=bounds['lon'][1],
            value=(bounds['lon'][0] + bounds['lon'][1]) / 2, step=0.01
        )
        discovery_doy = st.slider('Discovery Day of Year', min_value=1, max_value=366, value=150)
        discovery_hour = st.slider('Discovery Hour (0-23)', min_value=0, max_value=23, value=14)

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
    st.write("Class Probabilities:")
    for i, prob in enumerate(probs):
        st.write(f"{risk_mapping[i]}: {prob:.2f}")
