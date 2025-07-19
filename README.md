# Wildfire Risk Prediction Project 🚧 (Work in Progress)

## 📌 Project Overview
This project predicts the **expected severity or risk level** of wildfires in specific US regions based on historical incidents and environmental factors. Using machine learning models, the system forecasts wildfire **risk levels (Low, Medium, High)** based on location, season, and discovery conditions **if a fire occurs.**

## 🎯 Objective
- Predict wildfire severity risk levels based on environmental and historical data.
- Enable proactive disaster management and resource allocation.
- Provide geospatial visualizations to interpret regional wildfire risks.

## 🗂️ Data Sources
- **Kaggle Wildfire Data:** Historical US wildfire incidents with metadata.  
  [https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires](https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires)

- **(Planned/Optional)**
  - **NASA FIRMS:** Active fire locations and characteristics  
    [https://firms.modaps.eosdis.nasa.gov/download/](https://firms.modaps.eosdis.nasa.gov/download/)
  - **NOAA API:** Historical weather data (temperature, humidity, wind speed, precipitation)  
    [https://www.ncdc.noaa.gov/cdo-web/webservices/v2](https://www.ncdc.noaa.gov/cdo-web/webservices/v2)

## 🧰 Tech Stack
- **Python:** Data processing, modeling
- **Pandas, NumPy:** Data manipulation
- **Scikit-Learn, XGBoost:** Machine learning models
- **Matplotlib, Seaborn:** Visualizations
- **(Optional)** Folium, Plotly: Geospatial visualization
- **Streamlit (Planned):** Interactive dashboard UI

## 🛠️ Methodology
1. Data preparation and cleaning of wildfire datasets.
2. Feature engineering:
   - Geolocation: Latitude, Longitude
   - Temporal: Day of Year, Hour of Day, Season
   - Ownership and State Encoding
3. Label transformation:
   - Converting **FIRE_SIZE_CLASS** into **risk levels**: Low, Medium, High.
4. Model training:
   - Random Forest Classifier
   - XGBoost Classifier
5. Evaluation:
   - Accuracy, Precision, Recall, F1-score
   - Feature importance analysis

## 📊 Deliverables
- Cleaned and processed wildfire dataset.
- Trained wildfire severity risk prediction models.
- Feature importance insights.
- **(Planned)** Interactive dashboard for risk prediction visualization.

## 🚀 Future Enhancements
- Integrate real-time weather data via APIs.
- Include vegetation, drought index, and humidity as features.
- Deploy a **Streamlit dashboard** with real-time location risk queries.
- Add time-series forecasting for seasonal wildfire trends.

## 🗒️ Status
> **Project is a work in progress — actively developing the data pipeline, model, and dashboard components.**

---
## 📁 Project Structure.


```
wildfire-risk-prediction/
│
├── data/                  # Raw and processed data files
│    ├── raw/              # Original data files (e.g., SQLite, CSV)
│    └── processed/        # Cleaned & merged data for modeling
│
├── notebooks/             # Jupyter notebooks for EDA, experiments
│    ├── 01_data_exploration.ipynb
│    ├── 02_feature_engineering.ipynb
│    └── 03_model_training.ipynb
│
├── src/                   # Python modules/scripts
│    ├── data_loader.py    # Load and query data
│    ├── preprocessing.py  # Data cleaning and feature engineering
│    ├── model.py          # Model training and evaluation
│    ├── utils.py          # Helper functions
│    └── visualization.py  # Plots, maps, feature importance
│
├── models/                # Saved models (.joblib, .pkl)
│    └── wildfire_risk_model.joblib
│
├── app/                   # Streamlit dashboard
│    ├── dashboard.py      # Main Streamlit app script
│    └── requirements.txt  # Libraries needed for deployment
│
├── outputs/               # Generated plots, figures, reports
│
├── README.md              # Project overview and instructions
├── requirements.txt       # Project dependencies
├── .gitignore             # Files to ignore in version control
└── LICENSE                # Licensing (optional)
```
---

Feel free to contribute or suggest features by opening an issue or pull request!