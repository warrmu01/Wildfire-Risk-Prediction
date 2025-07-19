# Wildfire Risk Prediction Project 🚧 (Work in Progress)

## 📌 Project Overview
This project aims to predict the likelihood of wildfires occurring in specific regions based on historical fire incidents and weather data. Using machine learning models, the system forecasts wildfire risk levels (Low, Medium, High) and visualizes these predictions on an interactive map.

## 🎯 Objective
- Predict wildfire occurrence risk using weather conditions and historical fire data.
- Enable proactive disaster management and resource allocation.
- Provide geospatial visualizations to easily interpret risk areas.

## 🗂️ Data Sources
- **NASA FIRMS:** Active fire locations and characteristics  
https://firms.modaps.eosdis.nasa.gov/download/
- **NOAA API / Kaggle Dataset:** Historical weather data including temperature, humidity, wind speed, and precipitation.
  - NOAA API: https://www.ncdc.noaa.gov/cdo-web/webservices/v2
  - Kaggle Wildfire Data: https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires

## 🧰 Tech Stack
- **Python:** Data processing, modeling
- **Pandas, NumPy:** Data manipulation
- **Scikit-Learn, XGBoost:** Machine learning models
- **Folium, Plotly:** Geospatial and data visualization
- **Streamlit:** Interactive dashboard UI

## 🛠️ Methodology
1. Data collection and merging of wildfire and weather datasets.
2. Feature engineering (e.g., temperature, humidity, wind speed, precipitation, month).
3. Model training using Random Forest and XGBoost classifiers.
4. Evaluation with precision, recall, F1-score, and ROC-AUC.
5. Visualization of risk predictions on an interactive map.
6. Deployment of a Streamlit dashboard for real-time prediction and visualization.

## 📊 Deliverables
- Trained wildfire risk prediction models.
- Feature importance plots.
- Interactive Streamlit dashboard.
- Geospatial visualizations of predicted risks.

## 🚀 Future Enhancements
- Integrate real-time weather APIs.
- Dynamic user-selected location predictions.
- Time-series forecasting for wildfire trends.
- Deployment to Streamlit Cloud.

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