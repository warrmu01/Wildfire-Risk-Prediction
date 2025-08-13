# Wildfire Risk Prediction Project 

## 📌 Project Overview
This project predicts the **expected severity or risk level** of wildfires in specific US regions based on historical incidents and environmental factors. Using machine learning models, the system forecasts wildfire **risk levels (Low, Medium, High)** based on location, season, and discovery conditions **if a fire occurs.**

## 🎯 Objective
- Predict wildfire severity risk levels to support proactive disaster management.
- Enable decision-makers to allocate firefighting resources efficiently.
- Provide a **Streamlit dashboard** for interactive wildfire risk prediction.

## 🗂️ Data Sources
- **Kaggle Wildfire Dataset:**  
  [188 Million US Wildfires (Kaggle)](https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires)

- **(Planned Integrations)**
  - **NASA FIRMS:** Active fire locations and characteristics  
    [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/download/)
  - **NOAA API:** Historical weather data (temperature, humidity, wind speed, precipitation)  
    [NOAA API](https://www.ncdc.noaa.gov/cdo-web/webservices/v2)

## 🧰 Tech Stack
- **Python:** Data processing & modeling
- **Pandas, NumPy:** Data manipulation
- **Scikit-Learn, XGBoost:** Machine learning models
- **Matplotlib, Seaborn:** Data visualization
- **Streamlit:** Interactive web-based prediction dashboard

## 🛠️ Methodology
1. **Data Preparation:**
   - Data cleaning and handling missing values.
   - Feature engineering (location, season, ownership, cause).
2. **Feature Engineering:**
   - **Geolocation:** Latitude, Longitude
   - **Temporal:** Day of Year, Hour of Day, Season
   - **Categorical:** State, Owner Description, Cause Descriptions
3. **Label Transformation:**
   - Mapping **FIRE_SIZE_CLASS** to risk levels: Low, Medium, High.
4. **Model Training:**
   - Random Forest Classifier
   - XGBoost Classifier
   - SMOTE for balancing classes
5. **Model Evaluation:**
   - Metrics: Accuracy, Precision, Recall, F1-Score
   - Feature importance analysis

## 📊 Deliverables
- Processed wildfire dataset for modeling.
- Trained wildfire risk prediction model.
- **Interactive Streamlit dashboard** for live risk level predictions.
- Feature importance visualization.

## 🚀 Future Enhancements
- Integrate real-time weather data via APIs.
- Add vegetation, drought index, and humidity as features.
- Deploy the Streamlit dashboard to Streamlit Cloud or Hugging Face Spaces.
- Add time-series forecasting for seasonal wildfire trends.

## 🗒️ Status
> ✅ **MVP Completed:** Model trained and dashboard functional for severity prediction.  
> 🔄 Further fine-tuning and feature additions in progress.


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
├── models/                # Saved models (will not be availabel on github because they are too big)
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