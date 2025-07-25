import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def prepare_features(df: pd.DataFrame):
    """
    Perform feature engineering on the wildfire dataset.
    Returns processed dataframe, scaler, and label encoders.
    """

    # Drop columns with excessive missing values
    drop_cols = [
        'LOCAL_FIRE_REPORT_ID', 'LOCAL_INCIDENT_ID', 'FIRE_CODE', 'FIRE_NAME',
        'ICS_209_INCIDENT_NUMBER', 'ICS_209_NAME', 'MTBS_ID', 'MTBS_FIRE_NAME',
        'COMPLEX_NAME', 'FIPS_CODE', 'FIPS_NAME', 'CONT_DATE', 'CONT_DOY', 
        'CONT_TIME', 'FIRE_DURATION', 'DISCOVERY_HOUR_MISSING', 'CONT_HOUR_MISSING'
    ]
    df = df.drop(columns=drop_cols, errors='ignore')

    # Parse times
    def time_to_hour(time_str):
        if pd.isna(time_str):
            return np.nan
        time_str = str(int(time_str)).zfill(4)
        return int(time_str[:2])

    df['DISCOVERY_HOUR'] = df['DISCOVERY_TIME'].apply(time_to_hour)
    df['DISCOVERY_HOUR'].fillna(df['DISCOVERY_HOUR'].median(), inplace=True)

    df['DISCOVERY_DATE'] = pd.to_datetime(df['DISCOVERY_DATE'], unit='D', origin='julian')
    
    # Season extraction
    def get_season(date):
        if pd.isnull(date):
            return 'Unknown'
        month = date.month
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    df['SEASON'] = df['DISCOVERY_DATE'].apply(get_season)

    # Simplify cause
    human_causes = ['Arson', 'Debris Burning', 'Equipment Use', 'Smoking', 'Campfire', 'Children', 'Fireworks']
    natural_causes = ['Lightning']

    def simplify_cause(cause):
        if cause in human_causes:
            return 'Human'
        elif cause in natural_causes:
            return 'Natural'
        else:
            return 'Unknown'

    df['CAUSE_SIMPLE'] = df['STAT_CAUSE_DESCR'].apply(simplify_cause)

    # # Risk level from FIRE_SIZE_CLASS
    # def map_fire_size_class_to_risk(fire_size_class):
    #     if fire_size_class in ['A', 'B']:
    #         return 'Low'
    #     elif fire_size_class in ['C', 'D', 'E']:
    #         return 'Medium'
    #     else:
    #         return 'High'

    # df['RISK_LEVEL'] = df['FIRE_SIZE_CLASS'].apply(map_fire_size_class_to_risk)

    df['FIRE_SIZE'] = np.log1p(df['FIRE_SIZE'])  # target = 'FIRE_SIZE_LOG'

    # Label encode categorical features
    label_cols = ['STATE', 'STAT_CAUSE_DESCR', 'OWNER_DESCR', 'SEASON', 'CAUSE_SIMPLE']
    label_encoders = {}
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Scaling numerical features
    # scale_cols = ['LATITUDE', 'LONGITUDE', 'DISCOVERY_DOY', 'DISCOVERY_HOUR']
    # scaler = StandardScaler()
    # df[scale_cols] = scaler.fit_transform(df[scale_cols])

    return df, label_encoders
