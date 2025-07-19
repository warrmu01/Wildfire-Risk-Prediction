import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def prepare_features(df: pd.DataFrame):
    """
    Perform feature engineering on the wildfire dataset.
    Returns processed dataframe and the fitted scaler.
    """

    # Drop columns with excessive missing values
    drop_cols = [
        'LOCAL_FIRE_REPORT_ID', 'LOCAL_INCIDENT_ID', 'FIRE_CODE', 'FIRE_NAME',
        'ICS_209_INCIDENT_NUMBER', 'ICS_209_NAME', 'MTBS_ID', 'MTBS_FIRE_NAME',
        'COMPLEX_NAME', 'FIPS_CODE', 'FIPS_NAME'
    ]
    df = df.drop(columns=drop_cols, errors='ignore')

    # Drop rows with missing essential time information
    time_columns = ['DISCOVERY_TIME', 'CONT_DATE', 'CONT_DOY', 'CONT_TIME']
    existing_time_columns = [col for col in time_columns if col in df.columns]
    df = df.dropna(subset=existing_time_columns)

    def time_to_hour(time_str):
        if pd.isna(time_str):
            return np.nan
        time_str = str(int(time_str)).zfill(4)
        return int(time_str[:2])

    df['DISCOVERY_HOUR'] = df['DISCOVERY_TIME'].apply(time_to_hour)
    df['CONT_HOUR'] = df['CONT_TIME'].apply(time_to_hour)

    df['DISCOVERY_DATE'] = pd.to_datetime(df['DISCOVERY_DATE'], unit='D', origin='julian')
    df['CONT_DATE'] = pd.to_datetime(df['CONT_DATE'], unit='D', origin='julian')

    df['FIRE_DURATION'] = (df['CONT_DATE'] - df['DISCOVERY_DATE']).dt.days
    df['FIRE_DURATION'] = df['FIRE_DURATION'].fillna(-1)

    df['DISCOVERY_HOUR_MISSING'] = df['DISCOVERY_HOUR'].isna().astype(int)
    df['CONT_HOUR_MISSING'] = df['CONT_HOUR'].isna().astype(int)

    df['DISCOVERY_HOUR'].fillna(df['DISCOVERY_HOUR'].median(), inplace=True)
    df['CONT_HOUR'].fillna(df['CONT_HOUR'].median(), inplace=True)

    label_cols = ['STATE', 'STAT_CAUSE_DESCR', 'OWNER_DESCR']
    le = LabelEncoder()
    for col in label_cols:
        df[col] = le.fit_transform(df[col].astype(str))

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

    # NEW: Map FIRE_SIZE_CLASS to RISK_LEVEL
    def map_fire_size_class_to_risk(fire_size_class):
        if fire_size_class in ['A', 'B']:
            return 'Low'
        elif fire_size_class in ['C', 'D', 'E']:
            return 'Medium'
        else:
            return 'High'

    df['RISK_LEVEL'] = df['FIRE_SIZE_CLASS'].apply(map_fire_size_class_to_risk)

    # Scale numeric features
    scale_cols = ['LATITUDE', 'LONGITUDE', 'DISCOVERY_DOY', 'DISCOVERY_HOUR',
                  'CONT_DOY', 'CONT_HOUR', 'FIRE_DURATION']
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    return df, scaler
