import pandas as pd
import numpy as np

def prepare_features(df: pd.DataFrame):
    """
    Perform wildfire-specific feature engineering.
    Returns processed dataframe ready for the ML pipeline.
    """

    # Drop columns with excessive missing values
    drop_cols = [
        'LOCAL_FIRE_REPORT_ID', 'LOCAL_INCIDENT_ID', 'FIRE_CODE', 'FIRE_NAME',
        'ICS_209_INCIDENT_NUMBER', 'ICS_209_NAME', 'MTBS_ID', 'MTBS_FIRE_NAME',
        'COMPLEX_NAME', 'FIPS_CODE', 'FIPS_NAME', 'CONT_DATE', 'CONT_DOY', 
        'CONT_TIME', 'FIRE_DURATION', 'DISCOVERY_HOUR_MISSING', 'CONT_HOUR_MISSING'
    ]
    df = df.drop(columns=drop_cols, errors='ignore')

    # Parse DISCOVERY_HOUR
    def time_to_hour(time_str):
        if pd.isna(time_str):
            return np.nan
        time_str = str(int(time_str)).zfill(4)
        return int(time_str[:2])

    df['DISCOVERY_HOUR'] = df['DISCOVERY_TIME'].apply(time_to_hour)
    df['DISCOVERY_HOUR'].fillna(df['DISCOVERY_HOUR'].median(), inplace=True)

    # Convert DISCOVERY_DATE from Julian to datetime
    df['DISCOVERY_DATE'] = pd.to_datetime(df['DISCOVERY_DATE'], unit='D', origin='julian')
    
    # Extract SEASON
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

    # Log-transform target
    df['FIRE_SIZE'] = np.log1p(df['FIRE_SIZE'])

    # Ensure categorical columns are strings (for OneHotEncoder later)
    categorical_cols = ['STATE', 'STAT_CAUSE_DESCR', 'OWNER_DESCR', 'SEASON', 'CAUSE_SIMPLE']
    for col in categorical_cols:
        df[col] = df[col].astype(str)

    return df
