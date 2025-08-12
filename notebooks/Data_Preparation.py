import pandas as pd
import numpy as np

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wildfire-specific feature engineering.
    Returns processed dataframe ready for the ML pipeline.
    """

    # 1) Drop rarely useful / high-missing columns
    drop_cols = [
        'LOCAL_FIRE_REPORT_ID','LOCAL_INCIDENT_ID','FIRE_CODE','FIRE_NAME',
        'ICS_209_INCIDENT_NUMBER','ICS_209_NAME','MTBS_ID','MTBS_FIRE_NAME',
        'COMPLEX_NAME','FIPS_CODE','FIPS_NAME','CONT_DATE','CONT_DOY',
        'CONT_TIME','FIRE_DURATION','DISCOVERY_HOUR_MISSING','CONT_HOUR_MISSING'
    ]
    df = df.drop(columns=drop_cols, errors='ignore')

    # 2) Parse discovery hour safely
    def time_to_hour(x):
        if pd.isna(x):
            return np.nan
        try:
            # handles 0, 30, 930, 1234, 5.0, '5', etc.
            s = str(int(float(x))).zfill(4)
            hh = int(s[:2])
            # Clamp to [0, 23]
            if hh < 0: hh = 0
            if hh > 23: hh = 23
            return hh
        except Exception:
            return np.nan

    if 'DISCOVERY_TIME' in df.columns:
        df['DISCOVERY_HOUR'] = df['DISCOVERY_TIME'].apply(time_to_hour)
        df['DISCOVERY_HOUR'] = df['DISCOVERY_HOUR'].fillna(df['DISCOVERY_HOUR'].median())
    else:
        # if column not present, default to median-ish hour
        df['DISCOVERY_HOUR'] = 12

    # 3) Parse discovery date from Julian and derive DOY
    # Coerce errors to NaT so season logic can handle it
    if 'DISCOVERY_DATE' in df.columns:
        df['DISCOVERY_DATE'] = pd.to_datetime(df['DISCOVERY_DATE'], unit='D', origin='julian', errors='coerce')
        # Ensure DISCOVERY_DOY exists
        df['DISCOVERY_DOY'] = df['DISCOVERY_DATE'].dt.dayofyear
        # If all NaN (rare), backfill with median DOY
        if df['DISCOVERY_DOY'].isna().all():
            df['DISCOVERY_DOY'] = 183  # mid-year fallback
        else:
            df['DISCOVERY_DOY'] = df['DISCOVERY_DOY'].fillna(df['DISCOVERY_DOY'].median())
    else:
        # Fallback if the raw column isn't there
        df['DISCOVERY_DATE'] = pd.NaT
        df['DISCOVERY_DOY'] = 183

    # 4) Season extraction
    def get_season(date):
        if pd.isna(date):
            return 'Unknown'
        m = int(date.month)
        if m in (12, 1, 2):   return 'Winter'
        if m in (3, 4, 5):    return 'Spring'
        if m in (6, 7, 8):    return 'Summer'
        return 'Fall'

    df['SEASON'] = df['DISCOVERY_DATE'].apply(get_season)

    # 5) Simplify cause
    human_causes = {'Arson','Debris Burning','Equipment Use','Smoking','Campfire','Children','Fireworks'}
    natural_causes = {'Lightning'}

    def simplify_cause(cause):
        if pd.isna(cause):
            return 'Unknown'
        c = str(cause)
        if c in human_causes:   return 'Human'
        if c in natural_causes: return 'Natural'
        return 'Unknown'

    df['CAUSE_SIMPLE'] = df.get('STAT_CAUSE_DESCR', pd.Series(index=df.index, dtype=object)).apply(simplify_cause)

    # 6) Target: log-transform (drop bad rows first)
    df = df[pd.to_numeric(df['FIRE_SIZE'], errors='coerce').notna()].copy()
    df['FIRE_SIZE'] = df['FIRE_SIZE'].clip(lower=0)  # guard tiny negatives
    df['FIRE_SIZE'] = np.log1p(df['FIRE_SIZE'])

    # 7) Ensure categoricals are clean strings (works for OneHot or Ordinal later)
    categorical_cols = ['STATE','STAT_CAUSE_DESCR','OWNER_DESCR','SEASON','CAUSE_SIMPLE']
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = 'Unknown'
        df[col] = df[col].fillna('Unknown').astype(str)

    # 8) Basic numeric hygiene for key numerics
    for col in ['LATITUDE','LONGITUDE','DISCOVERY_DOY','DISCOVERY_HOUR']:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Fill with median to avoid dropping rows
        df[col] = df[col].fillna(df[col].median())

    return df
