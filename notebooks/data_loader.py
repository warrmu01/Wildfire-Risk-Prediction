# data_loader.py

import sqlite3
import pandas as pd

def load_wildfire_data(db_path='data/FPA_FOD_20170508.sqlite'):
    """
    Load wildfire data from SQLite database.
    
    Args:
        db_path (str): Path to the SQLite database.

    Returns:
        pd.DataFrame: Loaded fires data.
    """
    conn = sqlite3.connect(db_path)
    fires_df = pd.read_sql('SELECT * FROM Fires', conn)
    conn.close()
    return fires_df

if __name__ == "__main__":
    df = load_wildfire_data()
    print("Sample data:")
    print(df.head())
