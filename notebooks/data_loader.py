# data_loader.py

import sqlite3
import pandas as pd

def load_fire_data(db_path: str) -> pd.DataFrame:
    """
    Loads the Fires data from the SQLite database.

    Args:
        db_path (str): Path to the SQLite database file.

    Returns:
        pd.DataFrame: DataFrame containing the Fires data.
    """
    conn = sqlite3.connect(db_path)
    
    # You can replace 'Fires' with the actual table name
    query = "SELECT * FROM Fires"
    df = pd.read_sql_query(query, conn)

    conn.close()
    return df

if __name__ == "__main__":
    # For testing
    db_path = 'data/FPA_FOD_20170508.sqlite'
    fire_data = load_fire_data(db_path)
    print(fire_data.head())