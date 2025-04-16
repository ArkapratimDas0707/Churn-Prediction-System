# scripts/load_data.py

import pandas as pd
import os

def load_data(file_path='..Data/Raw/Telco_churn.csv'):
    """
    Loads the Telco Churn data from the specified path.

    Parameters:
        file_path (str): Path to the raw dataset.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")
    
    data = pd.read_csv(file_path)
    return data
