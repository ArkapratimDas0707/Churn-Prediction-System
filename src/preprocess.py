# scripts/preprocess.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the Telco dataset:
    - Handles whitespace as missing values
    - Converts appropriate columns to numeric
    - Optimizes data types
    - Converts object columns to category where appropriate

    Parameters:
        data(pd.DataFrame): Raw Telco dataframe.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
   
    # Fix TotalCharges: convert to float 
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

    # Drop rows with missing values (11 from TotalCharges)
    data.dropna(inplace=True).reset_index()

    # Convert object columns to 'category' dtype
    for col in data.select_dtypes(include='object').columns:
        data[col] = data[col].astype('category')

    # Convert int and float to optimized types
    for col in data.select_dtypes(include='int').columns:
        data[col] = data[col].astype(np.int8)
    
    for col in data.select_dtypes(include='float').columns:
        data[col] = data[col].astype(np.float32)

    return data

def combine_streaming_services(data: pd.DataFrame) -> pd.DataFrame:
    """
    Combines 'StreamingTV' and 'StreamingMovies' into one feature

    Parameters:
        data (pd.DataFrame): Cleaned DataFrame.

    Returns:
        pd.DataFrame: Updated DataFrame.
    """
    streaming_cols = ['StreamingTV', 'StreamingMovies']
    data['StreamingServices'] = (
        data[streaming_cols]
        .replace({'Yes': 1, 'No': 0, 'No internet service': 0})
        .sum(axis=1)
        .map(lambda x: 'None' if x == 0 else 'One' if x == 1 else 'Both')
    )

    data.drop(columns=streaming_cols, inplace=True)
    return data

def combine_support_services(data: pd.DataFrame) -> pd.DataFrame:
    """
    Combines OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport into single feature

    Parameters:
        data (pd.DataFrame): Cleaned DataFrame.

    Returns:
        pd.DataFrame: Updated DataFrame.
    """
    support_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    data['SupportServices'] = (
        data[support_cols]
        .replace({'Yes': 1, 'No': 0, 'No internet service': 0})
        .sum(axis=1)
        .map(lambda x: 'None' if x == 0 else 'Some' if x <= 2 else 'Most/All')
    )

    data.drop(columns=support_cols, inplace=True)
    return data

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper function to apply all feature engineering steps.

    Parameters:
        data (pd.DataFrame): Cleaned DataFrame.

    Returns:
        pd.DataFrame: Engineered DataFrame.
    """
    data = combine_streaming_services(data)
    data = combine_support_services(data)
    return data

def encode_categoricals(data: pd.DataFrame) -> pd.DataFrame:

    ''''
    One-Hot encodes categorical columns to be used in ML training

    '''

    categorical_cols = data.select_dtypes(include='object').drop(columns='Churn').columns
    encoder = OneHotEncoder(drop='first', sparse=False)

    encoded_array = encoder.fit_transform(data[categorical_cols])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols), index=data.index)

    # Drop original and concat encoded
    data = data.drop(columns=categorical_cols)
    data = pd.concat([data, encoded_df], axis=1)
    return data

def preprocess_input(data: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline to be used in the Flask API.

    Parameters:
        data (pd.DataFrame): Raw user input.

    Returns:
        pd.DataFrame: Fully preprocessed input for prediction.
    """
    data = clean_data(data)
    data = engineer_features(data)
    data = encode_categoricals(data)
    return data

