"""
This module cleans the raw data from dividend Parquet files.
It processes both dividend data and dividend futures data, standardizing formats,
handling dates, and preparing tidy datasets for later analysis.
"""

import pandas as pd
import numpy as np
from settings import config
from pathlib import Path

# Configuration variables from settings
DATA_DIR = config("DATA_DIR")
USE_BBG = config("USE_BBG")
START_DT = config("START_DATE")
PAPER_END_DT = config("PAPER_END_DATE")
CURR_END_DT = config("CURR_END_DATE")


def clean_dividend_data():
    """
    Clean and process the dividend data Parquet file.
    
    Returns:
        pd.DataFrame: Cleaned dividend data with proper datetime index
                      and standardized column names.
    """
    # Read the raw dividend data
    div_data_path = Path(DATA_DIR) / "dividend_data.parquet"
    
    # Check if file exists
    if not div_data_path.exists():
        raise FileNotFoundError(f"Dividend data file not found at {div_data_path}")
    
    # Read the Parquet file
    df_div = pd.read_parquet(div_data_path)
    
    # Make sure the index is a datetime index
    if not pd.api.types.is_datetime64_dtype(df_div.index):
        # Try to handle both cases where index might be date or a separate Date column exists
        if 'Date' in df_div.columns:
            df_div['Date'] = pd.to_datetime(df_div['Date'])
            df_div = df_div.set_index('Date')
        else:
            df_div.index = pd.to_datetime(df_div.index)
    
    # Rename columns to more descriptive names
    df_div = df_div.rename(columns={
        'SPX Index_DIV': 'US_Dividend',
        'SX5E Index_DIV': 'EU_Dividend',
        'NKY Index_DIV': 'JP_Dividend'
    })
    
    # Sort the index
    df_div = df_div.sort_index()
    
    # Handle missing values (if any)
    df_div = df_div.fillna(method='ffill')
    
    return df_div


def clean_dividend_futures_data():
    """
    Clean and process the dividend futures data Parquet file.
    
    Returns:
        pd.DataFrame: Cleaned dividend futures data with proper datetime index
                      and standardized column names.
    """
    # Read the raw dividend futures data
    fut_data_path = Path(DATA_DIR) / "dividend_futures_data.parquet"
    
    # Check if file exists
    if not fut_data_path.exists():
        raise FileNotFoundError(f"Dividend futures data file not found at {fut_data_path}")
    
    # Read the Parquet file
    df_fut = pd.read_parquet(fut_data_path)
    
    # Make sure the index is a datetime index
    if not pd.api.types.is_datetime64_dtype(df_fut.index):
        # Try to handle both cases where index might be date or a separate Date column exists
        if 'Date' in df_fut.columns:
            df_fut['Date'] = pd.to_datetime(df_fut['Date'])
            df_fut = df_fut.set_index('Date')
        else:
            df_fut.index = pd.to_datetime(df_fut.index)
    
    # Rename columns to more descriptive names
    df_fut = df_fut.rename(columns={
        'ASD2 Index': 'US_Div_Future',
        'DED2 Index': 'EU_Div_Future',
        'MND2 Index': 'JP_Div_Future'
    })
    
    # Sort the index
    df_fut = df_fut.sort_index()
    
    # Handle missing values (if any)
    df_fut = df_fut.fillna(method='ffill')
    
    return df_fut


def resample_to_quarterly(df):
    """
    Resample data to quarterly frequency (end of quarter)
    
    Parameters:
        df (DataFrame): Input dataframe with datetime index
        
    Returns:
        DataFrame: Resampled dataframe
    """
    if df.empty:
        return df
    
    # Double-check the index is a datetime index
    if not pd.api.types.is_datetime64_dtype(df.index):
        raise TypeError("DataFrame index must be a datetime index for resampling")
    
    # Resample to end of quarter
    quarterly_df = df.resample('QE').last()
    
    return quarterly_df


def merge_dividend_data():
    """
    Merge the dividend data with dividend futures data into a single DataFrame.
    
    Returns:
        pd.DataFrame: Combined data from both sources
    """
    div_data = clean_dividend_data()
    fut_data = clean_dividend_futures_data()
    
    # Merge on index (date)
    merged_data = pd.merge(div_data, fut_data, 
                          left_index=True, 
                          right_index=True, 
                          how='outer')
    
    # Sort by date
    merged_data = merged_data.sort_index()
    
    return merged_data


def save_clean_data():
    """
    Save all cleaned datasets to the clean data directory.
    """
    # Create clean data directory if it doesn't exist
    clean_dir = Path(DATA_DIR) / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    
    # Get cleaned dataframes
    print("Cleaning dividend data...")
    div_data = clean_dividend_data()
    print("Dividend data index type:", type(div_data.index))
    print("First 5 rows of dividend data:")
    print(div_data.head())

    print("\nCleaning dividend futures data...")
    fut_data = clean_dividend_futures_data()
    print("Dividend futures data index type:", type(fut_data.index))
    
    print("\nMerging data...")
    merged_data = merge_dividend_data()
    print("Merged data index type:", type(merged_data.index))
    print("First 5 rows of merged data:")
    print(merged_data.head())
    
    # Save daily data as parquet files
    print("\nSaving daily data...")
    div_data.to_parquet(clean_dir / "dividend_data_clean.parquet")
    fut_data.to_parquet(clean_dir / "dividend_futures_clean.parquet")
    merged_data.to_parquet(clean_dir / "merged_dividend_data.parquet")
    
    # Create quarterly versions
    print("\nResampling to quarterly data...")
    try:
        div_data_quarterly = resample_to_quarterly(div_data)
        fut_data_quarterly = resample_to_quarterly(fut_data)
        merged_data_quarterly = resample_to_quarterly(merged_data)
        
        # Save quarterly data
        print("\nSaving quarterly data...")
        div_data_quarterly.to_parquet(clean_dir / "dividend_data_clean_quarterly.parquet")
        fut_data_quarterly.to_parquet(clean_dir / "dividend_futures_clean_quarterly.parquet")
        merged_data_quarterly.to_parquet(clean_dir / "merged_dividend_data_quarterly.parquet")
        
        print("\nFirst 5 rows of quarterly merged data:")
        print(merged_data_quarterly.head())
    except Exception as e:
        print(f"Error during resampling: {e}")
        print("Will continue with saving daily data only.")
    
    print("\nAll clean data files saved successfully in Parquet format.")


def main():
    """
    Main function to execute the data cleaning process.
    """
    save_clean_data()
    print("Data cleaning completed.")


if __name__ == "__main__":
    main()