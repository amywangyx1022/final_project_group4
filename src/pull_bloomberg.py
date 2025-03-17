"""
This module loads the S&P 500 index, Nikkei, Eurostoxx index as well as 30 year yields from each from Bloomberg.

If Bloomberg is not available, it will fall back to loading from Excel files.
"""

import pandas as pd
from settings import config
from pathlib import Path
import os
import warnings

END_DATE = config("CURR_END_DATE")
DATA_DIR = config("DATA_DIR")
START_DATE = config("START_DATE")
BASE_DIR = config("BASE_DIR")
USE_BBG = config("USE_BBG")


def pull_bbg_data(start_date, end_date):
    """
    Pull data from Bloomberg for indices and 30-year yields
    
    Parameters:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        DataFrame: Bloomberg data for indices and yields
    """    
    from xbbg import blp

    df = blp.bdh(['SPX Index','SX5E Index','NKY Index','USGG30YR Index','GDBR30 Index','GJGB30 Index'],
                "PX_LAST", start_date, end_date)
    df.columns = df.columns.droplevel(1)
    df.index.name = 'Date'

    return df

def pull_bbg_dividend_data(start_date, end_date):
    """
    Pull dividend data from Bloomberg
    
    Parameters:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        DataFrame: Bloomberg dividend data
    """
    from xbbg import blp

    # For S&P 500, Euro Stoxx 50, and Nikkei 225
    # Using BEST_DIV_YLD to get the dividend yield, which can be used with the index value
    # to calculate the dividend amount
    dividend_tickers = ['SPX Index', 'SX5E Index', 'NKY Index']
    
    # First pull dividend yields
    div_yield_df = blp.bdh(dividend_tickers, "EQY_DVD_YLD_12M", start_date, end_date)
    div_yield_df.columns = div_yield_df.columns.droplevel(1)
    div_yield_df.index.name = 'Date'
    
    # Pull index prices to calculate dividend values
    index_df = pull_bbg_data(start_date, end_date)
    
    # Create a new dataframe for dividend values
    div_df = pd.DataFrame(index=div_yield_df.index)
    
    # Calculate dividend values from yields and prices
    for ticker in dividend_tickers:
        # Convert yield from percentage to decimal
        div_yield = div_yield_df[ticker] / 100.0
        index_price = index_df[ticker]
        
        # Calculate dividend value (yield * price)
        # Common practice is to use trailing 12-month dividends
        div_df[f"{ticker}_DIV"] = div_yield * index_price
    
    return div_df

def pull_bbg_dividend_futures(start_date, end_date):
    """
    Pull dividend futures data (ASD2 Index and DED1 Index) from Bloomberg with quarterly dates
    
    Parameters:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        DataFrame: Bloomberg dividend futures data resampled to quarterly dates
    """
    from xbbg import blp
    
    # Define the dividend futures tickers
    futures_tickers = ['ASD2 Index', 'DED2 Index', 'MND2 Index']
    
    # Pull the daily data first
    df = blp.bdh(futures_tickers, "PX_LAST", start_date, end_date)
    df.columns = df.columns.droplevel(1)  # Remove the 'PX_LAST' level from column names
    df.index.name = 'Date'
    
    # Resample to end-of-quarter dates
    # First, ensure the index is datetime
    df.index = pd.to_datetime(df.index)
    
    return df

def load_csv_dividend_data():
    """
    Load dividend data from CSV file as a fallback when Bloomberg is not available
    
    Returns:
        DataFrame: Dividend data from CSV
    """
    csv_path = Path(DATA_DIR) / "dividend_data.csv"
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    print(f"Loading dividend data from {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
    
    return df


def load_csv_dividend_futures_data():
    """
    Load dividend futures data from CSV file as a fallback when Bloomberg is not available
    
    Returns:
        DataFrame: Dividend futures data from CSV
    """
    csv_path = Path(DATA_DIR) / "dividend_future_data.csv"
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    print(f"Loading dividend futures data from {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
    
    return df


if __name__ == "__main__":
    print(f"Starting data collection process. USE_BBG set to: {USE_BBG}")
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Initialize variables for data
    div_df = None
    div_futures_df = None
    
    # Try to get data from Bloomberg if enabled
    if USE_BBG:
        try:
            print("Trying to pull data from Bloomberg...")
            
            # Pull dividend data
            div_df = pull_bbg_dividend_data(START_DATE, END_DATE)
            
            # Pull dividend futures data
            div_futures_df = pull_bbg_dividend_futures(START_DATE, END_DATE)
            
            print("Successfully pulled data from Bloomberg!")
            
            # Save the data to both parquet and CSV formats
            div_df.to_parquet(Path(DATA_DIR) / "bloomberg_dividend_data.parquet")
            div_df.to_csv(Path(DATA_DIR) / "dividend_data.csv")
            print("Saved dividend data")
            
            div_futures_df.to_parquet(Path(DATA_DIR) / "bloomberg_dividend_futures_data.parquet")
            div_futures_df.to_csv(Path(DATA_DIR) / "dividend_future_data.csv")
            print("Saved dividend futures data")
            
        except Exception as e:
            print(f"Error pulling data from Bloomberg: {e}")
            print("Will try to load from CSV files instead.")
            USE_BBG = False
    
    # If Bloomberg failed or is disabled, try CSV files
    if not USE_BBG:
        try:
            print("Trying to load data from CSV files...")
            
            # Try to load each dataset from CSV if not already loaded
            if div_df is None:
                try:
                    div_df = load_csv_dividend_data()
                    div_df.to_parquet(Path(DATA_DIR) / "dividend_data.parquet")
                    print("Save dividend data")
                except Exception as e:
                    print(f"Error saving dividend data: {e}")
            
            if div_futures_df is None:
                try:
                    div_futures_df = load_csv_dividend_futures_data()
                    div_futures_path = Path(DATA_DIR) / "dividend_futures_data.parquet"
                    div_futures_df.to_parquet(div_futures_path)
                    print("Saved dividend future data")
                except Exception as e:
                    print(f"Error saving dividend futures data")
            
        except Exception as e:
            print(f"Error loading data from CSV files: {e}")
    

        
    print("Data collection process completed!")



