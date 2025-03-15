
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
    div_yield_df = blp.bdh(dividend_tickers, "BEST_DIV_YLD", start_date, end_date)
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

def load_bbg_excel(data_dir=BASE_DIR):
    """
    Load index data from Excel file as a fallback when Bloomberg is not available
    
    Parameters:
        data_dir (Path): Base directory path
        
    Returns:
        DataFrame: Index and yield data from Excel
    """
    # load index data from excel
    path = data_dir / 'data_manual' / 'index_prices.xlsx'
    print(f"Loading index data from {path}...")
    
    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel file not found at {path}")
    
    try:
        df = pd.read_excel(path, sheet_name='index')
        # Skip header row if present
        if isinstance(df.columns[0], str) and 'date' in df.columns[0].lower():
            # Headers are already column names
            pass
        else:
            # First row might contain headers
            df = df.iloc[1:]
        
        # Find date column
        date_col = None
        for col in df.columns:
            if 'date' in str(col).lower() or 'unnamed' in str(col).lower():
                date_col = col
                break
        
        if date_col:
            df = df.rename(columns={date_col: 'Date'})
        else:
            # Assume first column is date if no date column found
            df = df.rename(columns={df.columns[0]: 'Date'})
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Set the datetime column as index
        df.set_index('Date', inplace=True)
        
        # Rename columns to match Bloomberg format
        column_mapping = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if 's&p' in col_lower or 'sp500' in col_lower or 'sp 500' in col_lower:
                column_mapping[col] = 'SPX Index'
            elif 'euro' in col_lower or 'stoxx' in col_lower:
                column_mapping[col] = 'SX5E Index'
            elif 'nikkei' in col_lower or 'japan' in col_lower:
                column_mapping[col] = 'NKY Index'
            elif 'us' in col_lower and 'yield' in col_lower:
                column_mapping[col] = 'USGG30YR Index'
            elif ('germany' in col_lower or 'eu' in col_lower) and 'yield' in col_lower:
                column_mapping[col] = 'GDBR30 Index'
            elif 'japan' in col_lower and 'yield' in col_lower:
                column_mapping[col] = 'GJGB30 Index'
        
        # Apply renaming if mapping exists
        if column_mapping:
            df = df.rename(columns=column_mapping)
            
        # Make sure we have the required columns
        required_cols = ['SPX Index', 'SX5E Index', 'NKY Index']
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: Required column '{col}' not found in Excel data")
        
        return df
        
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        raise

def load_dividend_excel(data_dir=BASE_DIR):
    """
    Load dividend data from Excel file as a fallback when Bloomberg is not available
    
    Parameters:
        data_dir (Path): Base directory path
        
    Returns:
        DataFrame: Dividend data from Excel
    """
    path = data_dir / 'data_manual' / 'equity_dividend.xlsx'
    print(f"Loading dividend data from {path}...")
    
    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel file not found at {path}")
    
    try:
        # Try different sheet names
        try:
            df = pd.read_excel(path, sheet_name='dividend')
        except:
            try:
                df = pd.read_excel(path, sheet_name='Sheet1')
            except:
                df = pd.read_excel(path)  # Try default sheet
    except Exception as e:
        raise ValueError(f"Error loading dividend data: {e}")
    
    # Process the DataFrame
    # Skip the first row if it's a header
    if df.iloc[0].isna().any() or df.iloc[0].astype(str).str.contains('Date', case=False).any():
        df = df.iloc[1:]
    
    # Find date column
    date_col = None
    for col in df.columns:
        if 'date' in str(col).lower() or df[col].dtype == 'datetime64[ns]':
            date_col = col
            break
    
    if not date_col:
        # Try the first column
        date_col = df.columns[0]
        print(f"Warning: No date column identified. Using first column: '{date_col}'")
    
    # Rename date column and convert to datetime
    df = df.rename(columns={date_col: 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Set the datetime column as index
    df.set_index('Date', inplace=True)
    
    # Rename columns to match Bloomberg format
    column_mapping = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if 's&p' in col_lower or 'sp500' in col_lower or 'sp 500' in col_lower:
            column_mapping[col] = 'SPX Index_DIV'
        elif 'euro' in col_lower or 'stoxx' in col_lower:
            column_mapping[col] = 'SX5E Index_DIV'
        elif 'nikkei' in col_lower or 'japan' in col_lower:
            column_mapping[col] = 'NKY Index_DIV'
    
    # Apply renaming if mapping exists
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    return df

if __name__ == "__main__":
    print(f"Starting data collection process. USE_BBG set to: {USE_BBG}")
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Initialize variables for index and dividend data
    index_df = None
    div_df = None
    
    # Try to get data from Bloomberg if enabled
    if USE_BBG:
        try:
            print("Trying to pull data from Bloomberg...")
            # Pull index data
            index_df = pull_bbg_data(START_DATE, END_DATE)
            
            # Pull dividend data
            div_df = pull_bbg_dividend_data(START_DATE, END_DATE)
            
            print("Successfully pulled data from Bloomberg!")
            
        except Exception as e:
            print(f"Error pulling data from Bloomberg: {e}")
            print("Will try to load from Excel files instead.")
            USE_BBG = False
    
    # If Bloomberg failed or is disabled, try Excel files
    if not USE_BBG or index_df is None or div_df is None:
        try:
            print("Trying to load data from Excel files...")
            
            # Load index data from Excel
            if index_df is None:
                index_df = load_bbg_excel(BASE_DIR)
            
            # Load dividend data from Excel
            if div_df is None:
                div_df = load_dividend_excel(BASE_DIR)
                
            print("Successfully loaded data from Excel files!")
            
        except Exception as e:
            print(f"Error loading data from Excel: {e}")
            raise
    
    # Save the index data
    if index_df is not None:
        if USE_BBG:
            index_path = Path(DATA_DIR) / "bloomberg_index_data.parquet"
        else:
            index_path = Path(DATA_DIR) / "excel_index_data.parquet"
            
        index_df.to_parquet(index_path)
        print(f"Saved index data to {index_path}")
        
        # Also save as CSV for easy viewing
        csv_path = Path(DATA_DIR) / "index_data.csv"
        index_df.to_csv(csv_path)
        print(f"Saved index data to CSV: {csv_path}")
    
    # Save the dividend data
    if div_df is not None:
        if USE_BBG:
            div_path = Path(DATA_DIR) / "bloomberg_dividend_data.parquet"
        else:
            div_path = Path(DATA_DIR) / "excel_dividend_data.parquet"
            
        div_df.to_parquet(div_path)
        print(f"Saved dividend data to {div_path}")
        
        # Also save as CSV for easy viewing
        csv_path = Path(DATA_DIR) / "dividend_data.csv"
        div_df.to_csv(csv_path)
        print(f"Saved dividend data to CSV: {csv_path}")
    
    print("Data collection process completed!")