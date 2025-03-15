"""
This module cleans the raw data from Bloomberg or Excel files.
We save a clean version of the data in the data folder for record keeping.
"""

import pandas as pd
import numpy as np
from settings import config
from pathlib import Path

DATA_DIR = config("DATA_DIR")
USE_BBG = config("USE_BBG")
START_DT = config("START_DATE")
PAPER_END_DT = config("PAPER_END_DATE")
CURR_END_DT = config("CURR_END_DATE")

def clean_index_data(end_date, data_dir=DATA_DIR):
    """
    Clean and prepare index data from Bloomberg or Excel
    
    Parameters:
        end_date (str): End date to filter data
        data_dir (Path): Directory containing the data file
        
    Returns:
        DataFrame: Cleaned index data
    """
    print(f"Cleaning index data through {end_date}...")
    
    # Determine which file to load based on USE_BBG
    if USE_BBG:
        path = Path(data_dir) / "bloomberg_index_data.parquet"
    else:
        path = Path(data_dir) / "excel_index_data.parquet"
    
    if not path.exists():
        # Try the alternative if the primary file doesn't exist
        alternative_path = Path(data_dir) / ("excel_index_data.parquet" if USE_BBG else "bloomberg_index_data.parquet")
        if alternative_path.exists():
            path = alternative_path
            print(f"Using alternative index data source: {path}")
        else:
            raise FileNotFoundError(f"Index data file not found at {path} or {alternative_path}")
    
    # Load the data
    df = pd.read_parquet(path)
    
    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)
    
    # Filter for date range
    df = df.loc[START_DT:end_date]
    
    # Standard column names
    column_mapping = {
        'SPX Index': 'SP500',
        'SX5E Index': 'EUROSTOXX',
        'NKY Index': 'NIKKEI',
        'USGG30YR Index': 'US_30Y_YIELD',
        'GDBR30 Index': 'EU_30Y_YIELD',
        'GJGB30 Index': 'JP_30Y_YIELD'
    }
    
    # Apply column mapping if needed
    renamed_columns = {}
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            renamed_columns[old_col] = new_col
    
    if renamed_columns:
        df = df.rename(columns=renamed_columns)
    
    # Sort by date
    df = df.sort_index()
    
    # Handle any missing values
    df = df.ffill().bfill()
    
    return df

def clean_dividend_data(end_date, data_dir=DATA_DIR):
    """
    Clean and prepare dividend data from Bloomberg or Excel
    
    Parameters:
        end_date (str): End date to filter data
        data_dir (Path): Directory containing the data file
        
    Returns:
        DataFrame: Cleaned dividend data
    """
    print(f"Cleaning dividend data through {end_date}...")
    
    # Determine which file to load based on USE_BBG
    if USE_BBG:
        path = Path(data_dir) / "bloomberg_dividend_data.parquet"
    else:
        path = Path(data_dir) / "excel_dividend_data.parquet"
    
    if not path.exists():
        # Try the alternative if the primary file doesn't exist
        alternative_path = Path(data_dir) / ("excel_dividend_data.parquet" if USE_BBG else "bloomberg_dividend_data.parquet")
        if alternative_path.exists():
            path = alternative_path
            print(f"Using alternative dividend data source: {path}")
        else:
            raise FileNotFoundError(f"Dividend data file not found at {path} or {alternative_path}")
    
    # Load the data
    df = pd.read_parquet(path)
    
    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)
    
    # Filter for date range
    df = df.loc[START_DT:end_date]
    
    # Standard column names
    column_mapping = {
        'SPX Index_DIV': 'SP500_DIV',
        'SX5E Index_DIV': 'EUROSTOXX_DIV',
        'NKY Index_DIV': 'NIKKEI_DIV'
    }
    
    # Apply column mapping if needed
    renamed_columns = {}
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            renamed_columns[old_col] = new_col
    
    if renamed_columns:
        df = df.rename(columns=renamed_columns)
    
    # Also check for other possible column names
    for col in df.columns:
        col_lower = str(col).lower()
        if 's&p' in col_lower or 'sp500' in col_lower or 'sp 500' in col_lower:
            df = df.rename(columns={col: 'SP500_DIV'})
        elif 'euro' in col_lower or 'stoxx' in col_lower:
            df = df.rename(columns={col: 'EUROSTOXX_DIV'})
        elif 'nikkei' in col_lower or 'japan' in col_lower:
            df = df.rename(columns={col: 'NIKKEI_DIV'})
    
    # Sort by date
    df = df.sort_index()
    
    # Handle any missing values
    df = df.ffill().bfill()
    
    return df

def format_df(df, all_col=True):
    """
    Format dataframe values to 3 decimal places
    
    Parameters:
        df (DataFrame): Input dataframe
        all_col (bool): Whether to format all columns or just numeric ones
        
    Returns:
        DataFrame: Formatted dataframe
    """
    # Create a copy to avoid modifying the original
    formatted_df = df.copy()
    
    if all_col:
        formatted_df = formatted_df.applymap(lambda x: '{:.3f}'.format(x) if isinstance(x, (int, float)) else x)
    else:
        # Format only numeric columns (all except first column if it's a date)
        formatted_df.iloc[:, 1:] = formatted_df.iloc[:, 1:].applymap(
            lambda x: '{:.3f}'.format(x) if isinstance(x, (int, float)) else x
        )
    
    return formatted_df

def calculate_yields(price_data, dividend_data):
    """
    Calculate dividend yields from price and dividend data
    
    Parameters:
        price_data (DataFrame): Index price data
        dividend_data (DataFrame): Dividend data
        
    Returns:
        DataFrame: Dividend yields
    """
    print("Calculating dividend yields...")
    
    # Create a new DataFrame with index from price data
    result = pd.DataFrame(index=price_data.index)
    
    # Ensure dividend data is aligned with price data
    reindexed_div = dividend_data.reindex(index=price_data.index, method='ffill')
    
    # Calculate dividend yields
    for index_name in ['SP500', 'EUROSTOXX', 'NIKKEI']:
        div_col = f"{index_name}_DIV"
        if div_col in reindexed_div.columns and index_name in price_data.columns:
            result[f"{index_name}_YIELD"] = reindexed_div[div_col] / price_data[index_name]
    
    return result

def combine_data(price_data, dividend_data, yield_data):
    """
    Combine all data into a single DataFrame
    
    Parameters:
        price_data (DataFrame): Index price data
        dividend_data (DataFrame): Dividend data
        yield_data (DataFrame): Yield data
        
    Returns:
        DataFrame: Combined data
    """
    print("Combining all data...")
    
    # Create a new DataFrame with all dates
    all_dates = price_data.index.union(dividend_data.index).union(yield_data.index)
    combined_df = pd.DataFrame(index=all_dates)
    combined_df.index = pd.to_datetime(combined_df.index)
    combined_df = combined_df.sort_index()
    
    # Add price data
    for col in price_data.columns:
        combined_df[col] = price_data[col].reindex(index=combined_df.index, method='ffill')
    
    # Add dividend data
    for col in dividend_data.columns:
        combined_df[col] = dividend_data[col].reindex(index=combined_df.index, method='ffill')
    
    # Add yield data
    for col in yield_data.columns:
        combined_df[col] = yield_data[col].reindex(index=combined_df.index, method='ffill')
    
    return combined_df

if __name__ == "__main__":
    # Clean index data
    price_df = clean_index_data(PAPER_END_DT, data_dir=DATA_DIR)
    
    # Save cleaned index data
    price_path = Path(DATA_DIR) / "cleaned_index_data.parquet"
    price_df.to_parquet(price_path)
    print(f"Saved cleaned index data to {price_path}")
    
    # Clean dividend data
    try:
        dividend_df = clean_dividend_data(PAPER_END_DT, data_dir=DATA_DIR)
        
        # Save cleaned dividend data
        div_path = Path(DATA_DIR) / "cleaned_dividend_data.parquet"
        dividend_df.to_parquet(div_path)
        print(f"Saved cleaned dividend data to {div_path}")
        
        # Calculate yields
        yields_df = calculate_yields(price_df, dividend_df)
        
        # Save yield data
        yield_path = Path(DATA_DIR) / "calculated_yields.parquet"
        yields_df.to_parquet(yield_path)
        print(f"Saved calculated yields to {yield_path}")
        
        # Combine all data
        combined_df = combine_data(price_df, dividend_df, yields_df)
        
        # Save combined data
        combined_path = Path(DATA_DIR) / "combined_data.parquet"
        combined_df.to_parquet(combined_path)
        print(f"Saved combined data to {combined_path}")
        
        # Also save an Excel version for easy inspection
        excel_path = Path(DATA_DIR) / "combined_data.xlsx"
        combined_df.to_excel(excel_path)
        print(f"Saved combined data to Excel: {excel_path}")
        
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Skipping dividend data processing.")
    
    # Check if we're also processing up to current date
    if PAPER_END_DT != CURR_END_DT:
        print(f"\nAlso processing data up to current end date: {CURR_END_DT}")
        
        # Clean index data up to current date
        current_price_df = clean_index_data(CURR_END_DT, data_dir=DATA_DIR)
        current_price_path = Path(DATA_DIR) / "cleaned_current_index_data.parquet"
        current_price_df.to_parquet(current_price_path)
        print(f"Saved cleaned current index data to {current_price_path}")
        
        try:
            # Clean dividend data up to current date
            current_dividend_df = clean_dividend_data(CURR_END_DT, data_dir=DATA_DIR)
            current_div_path = Path(DATA_DIR) / "cleaned_current_dividend_data.parquet"
            current_dividend_df.to_parquet(current_div_path)
            print(f"Saved cleaned current dividend data to {current_div_path}")
            
            # Calculate yields up to current date
            current_yields_df = calculate_yields(current_price_df, current_dividend_df)
            current_yield_path = Path(DATA_DIR) / "calculated_current_yields.parquet"
            current_yields_df.to_parquet(current_yield_path)
            print(f"Saved calculated current yields to {current_yield_path}")
            
            # Combine all current data
            current_combined_df = combine_data(current_price_df, current_dividend_df, current_yields_df)
            current_combined_path = Path(DATA_DIR) / "combined_current_data.parquet"
            current_combined_df.to_parquet(current_combined_path)
            print(f"Saved combined current data to {current_combined_path}")
            
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Skipping current dividend data processing.")