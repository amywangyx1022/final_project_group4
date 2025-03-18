"""
Unit tests for the clean_data.py module using pytest.

These tests verify that the data cleaning functions correctly process
the raw dividend data and dividend futures data.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import os
from settings import config
from clean_data import (
    clean_dividend_data,
    clean_dividend_futures_data,
    resample_to_quarterly,
    merge_dividend_data,
    clean_index_data
)

# Get configuration values
DATA_DIR = config("DATA_DIR")
START_DATE = config("START_DATE")
PAPER_END_DATE = config("PAPER_END_DATE")

@pytest.fixture
def setup_test_data_dir():
    """Create a temporary test data directory with sample data files."""
    # Create test directory
    test_dir = Path("test_data_dir")
    test_dir.mkdir(exist_ok=True)
    
    # Create sample dividend data
    sample_dates = pd.date_range(start='2020-01-01', end='2020-03-31', freq='B')
    div_data = pd.DataFrame({
        'SPX Index_DIV': np.random.uniform(50, 70, len(sample_dates)),
        'SX5E Index_DIV': np.random.uniform(100, 120, len(sample_dates)),
        'NKY Index_DIV': np.random.uniform(300, 400, len(sample_dates))
    }, index=sample_dates)
    div_data.index.name = 'Date'
    
    # Create sample futures data
    futures_data = pd.DataFrame({
        'ASD2 Index': np.random.uniform(100, 120, len(sample_dates)),
        'DED2 Index': np.random.uniform(150, 180, len(sample_dates)),
        'MND2 Index': np.random.uniform(80, 100, len(sample_dates))
    }, index=sample_dates)
    futures_data.index.name = 'Date'
    
    # Create sample index data
    index_data = pd.DataFrame({
        'SPX Index': np.random.uniform(3000, 3300, len(sample_dates)),
        'SX5E Index': np.random.uniform(3500, 3800, len(sample_dates)),
        'NKY Index': np.random.uniform(21000, 23000, len(sample_dates)),
        'USGG30YR Index': np.random.uniform(1.5, 2.5, len(sample_dates)),
        'GDBR30 Index': np.random.uniform(0.0, 0.5, len(sample_dates)),
        'GJGB30 Index': np.random.uniform(0.0, 0.3, len(sample_dates))
    }, index=sample_dates)
    index_data.index.name = 'Date'
    
    # Save to parquet files
    div_data.to_parquet(test_dir / "dividend_data.parquet")
    futures_data.to_parquet(test_dir / "dividend_futures_data.parquet")
    index_data.to_parquet(test_dir / "index_data.parquet")
    
    # Return the test directory and sample data
    yield {
        'dir': test_dir,
        'div_data': div_data,
        'futures_data': futures_data,
        'index_data': index_data
    }
    
    # Cleanup after tests
    if test_dir.exists():
        for file in test_dir.glob("*.parquet"):
            file.unlink()
        test_dir.rmdir()

def test_clean_dividend_data_returns_dataframe(setup_test_data_dir, monkeypatch):
    """Test if clean_dividend_data returns a pandas DataFrame."""
    # Redirect DATA_DIR to our test directory
    monkeypatch.setattr('clean_data.DATA_DIR', str(setup_test_data_dir['dir']))
    
    # Test the function
    df = clean_dividend_data()
    assert isinstance(df, pd.DataFrame), "clean_dividend_data should return a pandas DataFrame"

def test_clean_dividend_data_columns(setup_test_data_dir, monkeypatch):
    """Test if the DataFrame from clean_dividend_data has the expected columns."""
    # Redirect DATA_DIR to our test directory
    monkeypatch.setattr('clean_data.DATA_DIR', str(setup_test_data_dir['dir']))
    
    # Test the function
    df = clean_dividend_data()
    expected_columns = ['US_Dividend', 'EU_Dividend', 'JP_Dividend']
    assert sorted(df.columns.tolist()) == sorted(expected_columns), "clean_dividend_data returned DataFrame has incorrect column names"

def test_clean_dividend_data_index_type(setup_test_data_dir, monkeypatch):
    """Test if the DataFrame from clean_dividend_data has a datetime index."""
    # Redirect DATA_DIR to our test directory
    monkeypatch.setattr('clean_data.DATA_DIR', str(setup_test_data_dir['dir']))
    
    # Test the function
    df = clean_dividend_data()
    assert pd.api.types.is_datetime64_dtype(df.index), "clean_dividend_data index should be datetime64"

def test_clean_dividend_futures_data_returns_dataframe(setup_test_data_dir, monkeypatch):
    """Test if clean_dividend_futures_data returns a pandas DataFrame."""
    # Redirect DATA_DIR to our test directory
    monkeypatch.setattr('clean_data.DATA_DIR', str(setup_test_data_dir['dir']))
    
    # Test the function
    df = clean_dividend_futures_data()
    assert isinstance(df, pd.DataFrame), "clean_dividend_futures_data should return a pandas DataFrame"

def test_clean_dividend_futures_data_columns(setup_test_data_dir, monkeypatch):
    """Test if the DataFrame from clean_dividend_futures_data has the expected columns."""
    # Redirect DATA_DIR to our test directory
    monkeypatch.setattr('clean_data.DATA_DIR', str(setup_test_data_dir['dir']))
    
    # Test the function
    df = clean_dividend_futures_data()
    expected_columns = ['US_Div_Future', 'EU_Div_Future', 'JP_Div_Future']
    assert sorted(df.columns.tolist()) == sorted(expected_columns), "clean_dividend_futures_data returned DataFrame has incorrect column names"

def test_merge_dividend_data(setup_test_data_dir, monkeypatch):
    """Test if merge_dividend_data correctly merges dividend and futures data."""
    # Redirect DATA_DIR to our test directory
    monkeypatch.setattr('clean_data.DATA_DIR', str(setup_test_data_dir['dir']))
    
    # Test the function
    merged_df = merge_dividend_data()
    
    # Check if merge worked
    assert isinstance(merged_df, pd.DataFrame), "merge_dividend_data should return a pandas DataFrame"
    
    # Should have columns from both dataframes
    expected_columns = ['US_Dividend', 'EU_Dividend', 'JP_Dividend', 
                        'US_Div_Future', 'EU_Div_Future', 'JP_Div_Future']
    assert sorted(merged_df.columns.tolist()) == sorted(expected_columns), "Merged DataFrame has incorrect columns"
    
    # Should have at least as many rows as either input dataframe
    div_df = clean_dividend_data()
    fut_df = clean_dividend_futures_data()
    assert len(merged_df) >= max(len(div_df), len(fut_df)), "Merged DataFrame should contain at least as many rows as input DataFrames"

def test_clean_index_data(setup_test_data_dir, monkeypatch):
    """Test if clean_index_data correctly cleans index data."""
    # Redirect DATA_DIR to our test directory
    monkeypatch.setattr('clean_data.DATA_DIR', str(setup_test_data_dir['dir']))
    
    # Test the function with a specific end date
    end_date = '2020-03-31'
    df = clean_index_data(end_date, str(setup_test_data_dir['dir']))
    
    # Check basic properties
    assert isinstance(df, pd.DataFrame), "clean_index_data should return a pandas DataFrame"
    assert pd.api.types.is_datetime64_dtype(df.index), "clean_index_data index should be datetime64"
    
    # Check date range
    start_date = pd.to_datetime(START_DATE)
    end_date = pd.to_datetime(end_date)
    assert df.index.min() >= start_date, f"Index should start at or after {start_date}"
    assert df.index.max() <= end_date, f"Index should end at or before {end_date}"

def test_file_not_found_handling(monkeypatch, tmp_path):
    """Test that appropriate errors are raised when files are not found."""
    # Set DATA_DIR to an empty directory
    monkeypatch.setattr('clean_data.DATA_DIR', str(tmp_path))
    
    # Check that FileNotFoundError is raised
    with pytest.raises(FileNotFoundError):
        clean_dividend_data()
    
    with pytest.raises(FileNotFoundError):
        clean_dividend_futures_data()

if __name__ == "__main__":
    pytest.main(["-xvs", "test_clean_data.py"])