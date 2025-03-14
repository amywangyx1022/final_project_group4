import pytest
import pandas as pd
from pathlib import Path
import numpy as np
from settings import config
from pull_bloomberg import pull_bbg_data

DATA_DIR = config("DATA_DIR")


def test_load_bbg_data_returns_dataframe():
    # Test if the function returns a pandas DataFrame
    df = pull_bbg_data(DATA_DIR)
    assert isinstance(df, pd.DataFrame), "load_data should return a pandas DataFrame"
    
def test_load_bbg_data_columns():
    # Test if the DataFrame has the expected columns
    df = pull_bbg_data(DATA_DIR)
    expected_columns = ['SPX Index','SX5E Index','NKY Index','USGG30YR Index','GDBR30 Index','GJGB30 Index']
    assert df.columns.tolist() == expected_columns, "load_bbg_data returned DataFrame has incorrect column names"

def test_load_bbg_data_invalid_directory():
    # Test if the function raises an error when given an invalid directory
    with pytest.raises(FileNotFoundError):
        pull_bbg_data(data_dir=Path("invalid_directory"))

#TODO: CONFIRM DATA RANGE AND FIRST ROW VALUES
def test_load_bbg_data_date_range():
    # Test if the default date range has the expected start date and end date
    df = pull_bbg_data(DATA_DIR)
    assert df.index.min() == pd.Timestamp('2020-01-29'), "load_bbg_data returned DataFrame has incorrect start date"
    assert df.index.max() == pd.Timestamp('2024-01-31'), "load_bbg_data returned DataFrame has incorrect latest end date"

def test_load_bbg_data_specific_values():
    # Test if specific values in the DataFrame are correct
    df = pull_bbg_data(DATA_DIR)
    assert np.isclose(df.loc['1988-01-29', 'dividend yield'], 3.4909, atol=1e-4), "Incorrect 'dividend yield' value for 1988-01-29"
    assert np.isclose(df.loc['1988-02-29', 'index'], 267.82, atol=1e-2), "Incorrect 'index' value for 1988-02-29"
    assert np.isclose(df.loc['2017-06-30', 'futures'], 2421.00, atol=1e-2), "Incorrect 'futures' value for 2017-06-30"
    assert np.isclose(df.loc['2024-01-31', 'futures'], 4870.50, atol=1e-2), "Incorrect 'futures' value for 2017-06-30"