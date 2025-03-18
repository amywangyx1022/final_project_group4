import pytest
import pandas as pd
from pathlib import Path
import numpy as np
from settings import config
from pull_bloomberg import pull_equity_and_bond_index_data
import datetime

DATA_DIR = config("DATA_DIR")

END_DATE = config("CURR_END_DATE")

DATA_PULL_START_DATE = config("DATA_PULL_START_DATE")
START_DATE = config("START_DATE")
BASE_DIR = config("BASE_DIR")
USE_BBG = config("USE_BBG")

def test_load_bbg_data_returns_dataframe():
    # Test if the function returns a pandas DataFrame
    df = pull_equity_and_bond_index_data(START_DATE,END_DATE)
    assert isinstance(df, pd.DataFrame), "load_data should return a pandas DataFrame"
    
def test_load_bbg_data_columns():
    # Test if the DataFrame has the expected columns
    df = pull_equity_and_bond_index_data(START_DATE,END_DATE)
    expected_columns = ['SPX Index','SX5E Index','NKY Index','USGG30YR Index','GDBR30 Index','GJGB30 Index']
    assert df.columns.tolist() == expected_columns, "load_bbg_data returned DataFrame has incorrect column names"


#TODO: CONFIRM DATA RANGE AND FIRST ROW VALUES
def test_load_bbg_data_date_range():
    # Test if the default date range has the expected start date and end date
    df = pull_equity_and_bond_index_data(START_DATE,END_DATE)
    df.index = pd.to_datetime(df.index)
    assert df.index.min() == pd.Timestamp('2020-01-01'), "load_bbg_data returned DataFrame has incorrect start date"
    assert df.index.max() ==  pd.Timestamp('2025-02-28'), "load_bbg_data returned DataFrame has incorrect latest end date"

def test_load_bbg_data_specific_values():
    # Test if specific values in the DataFrame are correct
    df =pull_equity_and_bond_index_data(START_DATE,END_DATE)
    df.index = pd.to_datetime(df.index)
    assert np.isclose(df.loc[df.index==datetime.datetime(2020,1,3), 'SPX Index'].values[0], 3234.85, atol=1e-2), "Incorrect value for spx index"


if __name__ == "__main__":
    test_load_bbg_data_specific_values()