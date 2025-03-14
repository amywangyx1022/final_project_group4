"""
This module cleans the raw data from Bloomberg and the Federal Reserve.
The 1-year expiry future prices throughout the period are interpolated from the active futures data.
It saves the both the paper and current end date cleaned data to separate parquet files for future use.
The module also selects the 1-year zero-coupon yield corresponding to the Bloomberg dates, and saves the 
discount factors to a parquet file for future use.

"""

import pandas as pd
from settings import config
from pathlib import Path

DATA_DIR = config.DATA_DIR
USE_BBG = config.USE_BBG
START_DT = config.START_DT
PAPER_END_DT = config.PAPER_END_DT
CURR_END_DT = config.CURR_END_DT


def clean_bbg_data(end_date, data_dir=DATA_DIR):
        path = Path(data_dir) /"bloomberg_index_data.parquet"
        df = pd.read_parquet(path)

        df = df.loc[START_DT : end_date]

        df.index = pd.to_datetime(df.index)

        return df


def format_df(df, all_col):
    if all_col:
        df = df.applymap(lambda x: '{:.3f}'.format(x))
    else:
        df.iloc[:, 1:] = df.iloc[:, 1:].applymap(lambda x: '{:.3f}'.format(x))
    return df


if __name__ == "__main__":
    bbg_df = clean_bbg_data(PAPER_END_DT, data_dir=DATA_DIR)
  
    path = Path(DATA_DIR) / "pulled" / "clean_bbg_paper_data.parquet"
    bbg_df.to_parquet(path)
