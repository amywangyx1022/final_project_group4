"""
This module cleans the raw data from Bloomberg. we save a clean version of the data
in the _data folder for record keeping

"""

import pandas as pd
from settings import config
from pathlib import Path

DATA_DIR = config("DATA_DIR")
USE_BBG = config("USE_BBG")
START_DT = config("START_DATE")
PAPER_END_DT = config("PAPER_END_DATE")
CURR_END_DT = config("CURR_END_DATE")

def clean_bbg_data(end_date, data_dir=DATA_DIR):
        path = Path(data_dir) /"bloomberg_index_data.parquet"
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
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
    path = Path(DATA_DIR) / "cleaned_bbg_index_data.parquet"
    bbg_df.to_parquet(path)
