"""
This module loads the S&P 500 index, Dividend yields, and all active futures during 
the given period from Bloomberg. 

You must have a Bloomberg terminal open on this computer to run. You must
first install xbbg
"""

import pandas as pd
import config

END_DATE = "2025-03-01"
from pathlib import Path
from xbbg import blp
DATA_DIR = config.DATA_DIR
START_DATE = config.START_DATE
END_DATE = config.END_DATE

def pull_bbg_data(start_date,end_date):
    
    df = blp.bdh(['SPX Index','SX5E Index','NKY Index','USGG30YR Index','GDBR30 Index','GJGB30 Index'],"PX_LAST", start_date, end_date)
    df.columns = df.columns.droplevel(1)
    df.index.name = 'Date'

    return df


if __name__ == "__main__":

    df = pull_bbg_data(START_DATE,END_DATE)
    path = Path(DATA_DIR) / "bloomberg_index_data.parquet"
    df.to_parquet(path)