"""
This module loads the S&P 500 index, Nikkei, Eurostoxx index as well as 30 year yields from each from bloomberg

You must have a Bloomberg terminal open on this computer to run. You must
first install xbbg
"""

import pandas as pd
from settings import config
from pathlib import Path
from xbbg import blp

END_DATE = config("END_DATE")
DATA_DIR = config("DATA_DIR")
START_DATE = config("START_DATE")
BASE_DIR = config("BASE_DIR")
USE_BBG = config("USE_BBG")

def pull_bbg_data(start_date,end_date):
    
    df = blp.bdh(['SPX Index','SX5E Index','NKY Index','USGG30YR Index','GDBR30 Index','GJGB30 Index'],"PX_LAST", start_date, end_date)
    df.columns = df.columns.droplevel(1)
    df.index.name = 'Date'

    return df

def load_bbg_excel(data_dir=BASE_DIR):
    # load index data from excel
    path = data_dir / 'data_manual' / 'index_prices.xlsx'
    df = pd.read_excel(path, sheet_name='index')
    df = df.iloc[1:]
    df = df.rename(columns={"Unnamed: 0":'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Set the datetime column as index
    df.set_index('Date', inplace=True)
    
   
    return df


if __name__ == "__main__":
    if USE_BBG==True:
        df = pull_bbg_data(START_DATE,END_DATE)
       
    else:
        df = load_bbg_excel(BASE_DIR)
    path = Path(DATA_DIR) / "bloomberg_index_data.parquet"
    df.to_parquet(path)
        