
import pandas as pd 
from settings import config
from pathlib import Path
DATA_DIR = config("DATA_DIR")
import pandas as pd
from pathlib import Path

def calc_pct_returns(data_dir=DATA_DIR):
    path = Path(data_dir) / "clean"/"index_data_clean.parquet"
    df = pd.read_parquet(path)
    
    # Define the indices that need yield transformation

    yield_indices = ["USGG30YR Index", "GDBR30 Index", "GJGB30 Index"]
    
    # Apply 100 - yield transformation for these indices
    for idx in yield_indices:
        if idx in df.columns:
            df[idx] = 100 / ((1 + df[idx]*0.01) ** 30)
    pct_change = df.dropna().pct_change()
    cum_returns = (1 + pct_change).cumprod()  # Ensures it starts at 1

    return cum_returns



if __name__ == "__main__":
    
    df = calc_pct_returns(DATA_DIR)

        