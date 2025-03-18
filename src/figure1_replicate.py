
from matplotlib import pyplot as plt
from settings import config
from pathlib import Path

import pandas as pd
DATA_DIR = config("DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")
PAPER_END_DATE = config("PAPER_END_DATE")

from calc_functions import calc_pct_returns
returns = calc_pct_returns(DATA_DIR)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))  # 2 rows, 3 columns

# Flatten axes for easy iteration
axes = axes.flatten()

for i, col in enumerate(returns.columns):
    # Plot the data
    axes[i].plot(returns.index, returns[col], label=col)
    axes[i].set_title(col)
    
    # 1) Set the x-axis to start at the first date and end at the last date in returns
    axes[i].set_xlim([returns.index[0], returns.index[-1]])
    
    # 2) Generate ticks at the start of each month (MS = Month Start)
    xticks = pd.date_range(start=returns.index[0], end=returns.index[-1], freq="MS")
    
    # 3) Apply those ticks and label them with three-letter month abbreviations
    axes[i].set_xticks(xticks)
    axes[i].set_xticklabels([x.strftime("%b") for x in xticks])
    
    # Add grid
    axes[i].grid()

filename = OUTPUT_DIR /"figures"/ 'figure1.png'
plt.savefig(filename)