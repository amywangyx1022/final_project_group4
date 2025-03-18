
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

returns.describe()

column_names_map = {
    'SPX Index': 'US Stock Market Index',
    'SX5E Index': 'Euro Stoxx 50 Index',
    'NKY Index': 'Nikkei 225 Index',
    'USGG30YR Index': 'US 30-Year Gov Bond',
    'GDBR30 Index': 'German 30-Year Gov Bund',
    'GJGB30 Index': 'Japanese 30-Year Gov Bond'
}

escape_coverter = {
    '25%':'25\\%',
    '50%':'50\\%',
    '75%':'75\\%'
}


## Suppress scientific notation and limit to 3 decimal places
# Sets display, but doesn't affect formatting to LaTeX
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# Sets format for printing to LaTeX
float_format_func = lambda x: '{:.2f}'.format(x)

# Pooled summary stats
describe_all = (
    returns.
    describe().T.
    rename(index=column_names_map, columns=escape_coverter)
)
describe_all['count'] = describe_all['count'].astype(int)
describe_all.columns.name = 'Full Sample: 1947 - 2023'
latex_table_string_all = describe_all.to_latex(escape=False, float_format=float_format_func)

latex_table_string_split = [
    *latex_table_string_all.split('\n')[0:-3], # Skip the \end{tabular} and \bottomrule lines
]
latex_table_string = '\n'.join(latex_table_string_split)
# print(latex_table_string)
path = OUTPUT_DIR / "tables"/f'additional_stats.tex'
with open(path, "w") as text_file:
    text_file.write(latex_table_string)
