"""
Unit test for the figure1_replicate.py module using pytest.

This test focuses on ensuring the calculated values match the paper's
figure values within a reasonable tolerance.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from settings import config
from calc_functions import calc_pct_returns

# Get configuration values
DATA_DIR = config("DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")

def test_figure1_values_match_paper():
    """
    Test that the calculated values for Figure 1 match the paper's values
    within a reasonable tolerance.
    """
    # Get the calculated returns
    returns = calc_pct_returns(DATA_DIR)
    
    # Define tolerance
    tolerance = 0.5  
    
    # Expected ranges from Figure 1 in the paper
    expected_ranges = {
        'SPX Index': {
            'min': 0.60,  # Minimum value (around March 2020)
            'max': 1.05,  # Maximum value (around Jan/Feb 2020)
            'mar_23': 0.70,  # Approximate value on March 23 (market bottom)
            'jul_20': 1.03   # Approximate value by July 2020
        },
        'SX5E Index': {
            'min': 0.65,  # Minimum value (around March 2020)
            'max': 1.05,  # Maximum value (around Jan/Feb 2020)
            'mar_23': 0.70,  # Approximate value on March 23
            'jul_20': 0.95   # Approximate value by July 2020
        },
        'NKY Index': {
            'min': 0.70,  # Minimum value (around March 2020)
            'max': 1.05,  # Maximum value (around Jan/Feb 2020)
            'mar_23': 0.75,  # Approximate value on March 23
            'jul_20': 0.97   # Approximate value by July 2020
        },
        'USGG30YR Index': {
            'min': 0.90,  # Minimum value (start of period)
            'max': 1.50,  # Maximum value (around March 2020)
            'mar_23': 1.40,  # Approximate value on March 23
            'jul_20': 1.35   # Approximate value by July 2020
        },
        'GDBR30 Index': {
            'min': 1.00,  # Minimum value (start of period)
            'max': 1.30,  # Maximum value (around March 2020)
            'mar_23': 1.25,  # Approximate value on March 23
            'jul_20': 1.12   # Approximate value by July 2020
        },
        'GJGB30 Index': {
            'min': 0.94,  # Minimum value (July 2020)
            'max': 1.04,  # Maximum value (around March 2020)
            'mar_23': 1.02,  # Approximate value on March 23
            'jul_20': 0.95   # Approximate value by July 2020
        }
    }
    
    # Check min/max values for each index
    for column, ranges in expected_ranges.items():
        if column in returns.columns:
            # Check if min value is within tolerance
            actual_min = returns[column].min()
            expected_min = ranges['min']
            min_error = abs(actual_min - expected_min) / expected_min
            assert min_error <= tolerance, \
                f"Minimum value for {column}: calculated={actual_min:.4f}, expected={expected_min:.4f}, error={min_error:.4f}"
            
            # Check if max value is within tolerance
            actual_max = returns[column].max()
            expected_max = ranges['max']
            max_error = abs(actual_max - expected_max) / expected_max
            assert max_error <= tolerance, \
                f"Maximum value for {column}: calculated={actual_max:.4f}, expected={expected_max:.4f}, error={max_error:.4f}"
            
            # Check specific dates if they exist in the data
            mar_23 = pd.Timestamp('2020-03-23')
            jul_20 = pd.Timestamp('2020-07-20')
            
            if mar_23 in returns.index:
                actual_mar23 = returns.loc[mar_23, column]
                expected_mar23 = ranges['mar_23']
                mar23_error = abs(actual_mar23 - expected_mar23) / expected_mar23
                assert mar23_error <= tolerance, \
                    f"March 23 value for {column}: calculated={actual_mar23:.4f}, expected={expected_mar23:.4f}, error={mar23_error:.4f}"
            
            if jul_20 in returns.index:
                actual_jul20 = returns.loc[jul_20, column]
                expected_jul20 = ranges['jul_20']
                jul20_error = abs(actual_jul20 - expected_jul20) / expected_jul20
                assert jul20_error <= tolerance, \
                    f"July 20 value for {column}: calculated={actual_jul20:.4f}, expected={expected_jul20:.4f}, error={jul20_error:.4f}"

if __name__ == "__main__":
    pytest.main(["-vxs", "src/test_figure1.py"])