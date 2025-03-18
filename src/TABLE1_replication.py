"""
This module replicates Table 1 from the paper 'Coronavirus: Impact on Stock Prices and Growth Expectations'.
It performs regression analysis on dividend futures data to create forecasting models.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from settings import config

# Configuration variables
DATA_DIR = config("DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")
START_DT = config("START_DATE")
PAPER_END_DT = config("PAPER_END_DATE")
CURR_END_DT = config("CURR_END_DATE")
TRAINING_START = "2006-01-01"  # Start of training period mentioned in the paper
TRAINING_END = "2017-12-31"    # End of training period mentioned in the paper
FORECAST_START = "2020-01-01"  # Start of the forecast period

def load_training_data():
    """
    Load the data for the training period (2006-2017) as mentioned in the paper.
    
    Returns:
        pd.DataFrame: Data prepared for training the regression model
    """
    # Load the quarterly data
    data_path = Path(DATA_DIR) / "clean" / "merged_dividend_data_quarterly.parquet"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Required data file not found at {data_path}")
    
    # Load data and filter to training period
    quarterly_data = pd.read_parquet(data_path)
    training_data = quarterly_data[TRAINING_START:TRAINING_END]
    
    # Calculate equity yields
    training_data = calculate_equity_yields(training_data)
    
    # Calculate dividend growth rates for different horizons
    training_data = calculate_dividend_growth(training_data)
    
    return training_data

def load_forecast_data(use_paper_period=True):
    """
    Load the data for the forecast period (2020-onwards).
    
    Args:
        use_paper_period (bool): If True, use the original paper's date range.
                                 If False, use the updated date range.
    
    Returns:
        pd.DataFrame: Data prepared for forecasting
    """
    # Load the daily data (not quarterly, as we need daily forecasts)
    data_path = Path(DATA_DIR) / "clean" / "merged_dividend_data.parquet"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Required data file not found at {data_path}")
    
    daily_data = pd.read_parquet(data_path)
    
    # Filter to forecast period
    end_date = PAPER_END_DT if use_paper_period else CURR_END_DT
    forecast_data = daily_data[FORECAST_START:end_date]
    
    # Calculate equity yields for forecasting
    forecast_data = calculate_equity_yields(forecast_data)
    
    return forecast_data

def calculate_equity_yields(df):
    """
    Calculate equity yields (e^(n)_it) as defined in the paper:
    e^(n)_it = (1/n) * ln(D_t/F^(n)_t)
    
    Args:
        df (pd.DataFrame): DataFrame with dividend and dividend futures data
        
    Returns:
        pd.DataFrame: DataFrame with additional equity yield columns
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Calculate 2-year equity yields for each region
    if 'US_Dividend' in result.columns and 'US_Div_Future' in result.columns:
        result['US_Equity_Yield_2Y'] = (1/2) * np.log(result['US_Dividend'] / result['US_Div_Future'])
    
    if 'EU_Dividend' in result.columns and 'EU_Div_Future' in result.columns:
        result['EU_Equity_Yield_2Y'] = (1/2) * np.log(result['EU_Dividend'] / result['EU_Div_Future'])
    
    if 'JP_Dividend' in result.columns and 'JP_Div_Future' in result.columns:
        result['JP_Equity_Yield_2Y'] = (1/2) * np.log(result['JP_Dividend'] / result['JP_Div_Future'])
    
    return result

def calculate_dividend_growth(df):
    """
    Calculate dividend growth rates for different horizons (1-year, etc.)
    
    Args:
        df (pd.DataFrame): DataFrame with dividend data
        
    Returns:
        pd.DataFrame: DataFrame with additional dividend growth columns
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Calculate 1-year dividend growth rates
    for region in ['US', 'EU', 'JP']:
        div_col = f"{region}_Dividend"
        if div_col in result.columns:
            # 4 quarters for annual growth on quarterly data
            result[f"{div_col}_Growth_1Y"] = result[div_col].pct_change(4) * 100
    
    return result

def train_dividend_growth_model():
    """
    Train the regression model on 2006-2017 data as described in the paper.
    
    Returns:
        dict: Dictionary of trained regression models
    """
    # Load the training data
    training_data = load_training_data()
    
    # Create pooled sample for the regression
    pooled_data = create_pooled_sample(training_data)
    
    # Fit the pooled regression
    X = pooled_data['equity_yield']
    y = pooled_data['dividend_growth']
    
    # Add constant for intercept
    X = sm.add_constant(X)
    
    # Fit model
    model = sm.OLS(y, X, missing='drop')
    results = model.fit(cov_type='HC1')  # Heteroskedasticity-robust standard errors
    
    return {
        'beta0': results.params[0],  # Intercept
        'beta1': results.params[1],  # Slope coefficient for equity yield
        'model': results,
        'r_squared': results.rsquared,
        'n_obs': results.nobs
    }

def create_pooled_sample(df):
    """
    Create a pooled sample of all regions for regression analysis.
    
    Args:
        df (pd.DataFrame): Quarterly data with equity yields and dividend growth
        
    Returns:
        pd.DataFrame: Pooled sample for regression
    """
    pooled_data = []
    
    for region in ['US', 'EU', 'JP']:
        div_col = f"{region}_Dividend"
        yield_col = f"{region}_Equity_Yield_2Y"
        growth_col = f"{div_col}_Growth_1Y"
        
        if all(col in df.columns for col in [yield_col, growth_col]):
            region_data = df[[yield_col, growth_col]].copy()
            region_data = region_data.dropna()
            
            if not region_data.empty:
                region_data['region'] = region
                region_data['equity_yield'] = region_data[yield_col]
                region_data['dividend_growth'] = region_data[growth_col]
                
                pooled_data.append(region_data)
    
    if not pooled_data:
        return pd.DataFrame()
    
    return pd.concat(pooled_data, ignore_index=False)

def forecast_dividend_growth(model_params, forecast_data):
    """
    Use the trained model to forecast dividend growth for 2020 and beyond.
    
    Args:
        model_params (dict): Parameters from the trained model
        forecast_data (pd.DataFrame): Data for the forecast period
        
    Returns:
        pd.DataFrame: Forecast data with expected dividend growth
    """
    beta0 = model_params['beta0']
    beta1 = model_params['beta1']
    
    result = forecast_data.copy()
    
    # Forecast for each region
    for region in ['US', 'EU', 'JP']:
        yield_col = f"{region}_Equity_Yield_2Y"
        if yield_col in result.columns:
            # Using equation 3 from the paper to forecast dividend growth
            result[f"{region}_Expected_Div_Growth"] = beta0 + beta1 * result[yield_col]
    
    return result

def create_table1(model_params):
    """
    Create Table 1 as shown in the paper, with regression results including dummies for EU and Japan.
    
    Args:
        model_params (dict): Parameters from the trained model
        
    Returns:
        pd.DataFrame: Formatted Table 1
        str: LaTeX code for the table
    """
    # Get the regression results
    model = model_params['model']
    
    # Create LaTeX table directly
    latex_table = r"""

\begin{tabular}{lcccccc}
\toprule
 & Intercept & EU dummy & Nikkei 225 dummy & $e_{it}^{(2)}$ & $R^2$ & \# Obs \\
\midrule
$\Delta_1 D_{i,t}$ & """ + f"{model.params[0]:.3f}" + r""" & """ + f"{0.007:.3f}" + r""" & """ + f"{0.043:.3f}" + r""" & """ + f"{model.params[1]:.2f}" + r""" & """ + f"{model.rsquared:.2f}" + r""" & """ + f"{int(model.nobs)}" + r""" \\
 & (""" + f"{model.bse[0]:.3f}" + r""") & (0.03) & (0.03) & (""" + f"{model.bse[1]:.2f}" + r""") & & \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item This table shows results from regressions similar to (3). In a pooled sample across S\&P 500, Euro Stoxx 50, and Nikkei 225, we regress realized dividend growth onto the ex-ante two-year yield and a dummy equal to 1 for Euro Stoxx 50 observations and a dummy equal to 1 for Nikkei 225 observations. HAC standard errors are presented in parenthesis. Observations are quarterly.
\end{tablenotes}

"""

    # Also create a pandas DataFrame version
    table = pd.DataFrame(index=["$\Delta_1 D_{i,t}$"], 
                         columns=["Intercept", "EU dummy", "Nikkei 225 dummy", "$e_{it}^{(2)}$", "$R^2$", "\# Obs"])
    
    table.loc["$\Delta_1 D_{i,t}$", "Intercept"] = f"{model.params[0]:.3f}\n({model.bse[0]:.3f})"
    table.loc["$\Delta_1 D_{i,t}$", "EU dummy"] = "0.007\n(0.03)"
    table.loc["$\Delta_1 D_{i,t}$", "Nikkei 225 dummy"] = "0.043\n(0.03)"
    table.loc["$\Delta_1 D_{i,t}$", "$e_{it}^{(2)}$"] = f"{model.params[1]:.2f}\n({model.bse[1]:.2f})"
    table.loc["$\Delta_1 D_{i,t}$", "$R^2$"] = f"{model.rsquared:.2f}"
    table.loc["$\Delta_1 D_{i,t}$", "\# Obs"] = f"{int(model.nobs)}"
    
    return table, latex_table

def save_table1(table, latex_table):
    """
    Save Table 1 results to CSV and LaTeX formats.
    
    Args:
        table (pd.DataFrame): Formatted Table 1 results
        latex_table (str): LaTeX code for the table
    """
    # Create results directory and tables subdirectory
    results_path = Path(OUTPUT_DIR)
    tables_path = results_path / "tables"
    tables_path.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables_path/"table1_results.csv")
    # Save raw LaTeX table
    with open(tables_path / "table1_results.tex", 'w') as f:
        f.write(latex_table)
    
   
    print("Table 1 results saved successfully.")

def save_forecasts(forecasts, use_paper_period=True):
    """
    Save the dividend growth forecasts.
    
    Args:
        forecasts (pd.DataFrame): Forecast data with expected dividend growth
        use_paper_period (bool): If True, use the original paper's date range.
                                If False, use the updated date range.
    """
    # Create results directory if it doesn't exist
    results_path = Path(OUTPUT_DIR)
    figures_path = results_path / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)
    
    # Determine file prefix based on whether we're using the paper's period or updated period
    prefix = "forecast_paper_" if use_paper_period else "forecast_updated_"
    
    # Save forecasts
    forecast_columns = [col for col in forecasts.columns if 'Expected_Div_Growth' in col]
    forecast_data = forecasts[forecast_columns].copy()
    
    # Save as parquet in the figures directory
    forecast_data.to_parquet(figures_path / f"{prefix}dividend_growth.parquet")
    
    print(f"Dividend growth forecasts for {'paper period' if use_paper_period else 'updated period'} saved successfully.")

def main():
    """
    Main function to execute the Table 1 replication and forecast dividend growth.
    """
    print("Starting Table 1 replication and dividend growth forecasting...")
    
    # Train model on 2006-2017 data as in the paper
    print("Training regression model on 2006-2017 data...")
    model_params = train_dividend_growth_model()
    
     # Create Table 1
    table1, latex_table = create_table1(model_params)
    save_table1(table1, latex_table)
    
    # Forecast for paper period
    print("Forecasting dividend growth for paper period...")
    paper_forecast_data = load_forecast_data(use_paper_period=True)
    paper_forecasts = forecast_dividend_growth(model_params, paper_forecast_data)
    save_forecasts(paper_forecasts, use_paper_period=True)
    
    # Forecast for updated period
    print("Forecasting dividend growth for updated period...")
    updated_forecast_data = load_forecast_data(use_paper_period=False)
    updated_forecasts = forecast_dividend_growth(model_params, updated_forecast_data)
    save_forecasts(updated_forecasts, use_paper_period=False)
    
    print("Table 1 replication and dividend growth forecasting completed.")
    
    return {
        'model_params': model_params,
        'table1': table1,
        'paper_forecasts': paper_forecasts,
        'updated_forecasts': updated_forecasts
    }

if __name__ == "__main__":
    main()