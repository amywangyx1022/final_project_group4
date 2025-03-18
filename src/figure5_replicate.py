"""
This module replicates Figure 5 from the paper 'Coronavirus: Impact on Stock Prices and Growth Expectations'.
It visualizes the expected dividend and GDP growth based on forecasting models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from settings import config

# Configuration variables
DATA_DIR = config("DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")
START_DT = config("START_DATE")
PAPER_END_DT = config("PAPER_END_DATE")
CURR_END_DT = config("CURR_END_DATE")
FORECAST_START = "2020-01-01"  # Start date for the forecast

def load_forecast_data(use_paper_period=True):
    """
    Load the forecast data generated from the Table 1 replication.
    
    Args:
        use_paper_period (bool): If True, use the original paper's date range.
                                 If False, use the updated date range.
    
    Returns:
        pd.DataFrame: Data with forecasted dividend growth
    """
    # Determine file prefix based on whether we're using the paper's period or updated period
    prefix = "forecast_paper_" if use_paper_period else "forecast_updated_"
    
    # Load the forecast data
    forecast_path = Path(OUTPUT_DIR) / "figures" / f"{prefix}dividend_growth.parquet"
    
    if not forecast_path.exists():
        raise FileNotFoundError(f"Required forecast file not found at {forecast_path}")
    
    forecast_data = pd.read_parquet(forecast_path)
    
    return forecast_data

def convert_dividend_to_gdp_growth(dividend_growth, region):
    """
    Convert dividend growth expectations to GDP growth expectations
    using the multipliers from the paper.
    
    Args:
        dividend_growth (pd.Series): Dividend growth expectations
        region (str): Region code ('US', 'EU', or 'JP')
        
    Returns:
        pd.Series: GDP growth expectations
    """
    # Coefficients to convert from dividend growth to GDP growth
    # These are from the paper's estimates
    conversion_factors = {
        'US': 0.67,
        'EU': 0.33,
        'JP': 0.46
    }
    
    return dividend_growth * conversion_factors[region]

def prepare_figure5_data(use_paper_period=True):
    """
    Prepare the data needed for Figure 5.
    
    Args:
        use_paper_period (bool): If True, use the original paper's date range.
                                 If False, use the updated date range.
    
    Returns:
        pd.DataFrame: Data prepared for Figure 5
    """
    # Load the forecast data
    forecast_data = load_forecast_data(use_paper_period)
    
    # Create a DataFrame to store the processed data
    fig5_data = pd.DataFrame(index=forecast_data.index)
    
    # Process each region
    for region in ['US', 'EU', 'JP']:
        div_growth_col = f"{region}_Expected_Div_Growth"
        
        if div_growth_col in forecast_data.columns:
            # Store dividend growth
            fig5_data[f"{region}_Dividend_Growth"] = forecast_data[div_growth_col]
            
            # Convert to GDP growth
            fig5_data[f"{region}_GDP_Growth"] = convert_dividend_to_gdp_growth(
                forecast_data[div_growth_col], region)
    
    # Extract key dates mentioned in the paper
    key_dates = [
        '2020-01-01',  # January 1
        '2020-01-23',  # Lockdown of Wuhan
        '2020-02-22',  # Italy quarantine
        '2020-03-11',  # US travel ban from EU
        '2020-03-13',  # US national emergency
        '2020-03-24',  # US fiscal stimulus news
    ]
    
    # Add any other significant dates that should be marked
    end_date = PAPER_END_DT if use_paper_period else CURR_END_DT
    if pd.to_datetime(end_date) not in pd.to_datetime(key_dates):
        key_dates.append(end_date)
    
    # Create a version with only key dates for specific points in the figure
    key_dates_df = fig5_data.reindex(pd.to_datetime(key_dates), method='nearest')
    
    return fig5_data, key_dates_df, key_dates

def create_figure5(fig5_data, key_dates_df, key_dates, use_paper_period=True):
    """
    Create Figure 5 showing expected dividend and GDP growth.
    
    Args:
        fig5_data (pd.DataFrame): Data for the full time series
        key_dates_df (pd.DataFrame): Data for key dates
        key_dates (list): List of key date strings
        use_paper_period (bool): If True, use the original paper's date range.
                                 If False, use the updated date range.
    
    Returns:
        tuple: Tuple of (fig1, fig2) matplotlib figure objects
    """
    # Create figure for panel a (dividend growth)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot dividend growth for each region
    ax1.plot(fig5_data.index, fig5_data['US_Dividend_Growth'], 
             label='US', color='blue', linewidth=2)
    ax1.plot(fig5_data.index, fig5_data['EU_Dividend_Growth'], 
             label='EU', color='red', linewidth=2)
    ax1.plot(fig5_data.index, fig5_data['JP_Dividend_Growth'], 
             label='Japan', color='gold', linewidth=2)
    
    # Add markers for key dates
    for region, color in [('US_Dividend_Growth', 'blue'), 
                          ('EU_Dividend_Growth', 'red'), 
                          ('JP_Dividend_Growth', 'gold')]:
        ax1.scatter(key_dates_df.index, key_dates_df[region], 
                   color=color, s=50, zorder=5)
    
    # Configure x-axis for dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    
    # Add vertical lines for key events
    event_labels = [
        'Jan 1',
        'Wuhan Lockdown',
        'Italy Quarantine',
        'US Travel Ban',
        'US Emergency',
        'Fiscal Stimulus'
    ]
    
    for date, label in zip(key_dates, event_labels):
        if date != key_dates[-1]:  # Skip the last date if it's just the end date
            ax1.axvline(x=pd.to_datetime(date), color='gray', linestyle='--', alpha=0.7)
            ax1.text(pd.to_datetime(date), ax1.get_ylim()[0], label, 
                    rotation=90, verticalalignment='bottom', fontsize=8)
    
    # Add title and labels
    ax1.set_title('Panel A: Expected Dividend Growth', fontsize=14)
    ax1.set_ylabel('Change in Expected Growth (%)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Set y-axis limits similar to the paper
    y_min = min(fig5_data[['US_Dividend_Growth', 'EU_Dividend_Growth', 'JP_Dividend_Growth']].min().min() * 1.1, -15)
    y_max = max(2, fig5_data[['US_Dividend_Growth', 'EU_Dividend_Growth', 'JP_Dividend_Growth']].max().max() * 1.1)
    ax1.set_ylim([y_min, y_max])
    
    # Create figure for panel b (GDP growth)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Plot GDP growth for each region
    ax2.plot(fig5_data.index, fig5_data['US_GDP_Growth'], 
             label='US', color='blue', linewidth=2)
    ax2.plot(fig5_data.index, fig5_data['EU_GDP_Growth'], 
             label='EU', color='red', linewidth=2)
    ax2.plot(fig5_data.index, fig5_data['JP_GDP_Growth'], 
             label='Japan', color='gold', linewidth=2)
    
    # Add markers for key dates
    for region, color in [('US_GDP_Growth', 'blue'), 
                          ('EU_GDP_Growth', 'red'), 
                          ('JP_GDP_Growth', 'gold')]:
        ax2.scatter(key_dates_df.index, key_dates_df[region], 
                   color=color, s=50, zorder=5)
    
    # Configure x-axis for dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    
    # Add vertical lines for key events
    for date, label in zip(key_dates, event_labels):
        if date != key_dates[-1]:  # Skip the last date if it's just the end date
            ax2.axvline(x=pd.to_datetime(date), color='gray', linestyle='--', alpha=0.7)
            ax2.text(pd.to_datetime(date), ax2.get_ylim()[0], label, 
                    rotation=90, verticalalignment='bottom', fontsize=8)
    
    # Add title and labels
    ax2.set_title('Panel B: Expected GDP Growth', fontsize=14)
    ax2.set_ylabel('Change in Expected Growth (%)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Set y-axis limits similar to the paper
    y_min = min(fig5_data[['US_GDP_Growth', 'EU_GDP_Growth', 'JP_GDP_Growth']].min().min() * 1.1, -10)
    y_max = max(2, fig5_data[['US_GDP_Growth', 'EU_GDP_Growth', 'JP_GDP_Growth']].max().max() * 1.1)
    ax2.set_ylim([y_min, y_max])
    
    # Adjust layout
    fig1.tight_layout()
    fig2.tight_layout()
    
    return fig1, fig2

def save_figure5(fig1, fig2, use_paper_period=True):
    """
    Save Figure 5 to the output directory.
    
    Args:
        fig1 (matplotlib.figure.Figure): Panel A figure
        fig2 (matplotlib.figure.Figure): Panel B figure
        use_paper_period (bool): If True, use the original paper's date range.
                                 If False, use the updated date range.
    """
    # Create figures directory if it doesn't exist
    figures_path = Path(OUTPUT_DIR) / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)
    
    # Determine file prefix
    prefix = "paper_" if use_paper_period else "updated_"
    
    # Save Panel A (dividend growth)
    fig1.savefig(figures_path / f"{prefix}figure5_panel_a.png", dpi=300, bbox_inches='tight')
   
    # Save Panel B (GDP growth)
    fig2.savefig(figures_path / f"{prefix}figure5_panel_b.png", dpi=300, bbox_inches='tight')
  
    # Create a combined figure
    fig_combined, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Copy content from individual figures to combined figure
    for ax_src, ax_dest in zip([fig1.axes[0], fig2.axes[0]], [ax1, ax2]):
        for line in ax_src.lines:
            ax_dest.plot(line.get_xdata(), line.get_ydata(), 
                        color=line.get_color(), 
                        linestyle=line.get_linestyle(),
                        linewidth=line.get_linewidth(),
                        label=line.get_label())
        
        for collection in ax_src.collections:
            ax_dest.scatter(collection.get_offsets()[:, 0], collection.get_offsets()[:, 1],
                           color=collection.get_facecolor()[0],
                           s=collection.get_sizes()[0])
        
        # Copy labels and title
        ax_dest.set_title(ax_src.get_title())
        ax_dest.set_xlabel(ax_src.get_xlabel())
        ax_dest.set_ylabel(ax_src.get_ylabel())
        ax_dest.set_ylim(ax_src.get_ylim())
        ax_dest.legend()
        ax_dest.grid(True, linestyle='--', alpha=0.7)
        
        # Configure date formatting
        ax_dest.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_dest.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
     
    
    # Adjust layout
    fig_combined.tight_layout()

    # Save combined figure
    fig_combined.savefig(figures_path / f"{prefix}figure5_combined.png", dpi=300, bbox_inches='tight')
   
    print(f"Figure 5 saved successfully for {'paper period' if use_paper_period else 'updated period'}.")

def main():
    """
    Main function to execute the Figure 5 replication.
    """
    print("Starting Figure 5 replication...")
    
    # Create figure for paper period
    print("Creating figure for paper period...")
    fig5_data, key_dates_df, key_dates = prepare_figure5_data(use_paper_period=True)
    fig1_paper, fig2_paper = create_figure5(fig5_data, key_dates_df, key_dates, use_paper_period=True)
    save_figure5(fig1_paper, fig2_paper, use_paper_period=True)
    
    # Create figure for updated period
    print("Creating figure for updated period...")
    fig5_data_updated, key_dates_df_updated, key_dates_updated = prepare_figure5_data(use_paper_period=False)
    fig1_updated, fig2_updated = create_figure5(fig5_data_updated, key_dates_df_updated, key_dates_updated, use_paper_period=False)
    save_figure5(fig1_updated, fig2_updated, use_paper_period=False)
    
    print("Figure 5 replication completed.")
    
    return {
        'fig5_data': fig5_data,
        'fig5_data_updated': fig5_data_updated,
        'panel_a_paper': fig1_paper,
        'panel_b_paper': fig2_paper,
        'panel_a_updated': fig1_updated,
        'panel_b_updated': fig2_updated
    }

if __name__ == "__main__":
    main()