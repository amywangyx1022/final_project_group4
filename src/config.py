"""
This configuration module sets up the directory structure and global settings for the project.
It defines the base, data, and output directories, as well as key date parameters for the analysis.
"""

from decouple import config
from pathlib import Path
 
BASE_DIR = Path(__file__).parent.parent

DATA_DIR = config('DATA_DIR', default=BASE_DIR / 'data/pulled/', cast=Path)
OUTPUT_DIR = config('OUTPUT_DIR', default=BASE_DIR / 'output/', cast=Path)

USE_BBG = False
START_DATE = "2020-01-01"

END_DATE = "2025-03-01"


if __name__ == "__main__":
    
    ## If they don't exist, create the data and output directories
    (DATA_DIR / 'pulled').mkdir(parents=True, exist_ok=True)

    # Sometimes, I'll create other folders to organize the data
    # (DATA_DIR / 'intermediate').mkdir(parents=True, exist_ok=True)
    # (DATA_DIR / 'derived').mkdir(parents=True, exist_ok=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    