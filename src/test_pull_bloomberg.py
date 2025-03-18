"""
This module tests the functionality of the pull_bloomberg.py module.
It focuses on testing the CSV fallback mechanism when Bloomberg is not available.
"""

import unittest
import pandas as pd
import os
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pull_bloomberg import (
    load_csv_dividend_data,
    load_csv_dividend_futures_data,
    load_csv_dividend_index_data
)

class TestPullBloombergFallback(unittest.TestCase):
    """Test cases for the pull_bloomberg.py module's fallback mechanisms."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_data_dir = Path("tests/test_data")
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample data for testing
        self.sample_index_data = pd.DataFrame({
            'SPX Index': [3230.78, 3257.85, 3276.24],
            'SX5E Index': [3748.47, 3768.96, 3791.69],
            'NKY Index': [23656.62, 23739.87, 23204.76],
            'USGG30YR Index': [2.39, 2.35, 2.30],
            'GDBR30 Index': [0.31, 0.28, 0.23],
            'GJGB30 Index': [0.43, 0.44, 0.42]
        }, index=pd.date_range(start='2020-01-01', periods=3, freq='D'))
        self.sample_index_data.index.name = 'Date'
        
        self.sample_dividend_data = pd.DataFrame({
            'SPX Index_DIV': [58.12, 58.65, 59.01],
            'SX5E Index_DIV': [123.45, 124.78, 125.36],
            'NKY Index_DIV': [456.78, 459.32, 462.15]
        }, index=pd.date_range(start='2020-01-01', periods=3, freq='D'))
        self.sample_dividend_data.index.name = 'Date'
        
        self.sample_futures_data = pd.DataFrame({
            'ASD2 Index': [150.23, 151.45, 149.87],
            'DED2 Index': [87.65, 86.43, 85.21],
            'MND2 Index': [345.67, 346.78, 344.56]
        }, index=pd.date_range(start='2020-01-01', periods=3, freq='D'))
        self.sample_futures_data.index.name = 'Date'
        
        # Save sample data to CSV files for testing the fallback mechanism
        os.makedirs(self.test_data_dir, exist_ok=True)
        self.sample_index_data.to_csv(self.test_data_dir / "index_data.csv")
        self.sample_dividend_data.to_csv(self.test_data_dir / "dividend_data.csv")
        self.sample_futures_data.to_csv(self.test_data_dir / "dividend_future_data.csv")
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove test files
        for file in self.test_data_dir.glob("*.csv"):
            try:
                file.unlink()
            except:
                pass
        try:
            self.test_data_dir.rmdir()
        except:
            pass
    
    @patch('src.pull_bloomberg.config')
    @patch('src.pull_bloomberg.os.path.exists')
    @patch('src.pull_bloomberg.pd.read_csv')
    def test_load_csv_dividend_data(self, mock_read_csv, mock_exists, mock_config):
        """Test loading dividend data from CSV."""
        mock_exists.return_value = True
        mock_read_csv.return_value = self.sample_dividend_data
        mock_config.return_value = str(self.test_data_dir)
        
        result = load_csv_dividend_data()
        
        mock_read_csv.assert_called_once()
        pd.testing.assert_frame_equal(result, self.sample_dividend_data)
    
    @patch('src.pull_bloomberg.config')
    @patch('src.pull_bloomberg.os.path.exists')
    @patch('src.pull_bloomberg.pd.read_csv')
    def test_load_csv_dividend_futures_data(self, mock_read_csv, mock_exists, mock_config):
        """Test loading dividend futures data from CSV."""
        mock_exists.return_value = True
        mock_read_csv.return_value = self.sample_futures_data
        mock_config.return_value = str(self.test_data_dir)
        
        result = load_csv_dividend_futures_data()
        
        mock_read_csv.assert_called_once()
        pd.testing.assert_frame_equal(result, self.sample_futures_data)
    
    @patch('src.pull_bloomberg.config')
    @patch('src.pull_bloomberg.os.path.exists')
    @patch('src.pull_bloomberg.pd.read_csv')
    def test_load_csv_dividend_index_data(self, mock_read_csv, mock_exists, mock_config):
        """Test loading dividend index data from CSV."""
        mock_exists.return_value = True
        mock_read_csv.return_value = self.sample_index_data
        mock_config.return_value = str(self.test_data_dir)
        
        result = load_csv_dividend_index_data()
        
        mock_read_csv.assert_called_once()
        pd.testing.assert_frame_equal(result, self.sample_index_data)
    
    @patch('src.pull_bloomberg.config')
    @patch('src.pull_bloomberg.os.path.exists')
    def test_load_csv_files_not_found(self, mock_exists, mock_config):
        """Test handling of missing CSV files."""
        mock_exists.return_value = False
        mock_config.return_value = str(self.test_data_dir)
        
        with self.assertRaises(FileNotFoundError):
            load_csv_dividend_data()
        
        with self.assertRaises(FileNotFoundError):
            load_csv_dividend_futures_data()
        
        with self.assertRaises(FileNotFoundError):
            load_csv_dividend_index_data()

    def test_integration_with_actual_files(self):
        """
        Integration test using actual CSV files.
        This test verifies that the functions can properly read CSV files
        without mocking.
        """
        # Patch the config function to point to our test data directory
        with patch('src.pull_bloomberg.config', return_value=str(self.test_data_dir)):
            # Test loading dividend data
            dividend_data = load_csv_dividend_data()
            self.assertIsInstance(dividend_data, pd.DataFrame)
            self.assertEqual(dividend_data.shape[0], 3)  # We have 3 rows
            
            # Test loading futures data
            futures_data = load_csv_dividend_futures_data()
            self.assertIsInstance(futures_data, pd.DataFrame)
            self.assertEqual(futures_data.shape[0], 3)  # We have 3 rows
            
            # Test loading index data
            index_data = load_csv_dividend_index_data()
            self.assertIsInstance(index_data, pd.DataFrame)
            self.assertEqual(index_data.shape[0], 3)  # We have 3 rows


if __name__ == '__main__':
    unittest.main()