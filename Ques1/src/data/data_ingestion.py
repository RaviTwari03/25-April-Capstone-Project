import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import logging
from pathlib import Path
import os

class DataIngestion:
    """Handle data ingestion for retail demand forecasting."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def load_data(self, file_name: str) -> pd.DataFrame:
        """Load data from CSV file."""
        file_path = self.data_dir / file_name
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Loaded {len(df)} records from {file_name}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data from {file_name}: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data quality and structure."""
        required_columns = ['date', 'product_id', 'region', 'demand']
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check for null values in critical columns
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            self.logger.warning(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
        
        # Check data types
        if not pd.api.types.is_numeric_dtype(df['demand']):
            self.logger.error("Demand column must be numeric")
            return False
        
        # Check for negative demand
        negative_demand = (df['demand'] < 0).sum()
        if negative_demand > 0:
            self.logger.warning(f"Found {negative_demand} negative demand values")
        
        self.logger.info("Data validation completed successfully")
        return True
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive data summary."""
        summary = {
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max(),
                'unique_days': df['date'].nunique()
            },
            'products': {
                'unique_count': df['product_id'].nunique(),
                'sample_ids': df['product_id'].unique()[:5].tolist()
            },
            'regions': {
                'unique_count': df['region'].nunique(),
                'region_names': df['region'].unique().tolist()
            },
            'demand_stats': {
                'mean': float(df['demand'].mean()),
                'median': float(df['demand'].median()),
                'std': float(df['demand'].std()),
                'min': float(df['demand'].min()),
                'max': float(df['demand'].max())
            },
            'missing_values': df.isnull().sum().to_dict()
        }
        
        if 'category' in df.columns:
            summary['categories'] = {
                'unique_count': df['category'].nunique(),
                'category_names': df['category'].unique().tolist()
            }
        
        return summary
    
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and testing datasets."""
        try:
            train_df = self.load_data('retail_demand_train.csv')
            test_df = self.load_data('retail_demand_test.csv')
            
            # Validate both datasets
            if not self.validate_data(train_df):
                raise ValueError("Training data validation failed")
            if not self.validate_data(test_df):
                raise ValueError("Testing data validation failed")
            
            self.logger.info(f"Training data: {len(train_df)} records")
            self.logger.info(f"Testing data: {len(test_df)} records")
            
            return train_df, test_df
            
        except Exception as e:
            self.logger.error(f"Error loading training data: {str(e)}")
            raise
    
    def load_products_info(self) -> pd.DataFrame:
        """Load product information."""
        try:
            products_df = self.load_data('products.csv')
            self.logger.info(f"Loaded {len(products_df)} products")
            return products_df
        except Exception as e:
            self.logger.error(f"Error loading products info: {str(e)}")
            raise
    
    def filter_data_by_date(self, df: pd.DataFrame, 
                           start_date: str, end_date: str) -> pd.DataFrame:
        """Filter data by date range."""
        df['date'] = pd.to_datetime(df['date'])
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        filtered_df = df[mask].copy()
        
        self.logger.info(f"Filtered data: {len(filtered_df)} records "
                        f"from {len(df)} total records")
        
        return filtered_df
    
    def filter_data_by_products(self, df: pd.DataFrame, 
                               product_ids: List[str]) -> pd.DataFrame:
        """Filter data by specific product IDs."""
        filtered_df = df[df['product_id'].isin(product_ids)].copy()
        
        self.logger.info(f"Filtered by {len(product_ids)} products: "
                        f"{len(filtered_df)} records from {len(df)} total")
        
        return filtered_df
    
    def filter_data_by_regions(self, df: pd.DataFrame, 
                               regions: List[str]) -> pd.DataFrame:
        """Filter data by specific regions."""
        filtered_df = df[df['region'].isin(regions)].copy()
        
        self.logger.info(f"Filtered by {len(regions)} regions: "
                        f"{len(filtered_df)} records from {len(df)} total")
        
        return filtered_df
    
    def get_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from date column."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Extract time features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        
        # Add cyclical features for better seasonality capture
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        self.logger.info("Time series features extracted")
        return df
    
    def create_lag_features(self, df: pd.DataFrame, 
                           lag_days: List[int] = [1, 7, 14, 30]) -> pd.DataFrame:
        """Create lag features for demand."""
        df = df.copy()
        df = df.sort_values(['product_id', 'region', 'date'])
        
        for lag in lag_days:
            df[f'demand_lag_{lag}'] = df.groupby(['product_id', 'region'])['demand'].shift(lag)
        
        self.logger.info(f"Created lag features for days: {lag_days}")
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, 
                               windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """Create rolling window features."""
        df = df.copy()
        df = df.sort_values(['product_id', 'region', 'date'])
        
        for window in windows:
            # Rolling mean
            df[f'demand_roll_mean_{window}'] = df.groupby(['product_id', 'region'])['demand'] \
                .rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
            
            # Rolling std
            df[f'demand_roll_std_{window}'] = df.groupby(['product_id', 'region'])['demand'] \
                .rolling(window=window, min_periods=1).std().reset_index(0, drop=True)
        
        self.logger.info(f"Created rolling features for windows: {windows}")
        return df

if __name__ == "__main__":
    # Example usage
    ingestion = DataIngestion()
    
    try:
        # Load data
        train_df, test_df = ingestion.load_training_data()
        products_df = ingestion.load_products_info()
        
        # Get summary
        summary = ingestion.get_data_summary(train_df)
        print("Training Data Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Extract features
        train_df_featured = ingestion.get_time_series_features(train_df)
        train_df_featured = ingestion.create_lag_features(train_df_featured)
        train_df_featured = ingestion.create_rolling_features(train_df_featured)
        
        print(f"\nFeature engineering completed. "
              f"New shape: {train_df_featured.shape}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
