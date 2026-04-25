import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, List, Dict, Optional
import logging
import joblib
from pathlib import Path

class DataPreprocessor:
    """Handle data preprocessing for demand forecasting."""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        # Create absolute path from current working directory
        if not self.output_dir.is_absolute():
            self.output_dir = Path.cwd() / self.output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.logger = self._setup_logger()
        
        # Initialize preprocessors
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.column_transformer = None
        self.feature_columns = []
        self.target_column = 'demand'
        
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
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess raw data."""
        df = df.copy()
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Handle missing values
        if df.isnull().any().any():
            self.logger.warning("Found missing values, filling with appropriate methods")
            
            # Fill numeric columns with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)
            
            # Fill categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Remove outliers in demand (beyond 3 standard deviations)
        demand_mean = df['demand'].mean()
        demand_std = df['demand'].std()
        lower_bound = demand_mean - 3 * demand_std
        upper_bound = demand_mean + 3 * demand_std
        
        outlier_count = ((df['demand'] < lower_bound) | (df['demand'] > upper_bound)).sum()
        if outlier_count > 0:
            self.logger.info(f"Removing {outlier_count} demand outliers")
            df = df[(df['demand'] >= lower_bound) & (df['demand'] <= upper_bound)]
        
        return df
    
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features."""
        df = df.copy()
        
        # Basic time features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        
        # Cyclical features for seasonality
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Season indicators
        df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        
        self.logger.info("Time features extracted")
        return df
    
    def create_lag_features(self, df: pd.DataFrame, 
                           lag_days: List[int] = [1, 7, 14, 30]) -> pd.DataFrame:
        """Create lag features for demand."""
        df = df.copy()
        df = df.sort_values(['product_id', 'region', 'date'])
        
        for lag in lag_days:
            df[f'demand_lag_{lag}'] = df.groupby(['product_id', 'region'])['demand'].transform(
                lambda x: x.shift(lag)
            )
        
        self.logger.info(f"Created lag features for days: {lag_days}")
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, 
                               windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """Create rolling window features."""
        df = df.copy()
        df = df.sort_values(['product_id', 'region', 'date'])
        
        for window in windows:
            # Rolling mean
            df[f'demand_roll_mean_{window}'] = df.groupby(['product_id', 'region'])['demand'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # Rolling std
            df[f'demand_roll_std_{window}'] = df.groupby(['product_id', 'region'])['demand'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            
            # Rolling median
            df[f'demand_roll_median_{window}'] = df.groupby(['product_id', 'region'])['demand'].transform(
                lambda x: x.rolling(window=window, min_periods=1).median()
            )
        
        self.logger.info(f"Created rolling features for windows: {windows}")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                   categorical_cols: List[str]) -> pd.DataFrame:
        """Encode categorical features."""
        df = df.copy()
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
        
        self.logger.info(f"Encoded categorical features: {categorical_cols}")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare all features for modeling."""
        df = self.clean_data(df)
        df = self.extract_time_features(df)
        
        # Create lag and rolling features
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        
        # Encode categorical features
        categorical_cols = ['product_id', 'region', 'category']
        available_categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        if available_categorical_cols:
            df = self.encode_categorical_features(df, available_categorical_cols)
        
        # Remove rows with NaN values (created by lag features)
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        
        if initial_rows != final_rows:
            self.logger.info(f"Removed {initial_rows - final_rows} rows with NaN values")
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify feature columns for modeling."""
        # Exclude non-feature columns
        exclude_cols = ['date', 'demand', 'product_id', 'region', 'category']
        
        # Get all columns except excluded ones
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        self.feature_columns = feature_cols
        self.logger.info(f"Selected {len(feature_cols)} feature columns")
        
        return feature_cols
    
    def create_preprocessing_pipeline(self, categorical_cols: List[str], 
                                    numerical_cols: List[str]) -> ColumnTransformer:
        """Create preprocessing pipeline for sklearn."""
        # Create preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        
        self.column_transformer = preprocessor
        return preprocessor
    
    def split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Split data into features and target."""
        feature_cols = self.get_feature_columns(df)
        X = df[feature_cols]
        y = df[self.target_column]
        
        self.logger.info(f"Split data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
    
    def preprocess_training_data(self, train_df: pd.DataFrame, 
                               test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, 
                                                             np.ndarray, np.ndarray]:
        """Preprocess training and testing data."""
        self.logger.info("Starting data preprocessing...")
        
        # Prepare features
        train_processed = self.prepare_features(train_df)
        test_processed = self.prepare_features(test_df)
        
        # Split features and target
        X_train, y_train = self.split_features_target(train_processed)
        X_test, y_test = self.split_features_target(test_processed)
        
        # Identify categorical and numerical columns
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create and fit preprocessing pipeline
        preprocessor = self.create_preprocessing_pipeline(categorical_cols, numerical_cols)
        
        # Fit on training data and transform both sets
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Save preprocessor
        self.save_preprocessor()
        
        self.logger.info(f"Preprocessing completed. Train shape: {X_train_processed.shape}, "
                        f"Test shape: {X_test_processed.shape}")
        
        return X_train_processed, y_train.values, X_test_processed, y_test.values
    
    def preprocess_single_record(self, record: Dict) -> np.ndarray:
        """Preprocess a single record for prediction."""
        # Convert to DataFrame
        df = pd.DataFrame([record])
        
        # Apply same preprocessing
        df = self.clean_data(df)
        df = self.extract_time_features(df)
        
        # Encode categorical features
        categorical_cols = ['product_id', 'region', 'category']
        available_categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        if available_categorical_cols:
            df = self.encode_categorical_features(df, available_categorical_cols)
        
        # Get feature columns
        if not self.feature_columns:
            raise ValueError("Feature columns not set. Run preprocess_training_data first.")
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0  # Default value for missing features
        
        X = df[self.feature_columns]
        
        # Apply preprocessing pipeline
        if self.column_transformer is None:
            raise ValueError("Preprocessor not fitted. Run preprocess_training_data first.")
        
        X_processed = self.column_transformer.transform(X)
        
        return X_processed
    
    def save_preprocessor(self, filename: str = "preprocessor.pkl"):
        """Save the preprocessor object."""
        filepath = self.output_dir / filename
        
        preprocessor_data = {
            'column_transformer': self.column_transformer,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler
        }
        
        joblib.dump(preprocessor_data, filepath)
        self.logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filename: str = "preprocessor.pkl"):
        """Load the preprocessor object."""
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {filepath}")
        
        preprocessor_data = joblib.load(filepath)
        
        self.column_transformer = preprocessor_data['column_transformer']
        self.label_encoders = preprocessor_data['label_encoders']
        self.feature_columns = preprocessor_data['feature_columns']
        self.scaler = preprocessor_data['scaler']
        
        self.logger.info(f"Preprocessor loaded from {filepath}")

if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append('..')
    sys.path.append('../..')
    from data.data_ingestion import DataIngestion
    
    # Load data - adjust path
    ingestion = DataIngestion(data_dir="../../data/raw")
    train_df, test_df = ingestion.load_training_data()
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    X_train, y_train, X_test, y_test = preprocessor.preprocess_training_data(train_df, test_df)
    
    print(f"Preprocessing completed successfully!")
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
