import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import time

class DemandForecastingModel:
    """Demand forecasting model with multiple algorithms."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.logger = self._setup_logger()
        
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0)
        }
        
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
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
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    feature_names: Optional[list] = None) -> Dict[str, Dict]:
        """Train multiple models and return their performance."""
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
        
        results = {}
        
        for name, model in self.models.items():
            self.logger.info(f"Training {name}...")
            start_time = time.time()
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                      scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores)
            
            # Training predictions
            train_pred = model.predict(X_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            train_mae = mean_absolute_error(y_train, train_pred)
            train_r2 = r2_score(y_train, train_pred)
            
            # Store trained model
            self.trained_models[name] = model
            
            training_time = time.time() - start_time
            
            results[name] = {
                'model': model,
                'cv_rmse_mean': cv_rmse.mean(),
                'cv_rmse_std': cv_rmse.std(),
                'train_rmse': train_rmse,
                'train_mae': train_mae,
                'train_r2': train_r2,
                'training_time': training_time
            }
            
            self.logger.info(f"{name} - CV RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f}), "
                           f"Train R²: {train_r2:.4f}")
        
        # Find best model based on CV RMSE
        best_model_name = min(results.keys(), key=lambda x: results[x]['cv_rmse_mean'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        self.logger.info(f"Best model: {best_model_name} with CV RMSE: {results[best_model_name]['cv_rmse_mean']:.4f}")
        
        return results
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate a specific model on test data."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error
        }
        
        self.logger.info(f"{model_name} Test Results - RMSE: {metrics['rmse']:.4f}, "
                        f"MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray, 
                            model_name: str = 'random_forest') -> Dict[str, Any]:
        """Perform hyperparameter tuning for the best model."""
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'lasso': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        }
        
        if model_name not in param_grids:
            self.logger.warning(f"No hyperparameter grid defined for {model_name}")
            return {}
        
        self.logger.info(f"Performing hyperparameter tuning for {model_name}...")
        
        model = self.models[model_name]
        param_grid = param_grids[model_name]
        
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update the model with best parameters
        best_model = grid_search.best_estimator_
        self.trained_models[model_name] = best_model
        
        if model_name == self.best_model_name:
            self.best_model = best_model
        
        self.logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        self.logger.info(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': best_model
        }
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """Get feature importance for tree-based models."""
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            self.logger.warning(f"Model {model_name} does not support feature importance")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def predict(self, X: np.ndarray, model_name: Optional[str] = None) -> np.ndarray:
        """Make predictions using the specified or best model."""
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        return model.predict(X)
    
    def save_model(self, model_name: Optional[str] = None, 
                   filename: Optional[str] = None) -> str:
        """Save a trained model."""
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        if filename is None:
            filename = f"{model_name}_model.pkl"
        
        filepath = self.model_dir / filename
        
        model_data = {
            'model': self.trained_models[model_name],
            'model_name': model_name,
            'feature_names': self.feature_names,
            'training_metadata': {
                'best_model': self.best_model_name,
                'available_models': list(self.trained_models.keys())
            }
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
        
        return str(filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.trained_models[model_data['model_name']] = model_data['model']
        self.feature_names = model_data['feature_names']
        self.best_model_name = model_data['training_metadata']['best_model']
        self.best_model = model_data['model']
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all trained models."""
        summary = {
            'best_model': self.best_model_name,
            'total_models_trained': len(self.trained_models),
            'available_models': list(self.trained_models.keys()),
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names
        }
        
        return summary

if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from data.data_ingestion import DataIngestion
    from data.preprocessing import DataPreprocessor
    
    # Load and preprocess data
    ingestion = DataIngestion(data_dir="../../data/raw")
    train_df, test_df = ingestion.load_training_data()
    
    preprocessor = DataPreprocessor()
    X_train, y_train, X_test, y_test = preprocessor.preprocess_training_data(train_df, test_df)
    
    # Train models
    model = DemandForecastingModel()
    training_results = model.train_models(X_train, y_train)
    
    # Evaluate best model on test set
    test_metrics = model.evaluate_model(model.best_model_name, X_test, y_test)
    
    # Get feature importance
    importance_df = model.get_feature_importance()
    
    print("\n=== Model Training Results ===")
    for name, results in training_results.items():
        print(f"{name}: CV RMSE = {results['cv_rmse_mean']:.4f}")
    
    print(f"\n=== Best Model: {model.best_model_name} ===")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test MAE: {test_metrics['mae']:.4f}")
    print(f"Test R²: {test_metrics['r2']:.4f}")
    
    # Save the best model
    model.save_model()
    print(f"\nBest model saved successfully!")
    
    # Display top features
    if not importance_df.empty:
        print("\n=== Top 10 Important Features ===")
        print(importance_df.head(10))
