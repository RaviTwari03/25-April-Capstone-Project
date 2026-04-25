import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import json
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MLflowManager:
    """Manage MLflow experiments and model tracking."""
    
    def __init__(self, experiment_name: str = None, tracking_uri: str = None):
        self.experiment_name = experiment_name or os.getenv('MLFLOW_EXPERIMENT_NAME', 'retail_demand_forecasting')
        self.tracking_uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        
        self.logger = self._setup_logger()
        
        # Set up MLflow
        self._setup_mlflow()
        
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
    
    def _setup_mlflow(self):
        """Set up MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            self.logger.info(f"MLflow tracking URI set to: {self.tracking_uri}")
        except Exception as e:
            self.logger.warning(f"Failed to set MLflow tracking URI: {e}")
            self.logger.info("Using local MLflow tracking")
        
        # Set experiment
        try:
            mlflow.set_experiment(self.experiment_name)
            self.logger.info(f"MLflow experiment set to: {self.experiment_name}")
        except Exception as e:
            self.logger.error(f"Failed to set MLflow experiment: {e}")
    
    def log_model_training(self, model_name: str, model: Any, 
                          X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          training_metrics: Dict[str, float],
                          test_metrics: Dict[str, float],
                          hyperparameters: Dict[str, Any] = None,
                          feature_names: List[str] = None,
                          artifacts: Dict[str, str] = None) -> str:
        """Log model training to MLflow."""
        
        with mlflow.start_run(run_name=f"{model_name}_training"):
            # Log model name
            mlflow.set_tag("model_name", model_name)
            
            # Log hyperparameters
            if hyperparameters:
                for param, value in hyperparameters.items():
                    mlflow.log_param(param, value)
            
            # Log training metrics
            for metric, value in training_metrics.items():
                mlflow.log_metric(f"train_{metric}", value)
            
            # Log test metrics
            for metric, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric}", value)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log feature names
            if feature_names:
                mlflow.log_dict({"feature_names": feature_names}, "features.json")
            
            # Log artifacts
            if artifacts:
                for artifact_name, artifact_path in artifacts.items():
                    if Path(artifact_path).exists():
                        mlflow.log_artifact(artifact_path, artifact_name)
            
            # Log dataset info
            dataset_info = {
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "features": X_train.shape[1],
                "target_mean": float(np.mean(y_train)),
                "target_std": float(np.std(y_train))
            }
            mlflow.log_dict(dataset_info, "dataset_info.json")
            
            run_id = mlflow.active_run().info.run_id
            self.logger.info(f"Logged {model_name} training to MLflow with run_id: {run_id}")
            
            return run_id
    
    def log_model_evaluation(self, model_name: str, run_id: str,
                           evaluation_metrics: Dict[str, float],
                           predictions: np.ndarray = None,
                           actuals: np.ndarray = None) -> str:
        """Log model evaluation to MLflow."""
        
        with mlflow.start_run(run_id=run_id):
            # Log evaluation metrics
            for metric, value in evaluation_metrics.items():
                mlflow.log_metric(f"eval_{metric}", value)
            
            # Log predictions if provided
            if predictions is not None and actuals is not None:
                # Create predictions DataFrame
                pred_df = pd.DataFrame({
                    'actual': actuals,
                    'predicted': predictions,
                    'residual': actuals - predictions,
                    'abs_error': np.abs(actuals - predictions)
                })
                
                # Save and log predictions
                pred_path = f"predictions_{model_name}.csv"
                pred_df.to_csv(pred_path, index=False)
                mlflow.log_artifact(pred_path, "predictions")
                
                # Remove temporary file
                os.remove(pred_path)
            
            self.logger.info(f"Logged {model_name} evaluation to MLflow")
            
            return run_id
    
    def register_model(self, model_name: str, run_id: str, 
                      model_registry_name: str = None) -> str:
        """Register model in MLflow Model Registry."""
        
        if model_registry_name is None:
            model_registry_name = os.getenv('MODEL_NAME', 'demand_forecasting_model')
        
        try:
            # Register the model
            model_uri = f"runs:/{run_id}/model"
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=model_registry_name
            )
            
            self.logger.info(f"Registered model {model_name} as version {registered_model.version}")
            
            return registered_model.version
            
        except Exception as e:
            self.logger.error(f"Failed to register model: {e}")
            return None
    
    def get_model_versions(self, model_registry_name: str = None) -> List[Dict]:
        """Get all versions of a registered model."""
        
        if model_registry_name is None:
            model_registry_name = os.getenv('MODEL_NAME', 'demand_forecasting_model')
        
        try:
            client = mlflow.tracking.MlflowClient()
            model_versions = client.search_model_versions(f"name='{model_registry_name}'")
            
            versions_info = []
            for version in model_versions:
                versions_info.append({
                    'version': version.version,
                    'run_id': version.run_id,
                    'status': version.current_stage,
                    'creation_timestamp': version.creation_timestamp
                })
            
            return versions_info
            
        except Exception as e:
            self.logger.error(f"Failed to get model versions: {e}")
            return []
    
    def load_model_from_registry(self, model_registry_name: str = None, 
                                 version: str = None) -> Any:
        """Load model from MLflow Model Registry."""
        
        if model_registry_name is None:
            model_registry_name = os.getenv('MODEL_NAME', 'demand_forecasting_model')
        
        try:
            if version:
                model_uri = f"models:/{model_registry_name}/{version}"
            else:
                model_uri = f"models:/{model_registry_name}/latest"
            
            model = mlflow.sklearn.load_model(model_uri)
            
            self.logger.info(f"Loaded model from registry: {model_uri}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model from registry: {e}")
            return None
    
    def compare_models(self, model_registry_name: str = None) -> pd.DataFrame:
        """Compare all registered models."""
        
        if model_registry_name is None:
            model_registry_name = os.getenv('MODEL_NAME', 'demand_forecasting_model')
        
        try:
            client = mlflow.tracking.MlflowClient()
            model_versions = client.search_model_versions(f"name='{model_registry_name}'")
            
            comparison_data = []
            
            for version in model_versions:
                run = client.get_run(version.run_id)
                
                # Extract metrics
                metrics = {}
                for metric_name, metric_value in run.data.metrics.items():
                    metrics[metric_name] = metric_value
                
                # Extract parameters
                params = {}
                for param_name, param_value in run.data.params.items():
                    params[param_name] = param_value
                
                comparison_data.append({
                    'version': version.version,
                    'run_id': version.run_id,
                    'status': version.current_stage,
                    'test_rmse': metrics.get('test_rmse', np.nan),
                    'test_mae': metrics.get('test_mae', np.nan),
                    'test_r2': metrics.get('test_r2', np.nan),
                    'train_rmse': metrics.get('train_rmse', np.nan),
                    'train_r2': metrics.get('train_r2', np.nan),
                    'n_estimators': params.get('n_estimators', np.nan),
                    'max_depth': params.get('max_depth', np.nan),
                    'learning_rate': params.get('learning_rate', np.nan)
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Sort by test RMSE (lower is better)
            comparison_df = comparison_df.sort_values('test_rmse')
            
            return comparison_df
            
        except Exception as e:
            self.logger.error(f"Failed to compare models: {e}")
            return pd.DataFrame()
    
    def transition_model_stage(self, model_registry_name: str, version: str,
                              stage: str) -> bool:
        """Transition model to a different stage."""
        
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_registry_name,
                version=version,
                stage=stage
            )
            
            self.logger.info(f"Transitioned model {model_registry_name} version {version} to {stage}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to transition model stage: {e}")
            return False
    
    def start_mlflow_server(self, host: str = "localhost", port: int = 5000):
        """Start MLflow tracking server."""
        
        try:
            import subprocess
            import threading
            import time
            
            def run_server():
                subprocess.run([
                    "mlflow", "server", 
                    "--host", host, 
                    "--port", str(port),
                    "--backend-store-uri", "sqlite:///mlflow.db"
                ])
            
            # Run server in background thread
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            # Give server time to start
            time.sleep(3)
            
            self.logger.info(f"MLflow server started at http://{host}:{port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start MLflow server: {e}")

if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from models.demand_model import DemandForecastingModel
    from data.data_ingestion import DataIngestion
    from data.preprocessing import DataPreprocessor
    
    # Load and preprocess data
    ingestion = DataIngestion(data_dir="../../data/raw")
    train_df, test_df = ingestion.load_training_data()
    
    preprocessor = DataPreprocessor()
    X_train, y_train, X_test, y_test = preprocessor.preprocess_training_data(train_df, test_df)
    
    # Train model
    model_trainer = DemandForecastingModel()
    training_results = model_trainer.train_models(X_train, y_train)
    
    # Initialize MLflow manager
    mlflow_manager = MLflowManager()
    
    # Log best model
    best_model_name = model_trainer.best_model_name
    best_model = model_trainer.trained_models[best_model_name]
    
    training_metrics = training_results[best_model_name]
    test_metrics = model_trainer.evaluate_model(best_model_name, X_test, y_test)
    
    # Get hyperparameters
    hyperparameters = best_model.get_params()
    
    # Log to MLflow
    run_id = mlflow_manager.log_model_training(
        model_name=best_model_name,
        model=best_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        training_metrics=training_metrics,
        test_metrics=test_metrics,
        hyperparameters=hyperparameters,
        feature_names=preprocessor.feature_columns
    )
    
    # Register model
    model_version = mlflow_manager.register_model(best_model_name, run_id)
    
    print(f"Model logged and registered successfully!")
    print(f"Run ID: {run_id}")
    print(f"Model Version: {model_version}")
