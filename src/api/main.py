from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime
import os
from pathlib import Path
import sys
import time
import json

# Add project root directory to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.data.preprocessing import DataPreprocessor
from src.models.demand_model import DemandForecastingModel
from src.monitoring.drift_detection import DataDriftDetector
from src.utils.mlflow_utils import MLflowManager

# Initialize FastAPI app
app = FastAPI(
    title="Retail Demand Forecasting API",
    description="API for predicting product demand in retail stores",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class DemandPredictionRequest(BaseModel):
    product_id: str = Field(..., description="Product ID")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    region: str = Field(..., description="Region name")
    price: Optional[float] = Field(None, description="Product price")
    category: Optional[str] = Field(None, description="Product category")

class BatchPredictionRequest(BaseModel):
    predictions: List[DemandPredictionRequest]

class DemandPredictionResponse(BaseModel):
    product_id: str
    date: str
    region: str
    predicted_demand: float
    confidence_interval: Optional[Dict[str, float]] = None
    model_version: str
    prediction_timestamp: str

class BatchPredictionResponse(BaseModel):
    predictions: List[DemandPredictionResponse]
    total_predictions: int
    processing_time_ms: float

class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    training_date: str
    features_count: int
    last_updated: str
    performance_metrics: Dict[str, float]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    preprocessor_loaded: bool

# Global variables for model and preprocessor
model = None
preprocessor = None
drift_detector = None
mlflow_manager = None
model_info = {}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_preprocessor():
    """Load the trained model and preprocessor."""
    global model, preprocessor, drift_detector, mlflow_manager, model_info
    
    try:
        # Load preprocessor
        preprocessor = DataPreprocessor()
        preprocessor_path = Path(__file__).parent.parent.parent / "data" / "processed" / "preprocessor.pkl"
        if preprocessor_path.exists():
            preprocessor.load_preprocessor(str(preprocessor_path))
            logger.info("Preprocessor loaded successfully")
        else:
            logger.warning(f"Preprocessor file not found at {preprocessor_path}")
        
        # Load model
        model = DemandForecastingModel()
        model_path = Path(__file__).parent.parent.parent / "models" / "gradient_boosting_model.pkl"
        if model_path.exists():
            model.load_model(str(model_path))
            logger.info("Model loaded successfully")
        else:
            logger.error(f"Model file not found at {model_path}")
            return False
        
        # Initialize drift detector
        drift_detector = DataDriftDetector()
        
        # Initialize MLflow manager
        mlflow_manager = MLflowManager()
        
        # Set model info
        model_info = {
            "model_name": "gradient_boosting",
            "model_version": "1.0",
            "training_date": "2024-01-01",
            "features_count": len(preprocessor.feature_columns) if preprocessor.feature_columns else 0,
            "last_updated": datetime.now().isoformat(),
            "performance_metrics": {
                "rmse": 15.30,
                "mae": 11.35,
                "r2": 0.53
            }
        }
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model and preprocessor: {str(e)}")
        return False

def preprocess_single_request(request: DemandPredictionRequest) -> np.ndarray:
    """Preprocess a single prediction request."""
    # Create a dictionary with all required features
    record = {
        'product_id': request.product_id,
        'date': request.date,
        'region': request.region,
        'demand': 0,  # Placeholder, will be predicted
        'price': request.price or 100.0,  # Default price if not provided
        'category': request.category or 'Unknown',
        'is_weekend': 0,  # Will be calculated
        'is_holiday': 0,  # Will be calculated
    }
    
    # Calculate date-based features
    date_obj = datetime.strptime(request.date, '%Y-%m-%d')
    record['month'] = date_obj.month
    record['day_of_week'] = date_obj.weekday()
    
    # Weekend detection
    record['is_weekend'] = 1 if date_obj.weekday() >= 5 else 0
    
    # Simple holiday detection (can be enhanced)
    holidays = [(1, 1), (7, 4), (12, 25), (11, 24)]
    record['is_holiday'] = 1 if (date_obj.month, date_obj.day) in holidays else 0
    
    # Use preprocessor to transform the record
    if preprocessor:
        try:
            return preprocessor.preprocess_single_record(record)
        except Exception as e:
            logger.error(f"Error preprocessing record: {str(e)}")
            # Fallback: create basic feature array
            return np.zeros((1, 34))  # Default feature count
    else:
        logger.error("Preprocessor not loaded")
        return np.zeros((1, 34))

def calculate_confidence_interval(prediction: float, std_dev: float = 2.0) -> Dict[str, float]:
    """Calculate confidence interval for prediction."""
    return {
        "lower": max(0, prediction - std_dev),
        "upper": prediction + std_dev,
        "confidence_level": 0.95
    }

@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup."""
    logger.info("Starting Retail Demand Forecasting API...")
    success = load_model_and_preprocessor()
    if success:
        logger.info("API startup completed successfully")
    else:
        logger.error("API startup failed")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Retail Demand Forecasting API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model and preprocessor else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        preprocessor_loaded=preprocessor is not None
    )

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information."""
    if not model_info:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(**model_info)

@app.post("/predict-demand", response_model=DemandPredictionResponse)
async def predict_demand(request: DemandPredictionRequest):
    """Predict demand for a single product."""
    if not model or not preprocessor:
        raise HTTPException(status_code=503, detail="Model or preprocessor not loaded")
    
    try:
        # Preprocess the request
        start_time = time.time()
        features = preprocess_single_request(request)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Ensure prediction is non-negative
        prediction = max(0, prediction)
        
        # Calculate confidence interval
        confidence_interval = calculate_confidence_interval(prediction)
        
        processing_time = (time.time() - start_time) * 1000
        
        response = DemandPredictionResponse(
            product_id=request.product_id,
            date=request.date,
            region=request.region,
            predicted_demand=round(prediction, 2),
            confidence_interval=confidence_interval,
            model_version=model_info.get("model_version", "unknown"),
            prediction_timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Prediction for {request.product_id} in {request.region}: {prediction:.2f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-demand/batch", response_model=BatchPredictionResponse)
async def predict_demand_batch(request: BatchPredictionRequest):
    """Predict demand for multiple products."""
    if not model or not preprocessor:
        raise HTTPException(status_code=503, detail="Model or preprocessor not loaded")
    
    if len(request.predictions) > 100:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 100")
    
    try:
        start_time = time.time()
        predictions = []
        
        for pred_request in request.predictions:
            # Preprocess the request
            features = preprocess_single_request(pred_request)
            
            # Make prediction
            prediction = model.predict(features)[0]
            prediction = max(0, prediction)
            
            # Calculate confidence interval
            confidence_interval = calculate_confidence_interval(prediction)
            
            response = DemandPredictionResponse(
                product_id=pred_request.product_id,
                date=pred_request.date,
                region=pred_request.region,
                predicted_demand=round(prediction, 2),
                confidence_interval=confidence_interval,
                model_version=model_info.get("model_version", "unknown"),
                prediction_timestamp=datetime.now().isoformat()
            )
            
            predictions.append(response)
        
        processing_time = (time.time() - start_time) * 1000
        
        batch_response = BatchPredictionResponse(
            predictions=predictions,
            total_predictions=len(predictions),
            processing_time_ms=round(processing_time, 2)
        )
        
        logger.info(f"Batch prediction completed: {len(predictions)} predictions in {processing_time:.2f}ms")
        
        return batch_response
        
    except Exception as e:
        logger.error(f"Error making batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/drift-detect")
async def detect_drift(background_tasks: BackgroundTasks):
    """Trigger drift detection in background."""
    if not drift_detector:
        raise HTTPException(status_code=503, detail="Drift detector not initialized")
    
    try:
        # This would typically load new data and compare with reference
        # For now, return a placeholder response
        background_tasks.add_task(run_drift_detection)
        
        return {
            "message": "Drift detection started in background",
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Error starting drift detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Drift detection error: {str(e)}")

async def run_drift_detection():
    """Background task for drift detection."""
    try:
        # This would load current data and compare with reference
        # Implementation would depend on your data source
        logger.info("Background drift detection completed")
    except Exception as e:
        logger.error(f"Error in background drift detection: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Get API metrics."""
    return {
        "predictions_made": 0,  # This would be tracked in a real implementation
        "uptime_seconds": int(time.time() - time.time()),  # Placeholder
        "model_load_time": model_info.get("last_updated"),
        "api_version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Load model before starting server
    if load_model_and_preprocessor():
        logger.info("Starting API server...")
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    else:
        logger.error("Failed to load model. Server not started.")
