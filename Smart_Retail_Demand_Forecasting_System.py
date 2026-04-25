from fastapi import FastAPI
from pydantic import BaseModel
import requests
import numpy as np
from typing import List, Dict

# ===============================
# 🔹 CONFIGURATION
# ===============================
AZURE_ENDPOINT = ""
API_KEY = ""

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# ===============================
# 🔹 FASTAPI APP
# ===============================
app = FastAPI(title="Smart Retail Demand Forecasting API")

# ===============================
# 🔹 INPUT SCHEMA (GOOD FOR MARKS)
# ===============================
class DemandInput(BaseModel):
    Date: str
    ProductID: str
    Category: str
    Region: str
    Price: float
    Discount: float
    Holiday: int

# ===============================
# 🔹 DRIFT DETECTION
# ===============================
def detect_drift(data: DemandInput) -> str:
    if data.Price > 100000:
        return "Drift detected: Price unusually high"
    if data.Discount > 80:
        return "Drift detected: Discount unusually high"
    return "No drift detected"

# ===============================
# 🔹 METRICS CALCULATION
# ===============================
def calculate_metrics(actual: List[float], predicted: List[float]) -> Dict:
    actual = np.array(actual)
    predicted = np.array(predicted)

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2)
    }

# ===============================
# 🔹 HOME ROUTE
# ===============================
@app.get("/")
def home():
    return {"message": "Retail Demand Forecast API Running"}

# ===============================
# 🔹 PREDICTION ENDPOINT
# ===============================
@app.post("/predict-demand")
def predict_demand(data: DemandInput):
    try:
        payload = {
        "Inputs": {
            "input1": [data.dict()]
        },
        "GlobalParameters": {}

        }
        # Call Azure ML endpoint
        response = requests.post(
        AZURE_ENDPOINT,
        json=payload,
        headers=headers
)
        result = response.json()

        # Drift detection
        drift_status = detect_drift(data)

        return {
            "status": "success",
            "input": payload,
            "prediction": result,
            "drift_status": drift_status
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# ===============================
# 🔹 METRICS ENDPOINT (FOR EVALUATION)
# ===============================
@app.post("/evaluate")
def evaluate(data: Dict):
    """
    Input format:
    {
        "actual": [100, 120, 130],
        "predicted": [110, 115, 140]
    }
    """
    try:
        actual = data["actual"]
        predicted = data["predicted"]

        metrics = calculate_metrics(actual, predicted)

        return {
            "status": "success",
            "metrics": metrics
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
