from fastapi import FastAPI
from pydantic import BaseModel
import requests
import logging

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
# 🔹 LOGGING (AIOps Monitoring)
# ===============================
logging.basicConfig(
    filename="system.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ===============================
# 🔹 FASTAPI APP
# ===============================
app = FastAPI(title="Predictive Maintenance API")

# ===============================
# 🔹 INPUT SCHEMA (FULL FIXED)
# ===============================
class SensorData(BaseModel):
    temperature: float
    vibration: float
    pressure: float
    humidity: float = 50
    timestamp: str = "2024-05-01"
    machine_id: str = "M001"

# ===============================
# 🔹 ANOMALY DETECTION
# ===============================
def detect_anomaly(data: SensorData):
    if data.temperature > 100:
        return "High temperature anomaly"
    if data.vibration > 1.5:
        return "High vibration anomaly"
    if data.pressure > 80:
        return "High pressure anomaly"
    return None

# ===============================
# 🔹 HOME ROUTE
# ===============================
@app.get("/")
def home():
    return {"message": "Predictive Maintenance API Running"}

# ===============================
# 🔹 MAIN ENDPOINT
# ===============================
@app.post("/predict-failure")
def predict_failure(data: SensorData):
    try:
        # ✅ Correct Azure payload (ALL required columns)
        payload = {
            "Inputs": {
                "input1": [{
                    "Temperature": data.temperature,
                    "Vibration": data.vibration,
                    "Pressure": data.pressure,
                    "Humidity": data.humidity,
                    "Timestamp": data.timestamp,
                    "MachineID": data.machine_id
                }]
            },
            "GlobalParameters": {}
        }

        # 🔹 Call Azure ML endpoint
        response = requests.post(
            AZURE_ENDPOINT,
            json=payload,
            headers=headers
        )

        result = response.json()
        print("AZURE RESPONSE:", result)  # debug once

        # ✅ Safe prediction extraction
        if isinstance(result, list):
            failure_prob = result[0].get("Scored Probabilities", 0)

        elif "Results" in result:
            failure_prob = result["Results"]["WebServiceOutput0"][0]["Scored Labels"]

        else:
            failure_prob = result.get("prediction", 0)

        # 🔹 LOGGING
        logging.info(f"Input: {payload} | Prediction: {failure_prob}")

        # 🔹 ANOMALY DETECTION
        anomaly = detect_anomaly(data)

        # 🔹 ALERT SYSTEM (console + log)
        if failure_prob > 0.8 or anomaly:
            alert_msg = f"⚠️ ALERT! Failure Prob: {failure_prob}, Anomaly: {anomaly}"
            print(alert_msg)
            logging.warning(alert_msg)

                               

        return {
            "status": "success",
            "prediction": round(float(failure_prob), 2),
            "anomaly": anomaly
        }

    except Exception as e:
        logging.error(str(e))
        return {
            "status": "error",
            "message": str(e)
        }

