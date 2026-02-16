from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import uvicorn
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from src.driver_allocation import DriverAllocator
from src.model import MultiHorizonForecaster, SurgePricingModel
from src.utils import NUM_CLUSTERS

# Define Request Models
class AllocationRequest(BaseModel):
    timestamp: str = None
    total_drivers: int = 100

class PredictionRequest(BaseModel):
    timestamp: str
    zone_id: str
    active_drivers: int = 10

# Global State
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Models on Startup
    try:
        models['forecaster'] = MultiHorizonForecaster()
        models['forecaster'].load("models")
        
        kmeans_path = "models/kmeans_model.pkl"
        if os.path.exists(kmeans_path):
            models['kmeans'] = joblib.load(kmeans_path)
            print(f"Loaded KMeans from {kmeans_path}")
            
        models['allocator'] = DriverAllocator()
        models['surge_model'] = SurgePricingModel()
        
        print("All models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        
    yield
    
    # Cleanup
    models.clear()

app = FastAPI(title="Ride Demand AI API", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": list(models.keys())}

@app.post("/predict")
async def predict_demand(request: PredictionRequest):
    """
    Predicts demand for a single zone.
    """
    if 'forecaster' not in models:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    # In a real app, we would fetch current features from a Feature Store using zone_id
    # Here, we mock the feature vector
    try:
        ts = pd.to_datetime(request.timestamp)
        # Mock features
        mock_features = pd.DataFrame([{
            'hour': ts.hour,
            'minute': ts.minute,
            'day_of_week': ts.dayofweek,
            'is_weekend': 1 if ts.dayofweek >= 5 else 0,
            'month': ts.month,
            'hour_sin': np.sin(2 * np.pi * ts.hour/24),
            'hour_cos': np.cos(2 * np.pi * ts.hour/24),
            'day_sin': np.sin(2 * np.pi * ts.dayofweek/7),
            'day_cos': np.cos(2 * np.pi * ts.dayofweek/7),
            'lag_1': 10, 'lag_4': 10, 'lag_96': 10, 'rolling_mean_4': 10,
            'zone_lat': 0, 'zone_lon': 0, # Should lookup from zone map
            'active_drivers': request.active_drivers
        }])
        
        preds = models['forecaster'].predict(mock_features)
        
        # Calculate Surge
        surge = models['surge_model'].calculate_surge_multiplier(preds['15m'][0], request.active_drivers)
        
        return {
            "zone_id": request.zone_id,
            "forecast": {k: float(v[0]) for k, v in preds.items()},
            "surge_multiplier": surge
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/allocate")
async def allocate_drivers(request: AllocationRequest):
    """
    Optimizes driver allocation across all zones.
    """
    if 'kmeans' not in models or 'forecaster' not in models:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    try:
        ts = pd.to_datetime(request.timestamp) if request.timestamp else pd.Timestamp.now()
        
        # 1. Simulate State (All Zones)
        centroids = models['kmeans'].cluster_centers_
        zones = []
        for i in range(len(centroids)):
            lat, lon = centroids[i]
            # Mock Features
            zones.append({
                'zone_id': str(i),
                'zone_lat': lat, 'zone_lon': lon,
                'hour': ts.hour, 'minute': ts.minute,
                'day_of_week': ts.dayofweek,
                'is_weekend': 1 if ts.dayofweek >= 5 else 0,
                'month': ts.month,
                'hour_sin': np.sin(2 * np.pi * ts.hour/24),
                'hour_cos': np.cos(2 * np.pi * ts.hour/24),
                'day_sin': np.sin(2 * np.pi * ts.dayofweek/7),
                'day_cos': np.cos(2 * np.pi * ts.dayofweek/7),
                'lag_1': np.random.poisson(15), 
                'lag_4': np.random.poisson(15),
                'lag_96': np.random.poisson(15),
                'rolling_mean_4': np.random.poisson(15)
            })
            
        df = pd.DataFrame(zones)
        
        # 2. Predict
        preds = models['forecaster'].predict(df)
        df['predicted_demand'] = preds['15m']
        
        # 3. Optimize
        df = models['allocator'].optimize_allocation(df, request.total_drivers)
        
        # 4. Revenue Sim
        metrics = models['allocator'].simulate_revenue(df)
        
        result = df[['zone_id', 'predicted_demand', 'allocated_drivers', 'action', 'surge_multiplier']].to_dict(orient='records')
        
        return {
            "timestamp": ts.isoformat(),
            "metrics": metrics,
            "allocation": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
