from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from src.driver_allocation import DriverAllocator
from src.model import MultiHorizonForecaster, SurgePricingModel
from src.utils import NUM_CLUSTERS

app = Flask(__name__)

# Global State
FORECASTER = None
SURGE_MODEL = None
ALLOCATOR = None
KMEANS_MODEL = None

def load_models():
    global FORECASTER, SURGE_MODEL, ALLOCATOR, KMEANS_MODEL
    try:
        # Load Forecasting Model (LightGBM)
        FORECASTER = MultiHorizonForecaster()
        FORECASTER.load("models") # Loads all horizons
        
        # Load Clustering
        kmeans_path = "models/kmeans_model.pkl"
        if os.path.exists(kmeans_path):
            KMEANS_MODEL = joblib.load(kmeans_path)
            print(f"Loaded KMeans Model from {kmeans_path}")
            
        # Initialize Logic
        SURGE_MODEL = SurgePricingModel()
        ALLOCATOR = DriverAllocator() # Doesn't need model passed in init anymore
            
    except Exception as e:
        print(f"Error loading models: {e}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "forecaster_loaded": FORECASTER is not None})

@app.route('/predict_surge', methods=['POST'])
def predict_surge():
    """
    Returns demand forecast and surge pricing for a specific zone.
    Input: { "timestamp": "...", "zone_id": "...", "active_drivers": 10 }
    """
    if not FORECASTER:
        return jsonify({"error": "Models not loaded"}), 503
        
    try:
        data = request.json or {}
        # Parse inputs... 
        # For simplicity, we create a mock feature row for the requested zone/time
        # In production this would query Feature Store
        
        # ... logic omitted for brevity, let's focus on allocation ...
        return jsonify({"message": "Use /allocate for full optimization"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/allocate', methods=['POST'])
def get_allocation():
    """
    Returns optimal driver allocation for the whole city at a timestamp.
    """
    if not FORECASTER or not KMEANS_MODEL:
        return jsonify({"error": "Models not loaded"}), 503
    
    try:
        data = request.json or {}
        timestamp_str = data.get('timestamp')
        if timestamp_str:
            timestamp = pd.to_datetime(timestamp_str)
        else:
            timestamp = pd.Timestamp.now()
            
        print(f"Generating allocation for {timestamp}")
        
        # 1. Simulate Current State Features (All Zones)
        # Using KMeans centroids
        centroids = KMEANS_MODEL.cluster_centers_
        zones = []
        for i in range(len(centroids)):
            zone_id = str(i)
            lat, lon = centroids[i]
            
            # Mock Features (Ideally fetched from real-time DB)
            features = {
                'zone_id': zone_id,
                'zone_lat': lat,
                'zone_lon': lon,
                'hour': timestamp.hour,
                'minute': timestamp.minute,
                'day_of_week': timestamp.dayofweek,
                'is_weekend': 1 if timestamp.dayofweek >= 5 else 0,
                'month': timestamp.month,
                'hour_sin': np.sin(2 * np.pi * timestamp.hour/24),
                'hour_cos': np.cos(2 * np.pi * timestamp.hour/24),
                'day_sin': np.sin(2 * np.pi * timestamp.dayofweek/7),
                'day_cos': np.cos(2 * np.pi * timestamp.dayofweek/7),
                # Mocks
                'lag_1': np.random.poisson(15), 
                'lag_4': np.random.poisson(15),
                'lag_96': np.random.poisson(15),
                'rolling_mean_4': np.random.poisson(15)
            }
            zones.append(features)
            
        df = pd.DataFrame(zones)
        
        # 2. Predict Multi-Horizon Demand
        preds = FORECASTER.predict(df)
        df['predicted_demand'] = preds['15m'] # Use immediate horizon
        
        # 3. Optimize Allocation
        total_drivers = data.get('total_drivers', 100)
        allocation_df = ALLOCATOR.optimize_allocation(df, total_drivers)
        
        # 4. Simulate Revenue
        metrics = ALLOCATOR.simulate_revenue(allocation_df)
        
        # 5. Format Response
        result = allocation_df[['zone_id', 'predicted_demand', 'allocated_drivers', 'action', 'surge_multiplier']].to_dict(orient='records')
        
        return jsonify({
            "timestamp": timestamp.isoformat(),
            "metrics": metrics,
            "allocation": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0', port=5000)
