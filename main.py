import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_generation import generate_synthetic_data
from data_processing import process_data
from geospatial import visualize_demand_heatmap, visualize_zone_clusters
from model import MultiHorizonForecaster
from driver_allocation import DriverAllocator
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def run_pipeline():
    print("Starting Advanced Ride Demand & Surge System...")
    
    # 1. Data Generation (Supports supply + cancellation)
    if not os.path.exists("data/raw_rides.csv"):
        generate_synthetic_data(num_rides=20000) # Increased dataset
    else:
        print("Data already exists.")
        
    # 2. Processing (15-min intervals, Multi-Horizon Targets)
    process_data(interval='15T')
        
    df = pd.read_csv("data/processed_demand.csv")
    print(f"Loaded processed data: {df.shape}")
    
    # 3. Model Training (Multi-Horizon)
    forecaster = MultiHorizonForecaster()
    # Split
    # LightGBM handles splits internally via validation set usually, but here fit() expects train/test
    # The class separates internally if passed full dataframe? 
    # Ah, the class expects full df and splits it.
    metrics = forecaster.train(df)
    
    forecaster.save("models")
    
    # 4. Simulation & Allocation Strategy (15-min Window)
    print("\n--- Real-Time Optimization Simulation ---")
    allocator = DriverAllocator()
    
    # Pick a sample future window (e.g. first row of holdout set)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:]
    
    # Choose a specific timestamp to simulate
    target_time = test_df['time_bin'].iloc[0]
    print(f" Simulating: {target_time}")
    
    # Get all zones for this time
    sample_df = df[df['time_bin'] == target_time].copy()
    
    if sample_df.empty:
        print("Error: No data for simulation time.")
        return

    # Predict Next 15m, 30m, 60m
    predictions = forecaster.predict(sample_df) # returns dict {'15m': [], ...}
    
    # Focus on immediate 15m horizon for allocation
    allocation_df = sample_df.copy()
    allocation_df['predicted_demand'] = predictions['15m']
    
    # Optimize Drivers
    total_drivers = 150 # Simulated fleet
    allocation_df = allocator.optimize_allocation(allocation_df, total_drivers)
    
    print("\nSample Gap Analysis (Top Zones):")
    cols = ['zone_id', 'predicted_demand', 'allocated_drivers', 'gap', 'surge_multiplier', 'action']
    print(allocation_df[cols].head())
    
    # Revenue Simulation
    rev_metrics = allocator.simulate_revenue(allocation_df, avg_fare=15.0)
    print("\nProjected Performance:")
    for k, v in rev_metrics.items():
        print(f" - {k}: {v}")
    
    # 5. Visualizations
    raw_df = pd.read_csv("data/raw_rides.csv")
    visualize_demand_heatmap(raw_df.sample(min(1000, len(raw_df))), "demand_heatmap.html")
    
    # Zone Cluster Map (Aggregated Demand or Surge?)
    # Let's visualize Surge Multiplier logic
    allocation_df['surge_val'] = allocation_df['surge_multiplier']
    visualize_zone_clusters(allocation_df, value_col='surge_val', output_map="surge_heatmap.html")
    
    print("\nSystem upgrade complete.")
    print("Multi-Horizon Forecasting models saved.")
    print("Surge Pricing simulation enabled.")

if __name__ == "__main__":
    run_pipeline()
