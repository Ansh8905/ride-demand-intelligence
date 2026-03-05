import sys
import os
import warnings

warnings.filterwarnings('ignore')

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_generation import generate_synthetic_data
from src.data_processing import process_data
from src.geospatial import visualize_demand_heatmap, visualize_zone_clusters
from src.model import MultiHorizonForecaster
from src.driver_allocation import DriverAllocator
import pandas as pd
import numpy as np


def run_pipeline():
    print("=" * 60)
    print("  Ride Demand Intelligence - Full Pipeline")
    print("=" * 60)

    # 1. Data Generation
    print("\n[1/5] Data Generation...")
    if not os.path.exists("data/raw_rides.csv"):
        generate_synthetic_data(num_rides=20000)
    else:
        print("  Data already exists. Skipping generation.")

    # 2. Processing (15-min intervals, Multi-Horizon Targets)
    print("\n[2/5] Feature Engineering & Processing...")
    process_data(interval='15min')

    df = pd.read_csv("data/processed_demand.csv")
    print(f"  Loaded processed data: {df.shape}")

    # 3. Model Training (Multi-Horizon)
    print("\n[3/5] Training Multi-Horizon Forecast Models...")
    forecaster = MultiHorizonForecaster()
    metrics = forecaster.train(df)

    for horizon, m in metrics.items():
        print(f"  {horizon}: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, R2={m['R2']:.4f}")

    forecaster.save("models")

    # 4. Simulation & Allocation Strategy
    print("\n[4/5] Real-Time Optimization Simulation...")
    allocator = DriverAllocator()

    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:]
    target_time = test_df['time_bin'].iloc[0]
    print(f"  Simulating for: {target_time}")

    sample_df = df[df['time_bin'] == target_time].copy()

    if sample_df.empty:
        print("  Warning: No data for simulation time. Skipping.")
    else:
        predictions = forecaster.predict(sample_df)

        allocation_df = sample_df.copy()
        allocation_df['predicted_demand'] = predictions['15m']

        total_drivers = 150
        allocation_df = allocator.optimize_allocation(allocation_df, total_drivers)

        print("\n  Zone Allocation (Top 5):")
        cols = ['zone_id', 'predicted_demand', 'allocated_drivers', 'gap', 'surge_multiplier', 'action']
        available_cols = [c for c in cols if c in allocation_df.columns]
        print(allocation_df[available_cols].head().to_string(index=False))

        rev_metrics = allocator.simulate_revenue(allocation_df, avg_fare=15.0)
        print("\n  Projected Performance:")
        for k, v in rev_metrics.items():
            print(f"    {k}: {v}")

    # 5. Visualizations
    print("\n[5/5] Generating Visualizations...")
    raw_df = pd.read_csv("data/raw_rides.csv")
    visualize_demand_heatmap(raw_df.sample(min(1000, len(raw_df))), "demand_heatmap.html")

    if not sample_df.empty:
        allocation_df['surge_val'] = allocation_df['surge_multiplier']
        visualize_zone_clusters(allocation_df, value_col='surge_val', output_map="surge_heatmap.html")

    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("  - Models saved to models/")
    print("  - Maps: demand_heatmap.html, surge_heatmap.html")
    print("  - Run: streamlit run app.py  (for dashboard)")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
