import numpy as np
import pandas as pd
from datetime import timedelta
import os
from src.utils import *

def generate_synthetic_data(output_file="data/raw_rides.csv", num_rides=50000):
    """
    Generates synthetic ride data with timestamps, locations, active drivers, and cancellation flags.
    Simulates:
    - Daily seasonality (Morning/Evening peaks)
    - Weekly seasonality (Weekend spikes)
    - Geospatial hotspots (Downtown/Airport)
    - Supply-Demand dynamics (Active Drivers vs Requests)
    """
    print(f"Generating optimized synthetic data for {num_rides} rides...")
    ensure_directory("data")
    
    # Generate Timestamps (High Frequency: 5 min intervals for aggregation later)
    # Spanning 3 months
    full_date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='5T')
    
    # Create base demand curve
    # 1. Daily Pattern: Peaks at 8-9am and 5-7pm
    hour_weights = np.array([
        0.2, 0.1, 0.1, 0.1, 0.2, 0.5, 0.8, 1.2, 1.5, 1.2, # 0-9
        1.0, 0.9, 1.0, 1.0, 1.1, 1.3, 1.6, 2.0, 1.8, 1.5, # 10-19
        1.2, 1.0, 0.8, 0.5  # 20-23
    ])
    
    # 2. Weekly Pattern: Weekend > Weekday
    day_weights = np.array([1.0, 1.0, 1.0, 1.1, 1.3, 1.5, 1.4]) # Mon=0, Sun=6
    
    rides = []
    
    current_time = pd.Timestamp(START_DATE)
    end_time = pd.Timestamp(END_DATE)
    
    # Limit number of steps to avoid infinite loop
    max_steps = len(full_date_range)
    total_rides = 0
    
    for _ in range(max_steps):
        if total_rides >= num_rides:
            break
            
        # Determine number of rides in this 5-min window
        base_rate = 5 # avg rides per 5 min
        
        hour_factor = hour_weights[current_time.hour]
        day_factor = day_weights[current_time.dayofweek]
        
        # Random noise
        lambda_val = base_rate * hour_factor * day_factor * np.random.uniform(0.8, 1.2)
        n_rides_window = np.random.poisson(lambda_val)
        
        # Determine Active Drivers for this window
        # Supply lags demand slightly and has its own noise
        supply_factor = np.random.uniform(0.8, 1.2)
        
        # Peak hour shortage simulation
        if hour_factor > 1.5: 
            supply_factor = np.random.uniform(0.6, 0.9)
            
        active_drivers = max(1, int(lambda_val * supply_factor))
        
        # Define hotspots (Gaussian centers)
        hotspots = [
            (CITY_LAT_MIN + 0.05, CITY_LON_MIN + 0.05), # Center
            (CITY_LAT_MAX - 0.03, CITY_LON_MAX - 0.03), # Suburb
            (CITY_LAT_MIN + 0.08, CITY_LON_MAX - 0.08)  # Tech Park
        ]
        
        weights = [0.5, 0.3, 0.2] # Probability of picking a hotspot
        
        for _ in range(n_rides_window):
            # Pick a center
            idx = np.random.choice(range(len(hotspots)), p=weights)
            center = hotspots[idx]
            
            # Add Gaussian noise
            lat = np.clip(np.random.normal(center[0], 0.01), CITY_LAT_MIN, CITY_LAT_MAX)
            lon = np.clip(np.random.normal(center[1], 0.01), CITY_LON_MIN, CITY_LON_MAX)
            
            # Cancellations: Higher when supply < demand (proxy for wait time)
            cancel_prob = 0.05
            if active_drivers < n_rides_window: 
                cancel_prob = 0.2 # Surge induced cancellation
            
            is_cancelled = 1 if np.random.random() < cancel_prob else 0
            
            rides.append({
                'timestamp': current_time,
                'pickup_lat': lat,
                'pickup_long': lon,
                'active_drivers': active_drivers, # Supply at this time
                'is_cancelled': is_cancelled
            })
            total_rides += 1
            
            if total_rides >= num_rides:
                break
        
        current_time += pd.Timedelta(minutes=5)
        
    df = pd.DataFrame(rides)
    if 'ride_id' not in df.columns:
        df['ride_id'] = range(1, len(df) + 1)
        
    df.to_csv(output_file, index=False)
    print(f"Data generation complete. Saved to {output_file}")
