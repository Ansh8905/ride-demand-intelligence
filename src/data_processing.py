import pandas as pd
import numpy as np
import os
import math
from src.utils import *
from src.geospatial import train_kmeans_clustering, save_clustering_model, load_clustering_model, assign_clusters

def process_data(input_file="data/raw_rides.csv", output_file="data/processed_demand.csv", interval='15T'):
    """
    Cleans data, creates CLUSTER zones, aggregates demand, adds time features.
    Interval: Aggregation window (default 15 minutes).
    """
    print(f"Processing Data with Geospatial Clustering ({interval} interval)...")
    ensure_directory("data")
    
    # 1. Load Data
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        return

    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2. Assign Clusters (Geospatial Feature Engineering)
    kmeans_path = "models/kmeans_model.pkl"
    kmeans_model = load_clustering_model(kmeans_path)
    
    if kmeans_model is None:
        print("Training new KMeans clustering model on raw data...")
        kmeans_model = train_kmeans_clustering(df, n_clusters=NUM_CLUSTERS)
        save_clustering_model(kmeans_model, kmeans_path)
        
    labels = kmeans_model.predict(df[['pickup_lat', 'pickup_long']].values)
    df['zone_id'] = labels.astype(str)
    
    # 3. Time Aggregation
    df['time_bin'] = df['timestamp'].dt.floor(interval)
    
    # Aggregate demand: count rides per zone per interval
    # Also sum active_drivers if present (taking mean for the interval)
    agg_funcs = {'ride_id': 'count'}
    if 'active_drivers' in df.columns:
        agg_funcs['active_drivers'] = 'mean'
    if 'is_cancelled' in df.columns:
        agg_funcs['is_cancelled'] = 'sum'
        
    demand_df = df.groupby(['time_bin', 'zone_id']).agg(agg_funcs).reset_index()
    demand_df.rename(columns={'ride_id': 'demand'}, inplace=True)
    
    # Create Full Grid (Time x Zone)
    zones = demand_df['zone_id'].unique()
    min_time = demand_df['time_bin'].min()
    max_time = demand_df['time_bin'].max()
    full_time_range = pd.date_range(start=min_time, end=max_time, freq=interval)
    
    mi = pd.MultiIndex.from_product([full_time_range, zones], names=['time_bin', 'zone_id'])
    full_df = pd.DataFrame(index=mi).reset_index()
    
    processed_df = pd.merge(full_df, demand_df, on=['time_bin', 'zone_id'], how='left')
    processed_df['demand'] = processed_df['demand'].fillna(0)
    
    # Fill missing supply/cancellation with 0 (or mean?) 
    # For supply, 0 is dangerous, use forward fill or mean imputation
    if 'active_drivers' in processed_df.columns:
         processed_df['active_drivers'] = processed_df['active_drivers'].fillna(method='ffill').fillna(0)
    if 'is_cancelled' in processed_df.columns:
        processed_df['is_cancelled'] = processed_df['is_cancelled'].fillna(0)

    # 4. Feature Engineering
    processed_df['hour'] = processed_df['time_bin'].dt.hour
    processed_df['minute'] = processed_df['time_bin'].dt.minute
    processed_df['day_of_week'] = processed_df['time_bin'].dt.dayofweek
    processed_df['is_weekend'] = processed_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    processed_df['month'] = processed_df['time_bin'].dt.month
    
    # Cyclic Time Features (Fourier)
    processed_df['hour_sin'] = np.sin(2 * np.pi * processed_df['hour'] / 24)
    processed_df['hour_cos'] = np.cos(2 * np.pi * processed_df['hour'] / 24)
    processed_df['day_sin'] = np.sin(2 * np.pi * processed_df['day_of_week'] / 7)
    processed_df['day_cos'] = np.cos(2 * np.pi * processed_df['day_of_week'] / 7)
    
    # Lag Features (Short term and Long term)
    processed_df = processed_df.sort_values(by=['zone_id', 'time_bin'])
    
    # Lag 1 step (15 min), 4 steps (1 hour), 96 steps (24 hours)
    processed_df['lag_1'] = processed_df.groupby('zone_id')['demand'].shift(1)
    processed_df['lag_4'] = processed_df.groupby('zone_id')['demand'].shift(4)
    processed_df['lag_96'] = processed_df.groupby('zone_id')['demand'].shift(96) # 24h
    
    # Rolling stats
    processed_df['rolling_mean_4'] = processed_df.groupby('zone_id')['demand'].transform(lambda x: x.rolling(window=4).mean())
    
    # Multi-Horizon Targets
    # T+1 (15m), T+2 (30m), T+4 (60m)
    processed_df['target_15m'] = processed_df.groupby('zone_id')['demand'].shift(-1)
    processed_df['target_30m'] = processed_df.groupby('zone_id')['demand'].shift(-2)
    processed_df['target_60m'] = processed_df.groupby('zone_id')['demand'].shift(-4)
    
    # Supply-Demand Gap
    # Gap = Demand - Active Drivers
    # Positive Gap = Shortage (Surge should trigger)
    # Negative Gap = Surplus (Oversupply)
    if 'active_drivers' in processed_df.columns:
        processed_df['gap'] = processed_df['demand'] - processed_df['active_drivers']
    
    # Zone Centroids
    centroids = kmeans_model.cluster_centers_
    zone_centroid_map = {str(i): centroids[i] for i in range(len(centroids))}
    processed_df['zone_lat'] = processed_df['zone_id'].apply(lambda x: zone_centroid_map[x][0])
    processed_df['zone_lon'] = processed_df['zone_id'].apply(lambda x: zone_centroid_map[x][1])
    
    # Drop rows with NaN (lags/targets)
    processed_df = processed_df.dropna()
    
    processed_df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    
if __name__ == "__main__":
    process_data()
