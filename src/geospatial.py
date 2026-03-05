try:
    import geopandas as gpd
    from shapely.geometry import Polygon, Point
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    print("Warning: GeoPandas not installed. Spatial join and polygon features will be limited.")

import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
import os
import joblib
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi

try:
    from src.utils import *
except ImportError:
    from utils import *

def create_zone_geodataframe(city_lat_min=CITY_LAT_MIN, city_lat_max=CITY_LAT_MAX, 
                             city_lon_min=CITY_LON_MIN, city_lon_max=CITY_LON_MAX, 
                             grid_size_lat=GRID_SIZE_LAT, grid_size_lon=GRID_SIZE_LON):
    """
    Creates a GeoDataFrame of grid zones.
    """
    if not HAS_GEOPANDAS:
        print("GeoPandas not available. returning None.")
        return None

    zones = []
    
    # Generate grid
    lat_steps = int((city_lat_max - city_lat_min) / grid_size_lat)
    lon_steps = int((city_lon_max - city_lon_min) / grid_size_lon)
    
    for i in range(lat_steps):
        for j in range(lon_steps):
            # Calculate coordinates
            lat_start = city_lat_min + i * grid_size_lat
            lat_end = city_lat_min + (i + 1) * grid_size_lat
            lon_start = city_lon_min + j * grid_size_lon
            lon_end = city_lon_min + (j + 1) * grid_size_lon

            # Create polygon
            poly = Polygon([
                (lon_start, lat_start),
                (lon_end, lat_start),
                (lon_end, lat_end),
                (lon_start, lat_end)
            ])
            
            zone_id = f"{i}_{j}"
            zones.append({'zone_id': zone_id, 'geometry': poly})
            
    gdf = gpd.GeoDataFrame(zones, crs="EPSG:4326")
    return gdf

def train_kmeans_clustering(df, n_clusters=NUM_CLUSTERS):
    """
    Trains K-Means clustering on pickup locations to identify demand zones.
    """
    print(f"Training KMeans Clustering with {n_clusters} clusters...")
    coords = df[['pickup_lat', 'pickup_long']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(coords)
    return kmeans

def save_clustering_model(model, filepath="models/kmeans_model.pkl"):
    ensure_directory(os.path.dirname(filepath))
    joblib.dump(model, filepath)
    print(f"Clustering model saved to {filepath}")

def load_clustering_model(filepath="models/kmeans_model.pkl"):
    if os.path.exists(filepath):
        return joblib.load(filepath)
    else:
        print("Clustering model not found.")
        return None

def assign_clusters(df, model):
    """
    Assigns cluster labels to 'zone_id' column.
    """
    coords = df[['pickup_lat', 'pickup_long']].values
    labels = model.predict(coords)
    df['zone_id'] = labels.astype(str) # ensure string for consistency
    return df

def visualize_demand_heatmap(df, output_map="demand_heatmap.html"):
    """
    Generates a Folium heatmap of ride demand.
    Expects df with 'pickup_lat', 'pickup_long'.
    """
    print("Generating Demand Heatmap...")
    
    # Ensure dir
    output_dir = os.path.dirname(output_map)
    if output_dir:
        ensure_directory(output_dir)

    m = folium.Map(location=[(CITY_LAT_MIN + CITY_LAT_MAX)/2, (CITY_LON_MIN + CITY_LON_MAX)/2], zoom_start=12)
    
    heat_data = [[row['pickup_lat'], row['pickup_long']] for index, row in df.iterrows()]
    HeatMap(heat_data).add_to(m)
    
    m.save(output_map)
    print(f"Heatmap saved to {output_map}")

def visualize_zone_clusters(df, value_col='demand', output_map="zone_clusters.html"):
    """
    Visualizes demand intensity per cluster centroid using CircleMarkers.
    """
    print("Generating Zone Cluster Map...")
    
    # Ensure dir
    ensure_directory(os.path.dirname(output_map))

    # Center map
    center_lat = (CITY_LAT_MIN + CITY_LAT_MAX) / 2
    center_lon = (CITY_LON_MIN + CITY_LON_MAX) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Add Markers
    for idx, row in df.iterrows():
        # Radius proportional to demand (scaled)
        # simplistic scaling
        radius = np.sqrt(row[value_col]) * 2 if row[value_col] > 0 else 5
        
        folium.CircleMarker(
            location=[row['zone_lat'], row['zone_lon']],
            radius=radius,
            popup=f"Zone: {row['zone_id']}<br>Demand: {row[value_col]}",
            color='crimson',
            fill=True,
            fill_color='crimson'
        ).add_to(m)
    
    m.save(output_map)
    print(f"Zone cluster map saved to {output_map}")
