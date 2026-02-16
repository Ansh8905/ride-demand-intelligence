import os
import pandas as pd
import numpy as np

# Random seed for reproducibility
RANDOM_SEED = 42

# Geographically focused on a fictional city center (roughly SF)
CITY_LAT_MIN = 37.70
CITY_LAT_MAX = 37.82
CITY_LON_MIN = -122.52
CITY_LON_MAX = -122.35

# Date range for historical data
START_DATE = '2023-01-01'
END_DATE = '2023-03-31'

# Zone Grid Size (approx 1km x 1km or similar)
GRID_SIZE_LAT = 0.01
GRID_SIZE_LON = 0.01
NUM_CLUSTERS = 20 # Number of demand zones (clusters)

def ensure_directory(path):
    if not path: # Handle empty string (current directory)
        return
    if not os.path.exists(path):
        os.makedirs(path)
