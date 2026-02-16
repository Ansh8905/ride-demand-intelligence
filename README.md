# Ride Demand Prediction & Driver Allocation System

An end-to-end Machine Learning system to forecast ride demand across different city zones and optimize driver allocation.

## Project Structure

```
├── data/                   # Raw and processed data (generated)
├── models/                 # Saved machine learning models
├── notebooks/              # Jupyter Notebook for analysis and demonstration
│   └── Ride_Demand_Prediction_Pipeline.ipynb
├── src/                    # Modular Python source code
│   ├── data_generation.py  # Synthetic data generation
│   ├── data_processing.py  # Feature engineering and aggregation
│   ├── geospatial.py       # geospatial visualization and zone creation
│   ├── model.py            # Model training and evaluation (XGBoost/RF)
│   └── driver_allocation.py # Logic for allocating drivers
├── main.py                 # Main pipeline script (CLI)
├── create_notebook.py      # Script to generate the Jupyter Notebook
└── requirements.txt        # Python dependencies
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: On Windows, installing GeoPandas can be tricky. If `pip install geopandas` fails, try using Conda or download pre-built binaries (whl).*

## Usage

### Run the Full Pipeline
To generate data, train models, and visualize results in one go:
```bash
python main.py
```
This will:
- Generate synthetic rides in `data/raw_rides.csv`.
- Process features into `data/processed_demand.csv`.
- Train XGBoost models and save to `models/best_model.pkl`.
- Generate HTML maps (`demand_heatmap.html`, `zone_clusters.html`).
- Print sample driver allocation.

### Jupyter Notebook
Launch Jupyter Notebook to explore the steps interactively:
```bash
jupyter notebook notebooks/Ride_Demand_Prediction_Pipeline.ipynb
```
(If the notebook file is missing, run `python create_notebook.py` to generate it).

## Key Features (V2.0 - Intelligent System)
- **Multi-Horizon Forecasting**: LightGBM models predicting demand for 15m, 30m, and 60m ahead.
- **Dynamic Surge Pricing**: Real-time surge multiplier calculated from Supply-Demand Ratio (DSR).
- **Supply-Demand Gap Modeling**: Identifies shortage and surplus zones to guide driver reallocation.
- **Revenue Simulation**: projects financial impact of allocation strategies.
- **Geospatial Clustering**: K-Means centroids for dynamic zoning.
- **REST API**: Flask application for real-time inference and optimization.

## Run API
To start the prediction server with FastAPI (High Performance):
```bash
python app_fastapi.py
```
Outputs:
- Docs: `http://localhost:8000/docs`
- `POST /allocate`: Get optimal driver allocation.

