<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.55-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/LightGBM-3.3+-9ACD32?style=for-the-badge&logo=lightgbm&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-5.x-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
</p>

<h1 align="center">⚡ RidePulse — Demand Intelligence Platform</h1>

<p align="center">
  <strong>End-to-end ride demand forecasting, surge pricing, fleet optimization & geospatial analytics</strong><br>
  Powered by Multi-Horizon LightGBM · K-Means Clustering · Dynamic Pricing Engine<br>
  Premium Glassmorphism Dashboard with 6 Interactive Pages
</p>

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Architecture](#-architecture)
3. [Directory Structure](#-directory-structure)
4. [Features in Detail](#-features-in-detail)
5. [Installation & Setup](#-installation--setup)
6. [Running the Pipeline](#-running-the-pipeline)
7. [Launching the Dashboard](#-launching-the-dashboard)
8. [FastAPI REST Endpoints](#-fastapi-rest-endpoints)
9. [Module-by-Module Breakdown](#-module-by-module-breakdown)
10. [Machine Learning Models](#-machine-learning-models)
11. [Data Pipeline](#-data-pipeline)
12. [Dashboard Pages](#-dashboard-pages)
13. [Configuration & Tuning](#-configuration--tuning)
14. [Troubleshooting](#-troubleshooting)
15. [Tech Stack](#-tech-stack)
16. [License](#-license)

---

## 🔭 Overview

**RidePulse** is a production-grade ride demand intelligence platform that simulates, predicts, and optimizes ride-hailing operations across a metropolitan area (modeled on San Francisco's geography).

The platform solves three core problems:

| Problem | Solution | Algorithm |
|---------|----------|-----------|
| **"How many rides will we get in each zone in 15/30/60 min?"** | Multi-horizon demand forecasting | LightGBM Gradient Boosting (3 models) |
| **"Which zones need more drivers right now?"** | Real-time fleet optimization | Demand-Supply Ratio allocation |
| **"What should the surge price be?"** | Dynamic surge pricing | DSR-based pricing with floor/cap |

The system generates **90 days** of realistic synthetic ride data (Jan–Mar 2023), engineers 20+ features, trains multi-horizon ML models, clusters the city into **20 zones** via K-Means, and powers a **premium Streamlit dashboard** with 6 interactive pages.

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                          │
│  ┌────────────────────────┐   ┌─────────────────────────────┐   │
│  │ Streamlit Dashboard    │   │ FastAPI REST API             │   │
│  │ (6 pages, Plotly,      │   │ /health · /predict ·        │   │
│  │  Folium, Glassmorphism)│   │ /allocate                   │   │
│  └──────────┬─────────────┘   └──────────┬──────────────────┘   │
│             │                            │                      │
│  ┌──────────▼────────────────────────────▼──────────────────┐   │
│  │              INTELLIGENCE LAYER                           │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐  │   │
│  │  │ Multi-Horizon │ │ Surge Pricing│ │ Driver Allocation│  │   │
│  │  │ Forecaster    │ │ Engine       │ │ Optimizer        │  │   │
│  │  │ (LightGBM x3) │ │ (DSR-based) │ │ (DSR + Actions) │  │   │
│  │  └──────┬───────┘ └──────┬───────┘ └──────┬───────────┘  │   │
│  │         │                │                │               │   │
│  │  ┌──────▼────────────────▼────────────────▼───────────┐   │   │
│  │  │           FEATURE ENGINEERING PIPELINE              │   │   │
│  │  │  Time features · Lag features · Rolling stats ·    │   │   │
│  │  │  Cyclical encoding · Zone aggregation              │   │   │
│  │  └──────────────────────┬─────────────────────────────┘   │   │
│  └─────────────────────────┼─────────────────────────────────┘   │
│  ┌─────────────────────────▼─────────────────────────────────┐   │
│  │              GEOSPATIAL LAYER                              │   │
│  │  K-Means (20 clusters) · Heatmaps · Centroid analysis     │   │
│  └─────────────────────────┬─────────────────────────────────┘   │
│  ┌─────────────────────────▼─────────────────────────────────┐   │
│  │              DATA LAYER                                    │   │
│  │  Synthetic generator · 90-day window · Daily/weekly        │   │
│  │  seasonality · Hotspot patterns · Supply-demand dynamics   │   │
│  └───────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure

```
ride_demand_prediction/
│
├── app.py                          # Streamlit dashboard (6 pages, premium UI)
├── app_fastapi.py                  # FastAPI REST API server
├── main.py                         # CLI pipeline runner (data → train → simulate)
├── create_notebook.py              # Jupyter notebook generator
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── data/
│   ├── raw_rides.csv               # ~300K synthetic ride records
│   └── processed_demand.csv        # ~168K aggregated demand rows (15-min bins)
│
├── models/
│   ├── lgbm_15m.pkl                # LightGBM model — 15-minute horizon
│   ├── lgbm_30m.pkl                # LightGBM model — 30-minute horizon
│   ├── lgbm_60m.pkl                # LightGBM model — 60-minute horizon
│   └── kmeans_model.pkl            # K-Means geospatial clustering model
│
├── src/
│   ├── __init__.py                 # Package initializer
│   ├── data_generation.py          # Synthetic ride data generator
│   ├── data_processing.py          # Feature engineering & aggregation
│   ├── model.py                    # MultiHorizonForecaster & SurgePricingModel
│   ├── driver_allocation.py        # Fleet optimization & revenue simulation
│   ├── geospatial.py               # K-Means clustering & Folium visualization
│   ├── rl_agent.py                 # Q-Learning reinforcement learning agent
│   └── utils.py                    # Constants & utility functions
│
├── notebooks/
│   └── Ride_Demand_Prediction_Pipeline.ipynb  # Full pipeline notebook
│
├── demand_heatmap.html             # Standalone Folium demand heatmap
├── surge_heatmap.html              # Standalone Folium surge heatmap
└── zone_clusters.html              # Standalone Folium zone cluster map
```

---

## ✨ Features in Detail

### 1. Synthetic Data Generation
- **300K+ raw ride records** over 90 days (Jan 1 – Mar 31, 2023)
- Realistic **daily seasonality**: morning rush (7–9h), lunch (12–13h), evening peak (17–19h)
- **Weekly patterns**: weekday commute peaks, weekend leisure patterns
- **Geographic hotspots**: 6 predefined high-density areas (downtown, airports, hubs)
- **Supply–demand dynamics**: variable driver availability by time of day
- Columns: `timestamp`, `pickup_lat`, `pickup_long`, `dropoff_lat`, `dropoff_long`, `ride_duration`, `fare`, `demand_supply_ratio`

### 2. Feature Engineering Pipeline
- **15-minute binning**: Raw rides aggregated into zone × time-bin granularity
- **Time features**: `hour`, `minute`, `day_of_week`, `month`, `is_weekend`
- **Cyclical encoding**: `hour_sin/cos`, `day_sin/cos` — captures circular nature of time
- **Lag features**: `lag_1` (15m ago), `lag_4` (1h ago), `lag_96` (24h ago)
- **Rolling statistics**: `rolling_mean_4` (1h rolling average)
- **Multi-horizon targets**: `target_15m`, `target_30m`, `target_60m` — forward-shifted demand
- **Zone ID encoding**: Categorical zone identifiers from K-Means clustering

### 3. Multi-Horizon Forecasting
- **Three LightGBM regressors** each predicting a different time horizon
- Hyperparameters: 200 estimators, 31 leaves, learning rate 0.05, feature/bagging fraction 0.8
- Trained with silent mode (`verbose=-1`) for clean output
- Output: per-zone demand predictions at 15m, 30m, and 60m into the future

### 4. Geospatial Zone Clustering
- **K-Means with 20 clusters** applied to pickup coordinates
- Assigns every ride to a geographic zone
- Centroids used as zone centers for simulation
- Visualized with Folium CircleMarkers on dark-matter tile map

### 5. Dynamic Surge Pricing
- **Demand-Supply Ratio (DSR)** calculated per zone: `demand / max(available_drivers, 1)`
- Surge formula: `max(1.0, min(3.0, DSR × base_multiplier))`
- Floor = 1.0× (no discount below base), Cap = 3.0× (prevent price gouging)
- Applied per-zone in real-time simulation

### 6. Fleet Optimization
- **DSR-based allocation**: Drivers distributed proportionally to demand
- Minimum 1 driver per active zone
- **Gap analysis**: `gap = demand − allocated_drivers`
- **Action classification**: `balanced`, `surge_active`, `critical_shortage`, `oversupplied`
- **Revenue simulation**: `Σ(min(demand, drivers) × fare × surge)` per zone

### 7. Premium Dashboard (6 Pages)
- Glassmorphism design with animated gradients
- JetBrains Mono + Inter typography
- Plotly dark-theme charts with custom color scales
- Folium dark-matter maps with interactive overlays
- See [Dashboard Pages](#-dashboard-pages) section below

---

## 🚀 Installation & Setup

### Prerequisites
- **Python 3.10+** (tested on 3.10, 3.11, 3.12)
- **pip** package manager
- ~500MB disk space for data + models

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd ride_demand_prediction
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Dependencies Breakdown

| Package | Purpose | Version |
|---------|---------|---------|
| `pandas` | Data manipulation & aggregation | Latest |
| `numpy` | Numerical computing | Latest |
| `scikit-learn` | K-Means clustering, train/test split | Latest |
| `lightgbm` | Gradient boosting for demand prediction | Latest |
| `xgboost` | Alternative boosting (available) | Latest |
| `plotly` | Interactive charts (bar, scatter, radar, heatmap, treemap) | Latest |
| `folium` | Geographic map rendering | Latest |
| `streamlit` | Dashboard framework | 1.55+ |
| `streamlit-folium` | Folium ↔ Streamlit bridge | Latest |
| `fastapi` | REST API framework | Latest |
| `uvicorn` | ASGI server for FastAPI | Latest |
| `joblib` | Model serialization | Latest |
| `scipy` | Scientific computing utilities | Latest |
| `matplotlib` | Fallback plotting | Latest |
| `seaborn` | Statistical visualization | Latest |
| `jupyter` | Notebook support | Latest |

---

## ⚙️ Running the Pipeline

The `main.py` script executes the full end-to-end pipeline:

```bash
python main.py
```

### Pipeline Steps (Executed Sequentially)

| Step | Operation | Duration | Output |
|------|-----------|----------|--------|
| 1 | **Generate synthetic rides** | ~5s | `data/raw_rides.csv` (~300K rows) |
| 2 | **K-Means zone clustering** | ~2s | `models/kmeans_model.pkl` (20 clusters) |
| 3 | **Feature engineering** | ~10s | `data/processed_demand.csv` (~168K rows) |
| 4 | **Train multi-horizon models** | ~15s | `models/lgbm_15m.pkl`, `lgbm_30m.pkl`, `lgbm_60m.pkl` |
| 5 | **Run allocation simulation** | ~1s | Console output with zone allocations |
| 6 | **Generate Folium maps** | ~3s | `demand_heatmap.html`, `surge_heatmap.html`, `zone_clusters.html` |

**Total time: ~35 seconds** on a modern machine.

### Expected Console Output
```
============================================================
  RIDE DEMAND PREDICTION PIPELINE
============================================================

[1/6] Generating synthetic ride data...
  ✓ Generated 324,000 rides → data/raw_rides.csv

[2/6] Clustering zones (K-Means, 20 clusters)...
  ✓ 20 zone centroids computed → models/kmeans_model.pkl

[3/6] Engineering features...
  ✓ Processed 168,480 demand records → data/processed_demand.csv

[4/6] Training multi-horizon forecasters...
  ✓ 15m model → RMSE: 3.42, MAE: 2.18
  ✓ 30m model → RMSE: 3.78, MAE: 2.51
  ✓ 60m model → RMSE: 4.15, MAE: 2.89
  Models saved to models/

[5/6] Running driver allocation simulation...
  ...zone allocation table...

[6/6] Generating visualizations...
  ✓ demand_heatmap.html
  ✓ surge_heatmap.html
  ✓ zone_clusters.html

============================================================
  PIPELINE COMPLETE
============================================================
```

---

## 🎨 Launching the Dashboard

```bash
streamlit run app.py
```

Opens at **http://localhost:8501** in your browser.

### Dashboard Features
- **6 interactive pages** accessible from the sidebar
- **Sidebar simulation controls**: Fleet size, base fare, hour, day of week
- **Real-time recalculation** on parameter change
- **Glassmorphism UI** with animated gradient blobs, hover effects, and micro-interactions
- **Responsive layout** via Streamlit's wide mode

---

## 🌐 FastAPI REST Endpoints

```bash
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
```

| Method | Endpoint | Description | Request Body |
|--------|----------|-------------|--------------|
| `GET` | `/health` | Health check | — |
| `POST` | `/predict` | Get demand predictions | `{"hour": 17, "minute": 0, ...}` |
| `POST` | `/allocate` | Get fleet allocation | `{"total_drivers": 150, "zones": [...]}` |

### Example: Predict Demand
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"hour": 17, "day_of_week": 4, "zone_lat": 37.77, "zone_lon": -122.42}'
```

---

## 🧩 Module-by-Module Breakdown

### `src/data_generation.py`
**Purpose**: Create realistic synthetic ride data.
- `generate_rides(n_days=90)` → Generates rides with temporal patterns
- **Seasonality engine**: Combines hourly weights (rush hours), daily weights (weekday vs weekend), and random noise
- **Hotspot system**: 6 geographic points with higher ride probability densities
- **Supply modeling**: Base supply varies by hour (lower at night, higher during commute)
- **Output fields**: timestamp, pickup/dropoff lat-long, duration, fare, demand-supply ratio

### `src/data_processing.py`
**Purpose**: Transform raw rides into ML-ready features.
- `process_demand_data(raw_df, kmeans)` → Full feature engineering pipeline
- **Zone assignment**: Uses KMeans model to label each ride with a zone_id
- **Temporal aggregation**: Groups by (zone_id, 15-min time bin) and counts rides = demand
- **Feature creation**: Extracts hour, minute, day_of_week, month, is_weekend from timestamp
- **Cyclical encoding**: `sin/cos` transforms for hour (period=24) and day (period=7)
- **Lag features**: Shift demand by 1, 4, and 96 periods per zone
- **Rolling features**: 4-period rolling mean per zone
- **Target creation**: Forward-shift demand by 1 (15m), 2 (30m), 4 (60m) periods per zone
- Uses `ffill()` for NaN handling after lag/roll operations

### `src/model.py`
**Purpose**: Multi-horizon forecasting and surge pricing.
- `MultiHorizonForecaster` class:
  - Trains 3 independent LightGBM regressors (15m, 30m, 60m horizons)
  - `train(df)`: Splits features/targets, fits all 3 models, prints metrics
  - `predict(df)`: Returns dict `{'15m': array, '30m': array, '60m': array}`
  - `save(path)` / `load(path)`: Pickle serialization via joblib
- `SurgePricingModel` class:
  - `calculate_surge(demand, supply)`: DSR-based surge with floor=1.0, cap=3.0
  - Uses step function: below 0.8 DSR=1.0, linear scale above

### `src/driver_allocation.py`
**Purpose**: Optimize where drivers should be positioned.
- `DriverAllocator` class:
  - `optimize_allocation(zone_df, total_drivers)`:
    - Computes DSR per zone
    - Allocates drivers proportional to demand (weighted by DSR)
    - Ensures minimum 1 driver per active zone
    - Calculates gap, surge_multiplier, status, action per zone
  - `simulate_revenue(alloc_df, avg_fare)`:
    - Revenue = Σ min(demand, drivers) × fare × surge_multiplier
    - Returns `Total_Revenue`, `Service_Level`, `Avg_Surge`, `Total_Demand`

### `src/geospatial.py`
**Purpose**: Geographic clustering and map visualization.
- `create_zone_clusters(df, n_clusters=20)`: Fits KMeans, returns model + labels
- `create_demand_heatmap(df)`: Folium HeatMap of pickup density
- `create_zone_map(kmeans)`: Folium map with colored centroid markers
- `create_surge_heatmap(alloc_df)`: Folium map with surge-colored zone circles

### `src/utils.py`
**Purpose**: Shared constants and utilities.
- City bounds: `CITY_LAT_MIN=37.70`, `CITY_LAT_MAX=37.82`, `CITY_LON_MIN=-122.52`, `CITY_LON_MAX=-122.35`
- Date range: `START_DATE='2023-01-01'`, `END_DATE='2023-03-31'`
- Grid: `LAT_BINS=20`, `LON_BINS=20`
- Clustering: `NUM_CLUSTERS=20`
- `ensure_directory(path)`: Creates directory if it doesn't exist

### `src/rl_agent.py`
**Purpose**: Reinforcement learning for driver repositioning (experimental).
- `QLearningAgent`: Tabular Q-learning with epsilon-greedy exploration
- State space: zone × hour discretization
- Action space: stay, move-north, move-south, move-east, move-west
- Not currently integrated into the main pipeline — available for extension

---

## 🤖 Machine Learning Models

### LightGBM Demand Forecaster

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 200 | Balance between accuracy and training speed |
| `num_leaves` | 31 | Default tree complexity, prevents overfitting |
| `learning_rate` | 0.05 | Slow learning for better generalization |
| `feature_fraction` | 0.8 | Feature bagging reduces correlation between trees |
| `bagging_fraction` | 0.8 | Row sampling for regularization |
| `bagging_freq` | 5 | Apply bagging every 5 iterations |
| `verbose` | -1 | Silent training output |

### Feature Set Used

```
Numerical Features:
├── hour, minute, day_of_week, month, is_weekend
├── hour_sin, hour_cos, day_sin, day_cos     (cyclical)
├── lag_1, lag_4, lag_96                       (temporal lags)
├── rolling_mean_4                             (rolling stats)
└── zone_lat, zone_lon                         (spatial)
```

### Training Split
- **80/20** temporal train-test split (first 80% of time → train, last 20% → test)
- Prevents data leakage by respecting time ordering

### Performance (Typical)

| Horizon | RMSE | MAE | Notes |
|---------|------|-----|-------|
| 15 min | 3.2–3.8 | 2.0–2.5 | Highest accuracy (nearest future) |
| 30 min | 3.5–4.2 | 2.3–2.8 | Slight degradation |
| 60 min | 3.9–4.8 | 2.6–3.2 | Most uncertainty |

---

## 📊 Data Pipeline

```
Raw Rides CSV          Feature Engineering         Multi-Horizon Models
─────────────         ─────────────────────        ──────────────────
timestamp         →   15-min binning          →    Train LightGBM (15m)
pickup_lat/long   →   Zone assignment (KMeans) →   Train LightGBM (30m)
fare, duration    →   Time features            →   Train LightGBM (60m)
                  →   Cyclical encoding         →
                  →   Lag features (1,4,96)
                  →   Rolling mean (4)
                  →   Forward targets (1,2,4)

Output: 168K rows × 20+ cols
```

---

## 🎨 Dashboard Pages

### Page 1: 📊 Command Center
The main overview page. Shows:
- **5 KPI cards**: Revenue, Service Level, Avg Surge, Total Demand, Shortage Zones
- **Demand vs Supply bar chart**: Predicted demand (gradient bars) overlaid with allocated drivers (cyan diamonds) and surge line (rose area)
- **Surge heatbar**: Horizontal sorted bar chart of zone surge multipliers with color coding (green ≤1.1, amber ≤1.5, red >1.5)
- **Zone Allocation Matrix**: Interactive data table with Zone, Demand, Drivers, Gap, Surge, Action
- **24-Hour Demand Rhythm**: Spline area chart showing average demand by hour with peak annotation

### Page 2: 🗺️ Geospatial Intel
Three-tab spatial analysis:
- **Demand Heatmap**: Folium HeatMap on CartoDB dark_matter with Plasma gradient (purple → orange → yellow)
- **Zone Clusters**: 20 K-Means centroids displayed as colored CircleMarkers with zone ID labels
- **Scatter Density**: Plotly scatter_mapbox showing individual pickup points
- **Coordinate Distributions**: Latitude and longitude histograms

### Page 3: ⚡ Surge Simulator
Interactive simulation engine:
- Configure fleet size, capacity, base fare
- **Progress-bar animation** during simulation execution
- **5 KPI cards** with simulation results
- **Multi-Horizon Radar Chart**: Polar plot comparing 15m/30m/60m predictions across top 10 zones
- **Supply-Demand Gap**: Bar chart with red (shortage) and green (surplus) bars
- **Action Distribution**: Donut chart of fleet actions (balanced, surge_active, critical_shortage, oversupplied)
- **Revenue Treemap**: Proportional area visualization by zone, colored by surge multiplier

### Page 4: 🧠 ML Observatory
Model interpretation:
- **Model Architecture Cards**: Glass cards showing tree count, leaf count, learning rate for each horizon
- **Feature Importance**: Horizontal bar chart (top 15 features) with gradient coloring
- **Correlation Matrix**: Full-feature heatmap with Inferno colorscale and rendered correlation values

### Page 5: 📈 Time Series Lab
Temporal deep-dive:
- **Day-of-Week Pattern**: Bar chart of average demand by weekday
- **Hour × Day Heatmap**: 7×24 matrix showing demand intensity (Inferno colorscale)
- **Zone-Level Time Series**: Multi-select line chart plotting hourly demand for specific zones
- **Cumulative Demand Growth**: Cumulative area chart over the full 90-day period

### Page 6: 📋 Data Explorer
Raw data investigation:
- **5 KPI cards**: Row count, feature count, zone count, day span, raw ride count
- **Processed Data tab**: Top 200 rows + full `.describe()` statistics table
- **Raw Data tab**: Top 200 raw ride records
- **Feature Distributions**: Demand histogram + average demand per zone bar chart

---

## ⚙️ Configuration & Tuning

### Simulation Parameters (Sidebar)
| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| Fleet Size | 50–500 | 150 | Total drivers to allocate across zones |
| Avg Base Fare | $5–$50 | $15 | Base fare before surge multiplier |
| Hour of Day | 0–23 | 17 (5 PM) | Simulation hour (affects demand pattern) |
| Day of Week | Mon–Sun | Friday | Simulation day (weekday/weekend patterns) |

### Model Hyperparameters
Edit `src/model.py` → `MultiHorizonForecaster.__init__()` to tune:
```python
lgb.LGBMRegressor(
    n_estimators=200,       # More trees → better fit, slower training
    num_leaves=31,           # Higher → more complex trees
    learning_rate=0.05,      # Lower → more conservative, needs more trees
    feature_fraction=0.8,    # % of features per tree
    bagging_fraction=0.8,    # % of rows per tree
    bagging_freq=5,          # How often to bag
)
```

### Zone Count
Edit `src/utils.py`:
```python
NUM_CLUSTERS = 20   # Increase for finer zones, decrease for coarser
```

### Data Volume
Edit `src/data_generation.py` → `generate_rides()`:
- Adjust `n_days` parameter
- Modify `rides_per_day` for density

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: streamlit_folium` | `pip install streamlit-folium` |
| `FileNotFoundError: data/processed_demand.csv` | Run `python main.py` first |
| `ERR_CONNECTION_REFUSED` when opening dashboard | Check if port 8501 is already in use; kill the process or use `streamlit run app.py --server.port 8502` |
| Streamlit deprecation warnings | Ensure Streamlit 1.55+; the code uses `width="stretch"` instead of deprecated `use_container_width` |
| Models show poor RMSE | Re-run `python main.py` — synthetic data has randomness; or tune hyperparameters |
| Folium maps not rendering | Ensure `streamlit-folium>=0.15.0` is installed |
| `ImportError: src.X` | Ensure `src/__init__.py` exists (created automatically) |
| Blank charts on dashboard | Verify `data/processed_demand.csv` has data; check browser console for JS errors |

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **UI Framework** | Streamlit 1.55+ | Dashboard rendering & interactivity |
| **Charts** | Plotly 5.x | Bar, line, scatter, radar, heatmap, treemap, pie |
| **Maps** | Folium + streamlit-folium | HeatMap, CircleMarker, DivIcon overlays |
| **ML Training** | LightGBM | Gradient boosted tree regressors |
| **Clustering** | scikit-learn KMeans | Geospatial zone creation |
| **API** | FastAPI + Uvicorn | RESTful prediction & allocation endpoints |
| **Data** | pandas + NumPy | Aggregation, feature engineering, manipulation |
| **Serialization** | joblib | Model save/load to `.pkl` files |
| **Styling** | Custom CSS | Glassmorphism, animations, JetBrains Mono, Inter |
| **Language** | Python 3.10+ | End-to-end implementation |

---

## 📄 License

This project is for educational and demonstration purposes. Built as a full-stack ML engineering showcase.

---

<p align="center">
  <sub>Engineered with precision · Visualized with craft · Optimized with intelligence</sub>
</p>


