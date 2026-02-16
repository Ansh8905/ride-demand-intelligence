import json
import os

NOTEBOOK_CONTENT = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Ride Demand Prediction & Driver Allocation System\n",
    "\n",
    "This notebook implements an end-to-end Machine Learning pipeline for ride demand prediction and driver allocation optimization.\n",
    "\n",
    "## Steps:\n",
    "1. **Data Generation**: Create synthetic ride data with realistic patterns.\n",
    "2. **Preprocessing**: Convert timestamps, aggregate demand by zone, and compute lag features.\n",
    "3. **Model Training**: Train XGBoost and Random Forest models to forecast demand.\n",
    "4. **Evaluation**: Compare models using RMSE, MAE, and R² metrics.\n",
    "5. **Driver Allocation**: Optimize fleet distribution based on predicted demand.\n",
    "6. **Visualizations**: Interactive heatmaps of demand and allocation strategies."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import folium\n",
    "from IPython.display import display, IFrame\n",
    "\n",
    "# Add src to path\n",
    "sys.path.append(os.path.abspath('../src'))\n",
    "\n",
    "from data_generation import generate_synthetic_data\n",
    "from data_processing import process_data\n",
    "from geospatial import create_zone_geodataframe, visualize_demand_heatmap, visualize_zone_clusters\n",
    "from model import RideDemandPredictor\n",
    "from driver_allocation import DriverAllocator\n",
    "\n",
    "# Ensure output directories exist\n",
    "if not os.path.exists('../data'):\n",
    "    os.makedirs('../data')\n",
    "if not os.path.exists('../models'):\n",
    "    os.makedirs('../models')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Data Generation & Processing\n",
    "We generate synthetic ride data simulating 3 months of activity in a fictional city."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate Data\n",
    "generate_synthetic_data(num_rides=50000)\n",
    "\n",
    "# Process & Engineer Features\n",
    "process_data(input_file='../data/raw_rides.csv', output_file='../data/processed_demand.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Load & Explore Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('../data/processed_demand.csv')\n",
    "print(f\"Dataset Shape: {df.shape}\")\n",
    "display(df.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Model Training & Evaluation\n",
    "We train XGBoost and Random Forest regressors to predict demand."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictor = RideDemandPredictor()\n",
    "X_train, y_train, X_test, y_test = predictor.split_data(df)\n",
    "\n",
    "print(f\"Training Set: {X_train.shape}, Test Set: {X_test.shape}\")\n",
    "\n",
    "# Train Models\n",
    "models = predictor.train_models(X_train, y_train)\n",
    "\n",
    "# Evaluate\n",
    "metrics = predictor.evaluate(models, X_test, y_test)\n",
    "\n",
    "pd.DataFrame(metrics).T"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Driver Allocation Strategy\n",
    "Using the trained model, we forecast demand and allocate drivers to minimize unmet demand."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "best_model = models['XGBoost']\n",
    "allocator = DriverAllocator(best_model)\n",
    "\n",
    "# Simulate allocation for a sample of zones\n",
    "sample_features = X_test.sample(50).copy()\n",
    "sample_predictions = allocator.predict_demand(sample_features)\n",
    "\n",
    "allocation_df = sample_features.copy()\n",
    "allocation_df['predicted_demand'] = sample_predictions\n",
    "allocation_df['zone_id'] = df.loc[sample_features.index, 'zone_id']\n",
    "\n",
    "# Optimize for 100 available drivers\n",
    "allocation_results = allocator.optimize_allocation(allocation_df, total_drivers=100)\n",
    "\n",
    "display(allocation_results[['zone_id', 'predicted_demand', 'allocated_drivers', 'surge_prob']].head(10))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Visualizations\n",
    "Analyze spatial demand patterns."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate Heatmap using Raw Data\n",
    "raw_df = pd.read_csv('../data/raw_rides.csv')\n",
    "visualize_demand_heatmap(raw_df.sample(2000), output_map='../notebooks/demand_heatmap.html')\n",
    "\n",
    "print(\"Heatmap saved to notebooks/demand_heatmap.html\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

def create_notebook():
    output_path = os.path.join("notebooks", "Ride_Demand_Prediction_Pipeline.ipynb")
    ensure_directory("notebooks")
    
    with open(output_path, 'w') as f:
        json.dump(NOTEBOOK_CONTENT, f, indent=1)
    
    print(f"Notebook created at {output_path}")

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    create_notebook()
