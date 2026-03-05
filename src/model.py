import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

class MultiHorizonForecaster:
    def __init__(self):
        self.models = {} # Dictionary of horizons -> model
        
    def prepare_data(self, df):
        """
        Prepares features and targets for training.
        """
        # Features
        features = [
            'hour', 'minute', 'day_of_week', 'is_weekend', 'month',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'lag_1', 'lag_4', 'lag_96', 'rolling_mean_4',
            'zone_lat', 'zone_lon'
        ]
        
        # Add supply features if available
        if 'active_drivers' in df.columns:
            features.append('active_drivers')
        if 'gap' in df.columns:
            features.append('gap')
            
        targets = {
            '15m': 'target_15m',
            '30m': 'target_30m',
            '60m': 'target_60m'
        }
        
        return features, targets

    def train(self, df):
        """
        Trains distinct LightGBM models for 15m, 30m, and 60m horizons.
        """
        print("Training Multi-Horizon Forecast Models...")
        features, target_cols = self.prepare_data(df)
        
        # Sort chronologically
        df = df.sort_values(by=['time_bin'])
        split_idx = int(len(df) * 0.8)
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        metrics = {}
        
        for horizon, target_col in target_cols.items():
            print(f"  Training for {horizon} horizon (Target: {target_col})...")
            
            X_train = train_df[features]
            y_train = train_df[target_col]
            X_test = test_df[features]
            y_test = test_df[target_col]
            
            # LightGBM Regressor
            model = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                objective='regression',
                n_jobs=-1,
                random_state=42,
                verbose=-1
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False),
                           lgb.log_evaluation(period=0)]
            )
            
            self.models[horizon] = model
            
            # Evaluate
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics[horizon] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
            print(f"    {horizon} RMSE: {rmse:.4f}, R2: {r2:.4f}")
            
        return metrics

    def predict(self, features_df):
        """
        Returns predictions for all horizons.
        """
        predictions = {}
        features, _ = self.prepare_data(features_df) # Only gets feature list
        
        # Ensure only feature columns are used
        X = features_df[features]
        
        for horizon, model in self.models.items():
            pred = model.predict(X)
            predictions[horizon] = np.maximum(pred, 0) # Non-negative
            
        return predictions

    def save(self, directory="models"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for horizon, model in self.models.items():
            path = os.path.join(directory, f"lgbm_{horizon}.pkl")
            joblib.dump(model, path)
            print(f"Saved {horizon} model to {path}")

    def load(self, directory="models"):
        horizons = ['15m', '30m', '60m']
        for h in horizons:
            path = os.path.join(directory, f"lgbm_{h}.pkl")
            if os.path.exists(path):
                self.models[h] = joblib.load(path)
                print(f"Loaded {h} model from {path}")

class SurgePricingModel:
    def __init__(self, base_fare=5.0, price_per_km=1.5, price_per_min=0.25):
        self.base_fare = base_fare
        self.price_per_km = price_per_km
        self.price_per_min = price_per_min
        
    def calculate_surge_multiplier(self, demand, supply):
        """
        Calculates surge multiplier based on Demand-Supply Ratio (DSR).
        Sigmoid-like curve to smoothen transitions.
        """
        if supply == 0:
            return 3.0 # Max surge caps
        
        dsr = demand / supply
        
        if dsr <= 1.0:
            return 1.0 # No surge
        
        # Surge formula: 1 + 0.5 * (dsr - 1) capped at 3.0?
        # Or exponential:
        surge = 1.0 + 0.8 * (np.log(dsr + 0.1))
        
        # Cap
        surge = min(max(1.0, surge), 3.0)
        return round(surge, 2)
        
    def predict_price(self, distance_km, duration_min, surge_multiplier):
        base_cost = self.base_fare + (distance_km * self.price_per_km) + (duration_min * self.price_per_min)
        total_price = base_cost * surge_multiplier
        return round(total_price, 2)
