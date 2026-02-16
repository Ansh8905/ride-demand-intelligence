import pandas as pd
import numpy as np

class DriverAllocator:
    def __init__(self, model=None):
        self.model = model # Not strictly used if logic is separated
        
    def optimize_allocation(self, allocation_df, total_drivers=100):
        """
        Allocates drivers based on predicted demand (highest priority first).
        Calculates:
        - Optimal Supply
        - Shortage/Surplus (Gap)
        - Surge Multiplier
        - Action (Move In/Move Out)
        """
        
        # 1. Calculate Demand Share
        total_demand = allocation_df['predicted_demand'].sum()
        
        if total_demand <= 0:
            allocation_df['allocated_drivers'] = int(total_drivers / len(allocation_df))
            return allocation_df

        # Proportional Allocation
        allocation_df['prob'] = allocation_df['predicted_demand'] / total_demand
        allocation_df['allocated_drivers'] = (allocation_df['prob'] * total_drivers).round().astype(int)
        
        # Adjust remainder
        diff = total_drivers - allocation_df['allocated_drivers'].sum()
        max_idx = allocation_df['predicted_demand'].idxmax()
        allocation_df.loc[max_idx, 'allocated_drivers'] += diff
        
        # 2. Key Metrics
        capacity_per_driver = 3.0 # rides/hr capacity
        allocation_df['optimal_supply'] = np.ceil(allocation_df['predicted_demand'] / capacity_per_driver)
        
        # Gap = Demand - Supply (approximated by drivers * capacity for now, or just drivers)
        # Assuming 1 driver handles 1 currently active ride slot effectively. 
        # But Gap usually means Requests - Drivers.
        allocation_df['gap'] = allocation_df['predicted_demand'] - allocation_df['allocated_drivers']
        
        # 3. Surge Calculation (Rule-Based for Now, uses DSR)
        # Surge = f(Demand / Supply)
        allocation_df['dsr'] = allocation_df['predicted_demand'] / (allocation_df['allocated_drivers'] + 1e-5)
        
        allocation_df['surge_multiplier'] = allocation_df['dsr'].apply(lambda x: min(3.0, max(1.0, 1.0 + 0.5 * np.log(x + 1) if x > 1 else 1.0)))
        
        # 4. Action Recommendation
        allocation_df['action'] = 'Hold'
        
        # Thresholds
        shortage_limit = 5 
        surplus_limit = -3
        
        allocation_df.loc[allocation_df['gap'] > shortage_limit, 'action'] = 'Request Influx'
        allocation_df.loc[allocation_df['gap'] < surplus_limit, 'action'] = 'Move Out'
        
        return allocation_df

    def simulate_revenue(self, allocation_df, avg_fare=15.0):
        """
        Simulates revenue based on allocation efficiency.
        - Unmet demand = Lost Revenue
        - Surge = Increased Revenue (but slightly lower conversion)
        """
        metrics = {}
        
        # Conversion Rate drops as surge increases (Elasticity)
        # e.g. 1.0x -> 100%, 2.0x -> 60%
        allocation_df['conversion_rate'] = 1.0 / (allocation_df['surge_multiplier'] ** 0.8)
        
        # Effective Demand = Demand * Conversion
        allocation_df['effective_demand'] = allocation_df['predicted_demand'] * allocation_df['conversion_rate']
        
        # Fulfilled Rides = min(Effective Demand, Supply * Capacity)
        capacity = 3.0
        allocation_df['fulfilled_rides'] = np.minimum(
            allocation_df['effective_demand'], 
            allocation_df['allocated_drivers'] * capacity
        )
        
        # Revenue = Rides * (BaseFare * Surge)
        allocation_df['revenue'] = allocation_df['fulfilled_rides'] * (avg_fare * allocation_df['surge_multiplier'])
        
        total_revenue = allocation_df['revenue'].sum()
        service_level = allocation_df['fulfilled_rides'].sum() / allocation_df['predicted_demand'].sum()
        
        metrics['Total_Revenue'] = round(total_revenue, 2)
        metrics['Service_Level'] = round(service_level * 100, 1) # %
        metrics['Avg_Surge'] = round(allocation_df['surge_multiplier'].mean(), 2)
        
        return metrics
