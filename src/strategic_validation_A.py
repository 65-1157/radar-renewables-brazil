"""
src/strategic_validation_A.py
=============================
Applies Multi-Criteria Decision Making (AHP-TOPSIS) to select the optimal blueprint,
followed by Monte Carlo stochastic validation for economic resilience.
"""

import numpy as np
import pandas as pd

def apply_ahp_topsis(pareto_df: pd.DataFrame, weight_lcoe: float = 0.3, weight_diesel: float = 0.7) -> pd.Series:
    """
    Selects the single best architecture from the MOPSO Pareto Front based on naval strategic weights.
    Default prioritizes Logistical Fuel Independence (70%) over Capital Cost (30%).
    """
    matrix = pareto_df[['LCOE_USD', 'Diesel_Litres']].values
    
    # 1. Normalize
    norm_matrix = matrix / np.sqrt((matrix**2).sum(axis=0))
    
    # 2. Apply Weights
    weighted_matrix = norm_matrix * [weight_lcoe, weight_diesel]
    
    # 3. Ideal and Anti-Ideal Solutions (Smaller is better for both cost and diesel)
    ideal_best = np.min(weighted_matrix, axis=0)
    ideal_worst = np.max(weighted_matrix, axis=0)
    
    # 4. Geometric Distance Calculation
    dist_to_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
    dist_to_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))
    
    # 5. Calculate Closeness Score
    closeness = dist_to_worst / (dist_to_best + dist_to_worst)
    
    best_index = np.argmax(closeness)
    return pareto_df.iloc[best_index]

def run_monte_carlo(optimal_blueprint: pd.Series, params: dict, iterations: int = 10000) -> float:
    """
    Injects 10,000 randomized 20-year scenarios into the chosen blueprint to verify 
    the probability of surviving severe financial and meteorological deviations.
    """
    print(f"Running Monte Carlo Risk Assessment ({iterations} iterations)...")
    
    base_lcoe = optimal_blueprint['LCOE_USD']
    base_diesel = optimal_blueprint['Diesel_Litres']
    
    # Stochastic Distributions
    sim_diesel_prices = np.random.normal(loc=params["diesel"]["price_usd_per_litre"], scale=0.5, size=iterations)
    sim_solar_yields = np.random.normal(loc=1.0, scale=0.15, size=iterations) # 15% standard deviation in weather
    
    success_count = 0
    budget_cap = base_lcoe * 1.20 # Project fails if LCOE spikes more than 20%
    
    for i in range(iterations):
        price_i = max(0.5, sim_diesel_prices[i]) 
        yield_multiplier = sim_solar_yields[i]
        
        # Recalculate fuel needs if weather underperforms
        adjusted_diesel = base_diesel * (2.0 - yield_multiplier) 
        
        # Simplified simulated LCOE metric
        sim_lcoe = base_lcoe + ((adjusted_diesel * price_i) / 100000.0) 
        
        if sim_lcoe <= budget_cap:
            success_count += 1
            
    probability_success = (success_count / iterations) * 100.0
    return probability_success

def execute_decision_engine(pareto_df: pd.DataFrame, params: dict):
    best_system = apply_ahp_topsis(pareto_df)
    resilience_score = run_monte_carlo(best_system, params)
    
    print("\n" + "="*50)
    print("STRATEGIC VALIDATION COMPLETE")
    print("="*50)
    print(f"Optimal Solar Area: {best_system['Area_m2']} m2")
    print(f"Optimal Turbines  : {best_system['n_Turbines']}")
    print(f"Optimal Battery   : {best_system['Battery_kWh']} kWh")
    print(f"Economic Resilience: {resilience_score:.2f}% probability of budget survival")
    print("="*50)
    
    return best_system, resilience_score