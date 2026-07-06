"""
src/mopso_optimizer_A.py
========================
Multi-Objective Particle Swarm Optimization (MOPSO).
Pre-computes LSTM arrays to allow Numba parallelization.
"""

import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.mopso import MOPSO
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from src.dispatch_model_A import evaluate_predictive_dispatch

class MicrogridSizingProblem(ElementwiseProblem):
    def __init__(
        self, pv_arr, wind_arr, demand_arr, 
        lookahead_gen_arr, lookahead_dem_arr, params
    ):
        super().__init__(n_var=3, n_obj=2, 
                         xl=np.array([10.0, 0.0, 50.0]),     
                         xu=np.array([1000.0, 10.0, 2000.0])) 
        
        self.pv_arr = pv_arr
        self.wind_arr = wind_arr
        self.demand_arr = demand_arr
        self.lookahead_gen_arr = lookahead_gen_arr
        self.lookahead_dem_arr = lookahead_dem_arr
        self.params = params

    def _evaluate(self, x, out, *args, **kwargs):
        area_m2, n_turbines, battery_kwh = x[0], np.round(x[1]), x[2]
        
        lcoe, diesel = evaluate_predictive_dispatch(
            area_m2, n_turbines, battery_kwh,
            self.pv_arr, self.wind_arr, self.demand_arr,
            self.lookahead_gen_arr, self.lookahead_dem_arr,
            self.params
        )
        out["F"] = [lcoe, diesel]

def precompute_lstm_lookahead(forecast_bundle, demand_series, site):
    """
    Runs the LSTM once to generate static 3-day sum arrays. 
    This enables the Numba compiler in Layer 1 to operate without Python objects.
    """
    n_days = len(demand_series)
    dates = demand_series.index
    
    lookahead_gen = np.zeros(n_days)
    lookahead_dem = np.zeros(n_days)
    
    print(f"Pre-computing LSTM lookahead arrays for {site}...")
    for i in range(n_days - 3):
        lookahead = forecast_bundle.forecast(site=site, n_days=3, anchor_date=dates[i])
        lookahead_gen[i] = lookahead['solar_Q10'].sum() + lookahead['wind_Q10'].sum()
        lookahead_dem[i] = demand_series.iloc[i+1 : i+4].sum()
        
    return lookahead_gen, lookahead_dem

def run_swarm_optimization(
    actual_solar, actual_wind, demand, forecast_bundle, site, params
) -> pd.DataFrame:
    
    # 1. Prepare raw arrays for Numba
    pv_arr = (actual_solar * params["solar"]["efficiency"] * params["solar"]["performance_ratio"]).values
    wind_arr = (actual_wind * params["wind"]["rated_power_kw"]).values
    demand_arr = demand.values
    
    # 2. Pre-compute Neural Network lookaheads
    lookahead_gen_arr, lookahead_dem_arr = precompute_lstm_lookahead(forecast_bundle, demand, site)
    
    # 3. Initialize Swarm
    problem = MicrogridSizingProblem(
        pv_arr, wind_arr, demand_arr, lookahead_gen_arr, lookahead_dem_arr, params
    )
    
    algorithm = MOPSO(pop_size=50)
    termination = get_termination("n_gen", 40) 
    
    print(f"Executing MOPSO Optimization for {site}...")
    res = minimize(problem, algorithm, termination, seed=42, verbose=True)
    
    results_df = pd.DataFrame({
        "Area_m2": np.round(res.X[:, 0], 2),
        "n_Turbines": np.round(res.X[:, 1], 0),
        "Battery_kWh": np.round(res.X[:, 2], 2),
        "LCOE_USD": np.round(res.F[:, 0], 4),
        "Diesel_Litres": np.round(res.F[:, 1], 2)
    })
    
    output_path = f"outputs/pareto_front_{site}_A.csv"
    results_df.to_csv(output_path, index=False)
    return results_df