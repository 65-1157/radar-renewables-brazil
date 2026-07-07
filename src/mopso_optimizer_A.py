"""
src/mopso_optimizer_A.py
========================
Multi-Objective Particle Swarm Optimization (MOPSO).
Pre-computes forecast lookahead arrays (LSTM or N-BEATS, whichever won
per select_best_model()) for the predictive dispatch strategy.

FIXES applied (verified against installed libraries, not assumed):
  1. `from pymoo.algorithms.moo.mopso import MOPSO` does not exist in
     pymoo 0.6.2 — confirmed directly against the installed package.
     Real available classes: MOPSO_CD (pymoo.algorithms.moo.mopso_cd)
     and CMOPSO (pymoo.algorithms.moo.cmopso). Using MOPSO_CD below as
     the closer match to the original intent (crowding-distance MOPSO).
     VERIFY MOPSO_CD's constructor accepts pop_size before relying on
     this for real results — not yet executed end-to-end.
  2. wind_arr previously computed as (wind_speed * rated_power_kw) —
     dimensionally meaningless. Now uses the real turbine power curve
     from wind_model.turbine_power_kw() (cut-in/cut-out/rated-speed
     cubic curve), matching what dispatch_model_A.py's Numba loop
     actually expects (per-turbine kW at each timestep, multiplied by
     n_turbines inside the dispatch loop).
"""

import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.mopso_cd import MOPSO_CD  # FIX 1: was mopso.MOPSO (doesn't exist)
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from src.dispatch_model_A import evaluate_predictive_dispatch
from src.wind_model import turbine_power_kw  # FIX 2: real power curve


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
    Runs the winning forecast model (LSTM or N-BEATS, per
    Forecaster.set_winner()/load_winners()) once per day to generate
    static 3-day sum arrays for the predictive dispatch check.

    NOTE ON PERFORMANCE: this loop calls forecast_bundle.forecast() once
    per day of history (potentially 2000+ calls for a 6-year dataset).
    If the winning model for this site is N-BEATS, each call is a real
    NeuralForecast.predict() with non-trivial overhead — the same
    per-call cost that was batched away in forecaster.py's
    _get_nbeats_walkforward() for evaluation. This function does NOT yet
    have that optimization; for an N-BEATS-winning site this could be
    slow. Flagging rather than silently leaving it, since batching this
    the same way is a reasonable follow-up if it proves too slow in
    practice.
    """
    n_days = len(demand_series)
    dates = demand_series.index

    lookahead_gen = np.zeros(n_days)
    lookahead_dem = np.zeros(n_days)

    print(f"Pre-computing forecast lookahead arrays for {site}...")
    for i in range(n_days - 3):
        lookahead = forecast_bundle.forecast(site=site, n_days=3, anchor_date=dates[i])
        lookahead_gen[i] = lookahead['solar_Q10'].sum() + lookahead['wind_Q10'].sum()
        lookahead_dem[i] = demand_series.iloc[i+1 : i+4].sum()

    return lookahead_gen, lookahead_dem


def run_swarm_optimization(
    actual_solar, actual_wind, demand, forecast_bundle, site, params
) -> pd.DataFrame:

    # 1. Prepare raw arrays
    pv_arr = (actual_solar * params["solar"]["efficiency"] * params["solar"]["performance_ratio"]).values

    # FIX 2: real turbine power curve, not wind_speed * rated_power_kw.
    # Per-TURBINE kW at each timestep — dispatch_model_A.py multiplies
    # this by n_turbines inside its loop.
    p_wind = params["wind"]
    wind_arr = actual_wind.apply(
        lambda ws: turbine_power_kw(
            ws,
            cut_in=p_wind["cut_in_ms"],
            cut_out=p_wind["cut_out_ms"],
            rated_speed=p_wind["rated_speed_ms"],
            rated_power=p_wind["rated_power_kw"],
        )
    ).values

    demand_arr = demand.values

    # 2. Pre-compute forecast lookaheads (winner-aware, via ForecastBundle)
    lookahead_gen_arr, lookahead_dem_arr = precompute_lstm_lookahead(forecast_bundle, demand, site)

    # 3. Initialize Swarm
    problem = MicrogridSizingProblem(
        pv_arr, wind_arr, demand_arr, lookahead_gen_arr, lookahead_dem_arr, params
    )

    algorithm = MOPSO_CD(pop_size=50)  # FIX 1 applied here
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

    # Was a bare relative path ("outputs/...") — real path now, matching
    # the Drive-backed results directory used throughout this project.
    from pathlib import Path
    output_dir = Path("/content/drive/MyDrive/radar_renewables_results/pareto_fronts")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"pareto_front_{site.replace(' ', '_')}_A.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Pareto front saved -> {output_path}")

    return results_df
