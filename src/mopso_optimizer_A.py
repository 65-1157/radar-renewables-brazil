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
        lookahead_gen_arr, lookahead_dem_arr, params,
        site_type: str = "coastal",
    ):
        # Area_m2 upper bound widened 1000.0 -> 3000.0: verified on the
        # Salvador Pareto front that EVERY top candidate (by both LCOE
        # and diesel litres) was pinned exactly to the old 1000.0 bound,
        # with no exceptions — a specific signal the true unconstrained
        # optimum lies above it, not that 1000 m2 was itself optimal.
        # NOTE: 3000 m2 is a reasoned default, not a physically-confirmed
        # site constraint (e.g. actual available roof/ground area at each
        # station) — if MOPSO pins to this new bound again, it should be
        # widened further; if real site footprint limits exist, they
        # should replace this value before final paper numbers.
        #
        # Battery_kWh lower bound narrowed 50.0 -> 5.0: after the Area_m2
        # widening (which drove diesel usage down ~98.6%, to ~401 L),
        # re-checking the Battery_kWh distribution across the Salvador
        # front showed 192/197 points pinned exactly to the old 50.0
        # floor — the same boundary-pinning signature as Area_m2 had,
        # just at the opposite (lower) bound: with diesel already nearly
        # eliminated by solar, a large battery has little further value,
        # and the optimizer wanted to go below 50.0 kWh but couldn't.
        # 5.0 (not 0.0) keeps a small positive floor to avoid a literal
        # division-by-zero in _fast_numba_dispatch's battery-cycle
        # accounting (discharge_amount / battery_kwh_max).
        #
        # n_Turbines is left unchanged: showed genuine internal variation
        # across the original front (0-10), not pinned to either edge —
        # no evidence it needs widening. (Its later collapse to 0 across
        # the board, after the Area widening, is a separate, explained
        # effect — see mopso_optimizer_A.py's usage notes — not a bound
        # artifact.)
        #
        # Area_m2 upper bound widened AGAIN, 3000.0 -> 5000.0: after the
        # first widening, 5/6 sites settled at genuine interior values
        # (2085-2837 m2), but Ilha Grande's SELECTED blueprint remained
        # pinned exactly to the 3000.0 bound. Confirmed with the project
        # owner that this is an exploratory study with no real physical
        # area constraint at any site — so, per the same principle
        # applied to the first widening, a pinned bound means the search
        # space is still too tight, not that 3000 m2 was the answer.
        super().__init__(n_var=3, n_obj=2,
                         xl=np.array([10.0, 0.0, 5.0]),
                         xu=np.array([5000.0, 10.0, 2000.0]))

        self.pv_arr = pv_arr
        self.wind_arr = wind_arr
        self.demand_arr = demand_arr
        self.lookahead_gen_arr = lookahead_gen_arr
        self.lookahead_dem_arr = lookahead_dem_arr
        self.params = params
        self.site_type = site_type

    def _evaluate(self, x, out, *args, **kwargs):
        area_m2, n_turbines, battery_kwh = x[0], np.round(x[1]), x[2]

        lcoe, diesel = evaluate_predictive_dispatch(
            area_m2, n_turbines, battery_kwh,
            self.pv_arr, self.wind_arr, self.demand_arr,
            self.lookahead_gen_arr, self.lookahead_dem_arr,
            self.params, site_type=self.site_type,
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
    _get_nbeats_walkforward() for evaluation. Not batched here yet;
    flagged as a reasonable follow-up if it proves too slow in practice.

    MEMORY: periodic gc.collect()/torch.cuda.empty_cache() added every
    200 iterations — the same class of fix applied earlier to the
    Optuna tuning loops (tune_lstm/tune_nbeats), which this loop had
    been missing despite calling the same underlying forecast methods
    thousands of times across a full CELL 16 run (7 sites x ~2000+ days).
    """
    import time
    import gc
    import torch

    n_days = len(demand_series)
    dates = demand_series.index

    lookahead_gen = np.zeros(n_days)
    lookahead_dem = np.zeros(n_days)

    print(f"Pre-computing forecast lookahead arrays for {site} ({n_days - 3} days)...")
    t0 = time.time()
    CLEANUP_EVERY = 200
    PROGRESS_EVERY = 500

    for i in range(n_days - 3):
        lookahead = forecast_bundle.forecast(site=site, n_days=3, anchor_date=dates[i])
        lookahead_gen[i] = lookahead['solar_Q10'].sum() + lookahead['wind_Q10'].sum()
        lookahead_dem[i] = demand_series.iloc[i+1 : i+4].sum()

        if (i + 1) % CLEANUP_EVERY == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if (i + 1) % PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            eta = (elapsed / (i + 1)) * (n_days - 3 - (i + 1))
            print(f"  lookahead precompute [{i+1}/{n_days-3}] "
                  f"elapsed {elapsed:.0f}s, ETA ~{eta:.0f}s")

    return lookahead_gen, lookahead_dem


def run_swarm_optimization(
    actual_solar, actual_wind, demand, forecast_bundle, site, params,
    site_type: str = "coastal",
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
        pv_arr, wind_arr, demand_arr, lookahead_gen_arr, lookahead_dem_arr,
        params, site_type=site_type,
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

    # DEDUPLICATE: MOPSO_CD's archive (default archive_size=200) commonly
    # accumulates exact-duplicate points when multiple particles across
    # generations independently converge to the same strong optimum —
    # confirmed directly: 32/200 rows were exact duplicates for Salvador.
    # These aren't distinct trade-off solutions, just re-discoveries of
    # the same point, and would misleadingly inflate "N non-dominated
    # solutions found" if reported as-is in the paper.
    n_before = len(results_df)
    results_df = results_df.drop_duplicates().reset_index(drop=True)
    n_after = len(results_df)
    if n_after < n_before:
        print(f"Pareto front: removed {n_before - n_after} duplicate point(s) "
              f"({n_after} genuinely distinct solutions remain)")

    # Was a bare relative path ("outputs/...") — real path now, matching
    # the Drive-backed results directory used throughout this project.
    from pathlib import Path
    output_dir = Path("/content/drive/MyDrive/radar_renewables_results/pareto_fronts")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"pareto_front_{site.replace(' ', '_')}_A.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Pareto front saved -> {output_path}")

    return results_df
