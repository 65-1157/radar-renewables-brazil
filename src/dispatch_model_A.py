"""
src/dispatch_model_A.py
=======================
Ultra-fast, Numba-compiled operational simulation loop.
Implements the predictive dispatch (Strategy 3) and battery degradation logic.

FIX APPLIED: evaluate_predictive_dispatch() previously computed
total_diesel_litres over the FULL simulated period (the real dataset
span passed in from main_pstef_A.py — roughly 6.25 years, 2020-2026),
but then used it directly as "annual_diesel_cost" without dividing by
the number of years actually simulated. This mixed a correctly
annualized CAPEX term (via CRF) with a diesel cost term that was
actually a multi-year total mislabeled as annual — confirmed via
direct reproduction to overstate the diesel cost term by ~6.24x for
the actual data span used in this project. Both total_diesel_litres
and total_lifetime_demand are now correctly divided by the actual
number of simulated years before annualizing.
"""

import numpy as np
import pandas as pd
from numba import njit
from typing import Tuple

DAYS_PER_YEAR = 365.25


@njit
def _fast_numba_dispatch(
    area_m2,
    n_turbines,
    battery_kwh_max,
    pv_arr,
    wind_arr,
    demand_arr,
    lookahead_gen_arr,
    lookahead_dem_arr,
    diesel_eff,
    diesel_lhv
):
    """
    Core simulation loop compiled to C-level machine code via Numba.
    Contains zero Python objects (no Pandas, no classes) for maximum speed.
    """
    battery_current = float(battery_kwh_max)
    total_diesel_litres = 0.0
    battery_cycles = 0.0

    n_days = len(demand_arr) - 3

    for i in range(n_days):
        net_energy = (pv_arr[i] * area_m2) + (wind_arr[i] * n_turbines) - demand_arr[i]

        upcoming_generation = lookahead_gen_arr[i]
        upcoming_demand = lookahead_dem_arr[i]
        emergency_incoming = upcoming_generation < (upcoming_demand * 0.5)

        if emergency_incoming and battery_current < (0.5 * battery_kwh_max):
            deficit = battery_kwh_max - battery_current
            total_diesel_litres += (deficit / (diesel_eff * diesel_lhv))
            battery_current = float(battery_kwh_max)
        elif net_energy > 0.0:
            if battery_current + net_energy > battery_kwh_max:
                battery_current = float(battery_kwh_max)
            else:
                battery_current += net_energy
        else:
            discharge_amount = abs(net_energy)
            if battery_current >= discharge_amount:
                battery_current -= discharge_amount
                battery_cycles += (discharge_amount / battery_kwh_max)
            else:
                shortfall = discharge_amount - battery_current
                battery_current = 0.0
                total_diesel_litres += (shortfall / (diesel_eff * diesel_lhv))
                battery_cycles += 1.0

    return total_diesel_litres, battery_cycles


def evaluate_predictive_dispatch(
    area_m2: float,
    n_turbines: float,
    battery_kwh_max: float,
    pv_arr: np.ndarray,
    wind_arr: np.ndarray,
    demand_arr: np.ndarray,
    lookahead_gen_arr: np.ndarray,
    lookahead_dem_arr: np.ndarray,
    params: dict,
    site_type: str = "coastal",
) -> Tuple[float, float]:
    """
    Python wrapper that extracts parameters from dictionaries and calls the
    Numba-compiled engine. Returns LCOE (USD/kWh, over a 20-year project
    life) and TOTAL Diesel Litres over the simulated period.

    FIX: site_type now determines diesel price (trindade_price_usd_per_litre
    for "remote_island", else the flat price_usd_per_litre) — previously
    this function always used the flat price regardless of site, unlike
    diesel_model.py's compute_diesel_metrics(), which already handled
    this correctly. Without this fix, Ilha da Trindade's MOPSO sizing
    would optimize against an artificially cheap diesel baseline.
    """
    total_diesel_litres, battery_cycles = _fast_numba_dispatch(
        float(area_m2),
        float(n_turbines),
        float(battery_kwh_max),
        pv_arr,
        wind_arr,
        demand_arr,
        lookahead_gen_arr,
        lookahead_dem_arr,
        float(params["diesel"]["efficiency"]),
        float(params["diesel"]["lhv_kwh_per_litre"])
    )

    years_simulated = max(len(demand_arr) / DAYS_PER_YEAR, 1e-6)

    lifespan_cycles = 3000.0
    replacements_needed = np.floor((battery_cycles * 20.0) / lifespan_cycles)

    capex = (area_m2 * params["economics"]["cost_pv_m2"]) + \
            (n_turbines * params["economics"]["cost_wind_turbine"]) + \
            (battery_kwh_max * params["economics"]["cost_battery_kwh"])

    replacement_cost = replacements_needed * (battery_kwh_max * params["economics"]["cost_battery_kwh"])
    crf = (0.08 * (1.0 + 0.08)**20) / ((1.0 + 0.08)**20 - 1.0)

    diesel_price = (
        params["diesel"]["trindade_price_usd_per_litre"]
        if site_type == "remote_island"
        else params["diesel"]["price_usd_per_litre"]
    )
    annual_diesel_litres = total_diesel_litres / years_simulated
    annual_diesel_cost = annual_diesel_litres * diesel_price
    annualized_cost = (capex + replacement_cost) * crf + annual_diesel_cost

    total_lifetime_cost = annualized_cost * 20.0

    annual_demand = np.sum(demand_arr) / years_simulated
    total_lifetime_demand = annual_demand * 20.0

    lcoe = total_lifetime_cost / total_lifetime_demand if total_lifetime_demand > 0.0 else 9999.0

    return lcoe, total_diesel_litres
