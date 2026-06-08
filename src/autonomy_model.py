"""
src/autonomy_model.py
=====================
Autonomous operation analysis for radar stations powered by
solar PV + wind turbines + battery storage.

Answers RQ-A: For how long can the station operate on renewables
alone, and what battery capacity is required to extend that autonomy?

Two complementary analyses
--------------------------
1. HISTORICAL AUTONOMY
   Using the 6-year daily record, identifies every sequence of
   consecutive days where renewable output >= demand (no diesel
   needed). Reports distribution of autonomous run lengths per
   site per scenario.

2. BATTERY SIZING
   Given a target autonomy of N days, computes the minimum battery
   capacity (kWh) needed to bridge deficit periods, accounting for
   round-trip efficiency and depth-of-discharge limits.

3. QUANTILE AUTONOMY
   Uses Q10/Q50/Q90 renewable output series to give pessimistic,
   expected, and optimistic autonomy estimates.

Usage
-----
  from src.autonomy_model import run_autonomy_analysis
  results = run_autonomy_analysis(all_data, demand_df, params, sites_config)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

AUTONOMY_TARGETS: List[int] = [3, 7, 15]   # days of target autonomy
BATTERY_DOD: float = 0.80                   # depth of discharge limit
BATTERY_RTE: float = 0.90                   # round-trip efficiency
MIN_SOC: float = 0.20                       # minimum state of charge


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def consecutive_runs(mask: np.ndarray) -> List[int]:
    """
    Return list of lengths of consecutive True runs in a boolean array.

    Example
    -------
    mask = [F, T, T, T, F, F, T, T]
    returns [3, 2]
    """
    runs = []
    count = 0
    for val in mask:
        if val:
            count += 1
        else:
            if count > 0:
                runs.append(count)
            count = 0
    if count > 0:
        runs.append(count)
    return runs


def simulate_battery(
    renewable_kwh: np.ndarray,
    demand_kwh: np.ndarray,
    battery_capacity_kwh: float,
    initial_soc: float = 0.50,
    rte: float = BATTERY_RTE,
    min_soc: float = MIN_SOC,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate battery state of charge over a daily time series.

    Parameters
    ----------
    renewable_kwh      : daily renewable generation array
    demand_kwh         : daily demand array
    battery_capacity_kwh: usable battery capacity in kWh
    initial_soc        : starting state of charge [0, 1]
    rte                : round-trip efficiency
    min_soc            : minimum allowed SOC

    Returns
    -------
    soc          : state of charge array [0, 1]
    load_met     : bool array — True if demand fully met without diesel
    diesel_kwh   : diesel energy needed each day (0 if autonomous)
    """
    n = len(renewable_kwh)
    soc = np.zeros(n + 1)
    soc[0] = initial_soc
    load_met = np.zeros(n, dtype=bool)
    diesel_kwh = np.zeros(n)

    max_usable = battery_capacity_kwh * (1.0 - min_soc)

    for t in range(n):
        surplus = renewable_kwh[t] - demand_kwh[t]

        if surplus >= 0:
            # Renewable exceeds demand — charge battery
            charge = min(surplus * rte, max_usable - soc[t] * battery_capacity_kwh)
            soc[t + 1] = soc[t] + charge / battery_capacity_kwh
            soc[t + 1] = min(soc[t + 1], 1.0)
            load_met[t] = True
            diesel_kwh[t] = 0.0
        else:
            # Deficit — discharge battery
            deficit = abs(surplus)
            available = (soc[t] - min_soc) * battery_capacity_kwh
            discharge = min(deficit, available)
            soc[t + 1] = soc[t] - discharge / battery_capacity_kwh
            soc[t + 1] = max(soc[t + 1], min_soc)
            residual = deficit - discharge
            if residual <= 0.01:
                load_met[t] = True
                diesel_kwh[t] = 0.0
            else:
                load_met[t] = False
                diesel_kwh[t] = round(residual, 3)

    return soc[1:], load_met, diesel_kwh


def battery_kwh_for_autonomy(
    daily_deficit_kwh: np.ndarray,
    target_days: int,
    rte: float = BATTERY_RTE,
    dod: float = BATTERY_DOD,
) -> float:
    """
    Minimum battery capacity needed to cover the worst consecutive
    `target_days` deficit period in the historical record.

    Parameters
    ----------
    daily_deficit_kwh : array of max(0, demand - renewable) per day
    target_days       : desired autonomy in days
    rte               : round-trip efficiency
    dod               : depth of discharge limit

    Returns
    -------
    battery_kwh : minimum battery capacity in kWh
    """
    if len(daily_deficit_kwh) < target_days:
        return float('inf')

    # Rolling sum of worst target_days window
    rolling = pd.Series(daily_deficit_kwh).rolling(target_days).sum()
    worst_deficit = float(rolling.max())

    if worst_deficit <= 0:
        return 0.0

    # Account for round-trip efficiency and depth of discharge
    return round(worst_deficit / (rte * dod), 1)


# ---------------------------------------------------------------------------
# Per-site analysis
# ---------------------------------------------------------------------------

def analyse_site_autonomy(
    site_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    params: dict,
    area_m2: float,
    n_turbines: int,
    battery_kwh: float = 0.0,
) -> dict:
    """
    Compute autonomy metrics for one site × scenario combination.

    Returns dict with:
      autonomous_days_pct   : % of days with full renewable coverage
      max_consecutive_days  : longest autonomous run (days)
      mean_consecutive_days : mean autonomous run length
      median_consecutive_days
      autonomous_runs       : list of all run lengths
      battery_needed_3d     : kWh battery for 3-day autonomy
      battery_needed_7d     : kWh battery for 7-day autonomy
      battery_needed_15d    : kWh battery for 15-day autonomy
      soc_mean              : mean battery SOC if battery_kwh > 0
      diesel_days_with_battery : days diesel needed with given battery
    """
    from src.solar_model import run_solar_model
    from src.wind_model import run_wind_model

    site_df = run_solar_model(site_df, params, area_m2=area_m2)
    site_df = run_wind_model(site_df, params, n_turbines=n_turbines)

    n = min(len(site_df), len(demand_df))
    renewable = (
        site_df['pv_output_kwh'].values[:n]
        + site_df['wind_output_kwh'].values[:n]
    )
    demand = demand_df['demand_kwh'].values[:n]

    # Deficit per day (energy needed from diesel or battery)
    deficit = np.maximum(0, demand - renewable)

    # Autonomous mask (no battery)
    auto_mask = renewable >= demand

    # Consecutive autonomous runs
    runs = consecutive_runs(auto_mask)
    auto_pct = float(auto_mask.sum() / n * 100)

    # Battery sizing for target autonomy
    battery_3d = battery_kwh_for_autonomy(deficit, 3)
    battery_7d = battery_kwh_for_autonomy(deficit, 7)
    battery_15d = battery_kwh_for_autonomy(deficit, 15)

    result = {
        'autonomous_days_pct': round(auto_pct, 2),
        'max_consecutive_days': max(runs) if runs else 0,
        'mean_consecutive_days': round(float(np.mean(runs)), 1) if runs else 0,
        'median_consecutive_days': round(float(np.median(runs)), 1) if runs else 0,
        'n_autonomous_runs': len(runs),
        'battery_needed_3d_kwh': battery_3d,
        'battery_needed_7d_kwh': battery_7d,
        'battery_needed_15d_kwh': battery_15d,
    }

    # Simulate with battery if provided
    if battery_kwh > 0:
        soc, load_met, diesel = simulate_battery(
            renewable, demand, battery_kwh
        )
        result['soc_mean'] = round(float(soc.mean()), 3)
        result['soc_min'] = round(float(soc.min()), 3)
        result['diesel_days_with_battery'] = int((~load_met).sum())
        result['diesel_kwh_with_battery'] = round(float(diesel.sum()), 1)
        result['autonomous_pct_with_battery'] = round(
            float(load_met.sum() / n * 100), 2
        )

    return result


# ---------------------------------------------------------------------------
# Full scenario table
# ---------------------------------------------------------------------------

def autonomy_scenario_table(
    all_data: Dict[str, pd.DataFrame],
    demand_df: pd.DataFrame,
    params: dict,
    sites_config: list,
) -> pd.DataFrame:
    """
    Run autonomy analysis for all site × area × turbine combinations.

    Returns DataFrame with autonomy metrics per scenario.
    """
    site_types = {s['name']: s['type'] for s in sites_config}
    rows = []

    for area in params['solar']['scenarios_area_m2']:
        for n in params['wind']['scenarios_n_turbines']:
            for loc, df in all_data.items():
                metrics = analyse_site_autonomy(
                    df, demand_df, params,
                    area_m2=area,
                    n_turbines=n,
                )
                rows.append({
                    'location': loc,
                    'panel_area_m2': area,
                    'n_turbines': n,
                    'autonomous_days_pct': metrics['autonomous_days_pct'],
                    'max_consecutive_days': metrics['max_consecutive_days'],
                    'mean_consecutive_days': metrics['mean_consecutive_days'],
                    'median_consecutive_days': metrics['median_consecutive_days'],
                    'n_autonomous_runs': metrics['n_autonomous_runs'],
                    'battery_needed_3d_kwh': metrics['battery_needed_3d_kwh'],
                    'battery_needed_7d_kwh': metrics['battery_needed_7d_kwh'],
                    'battery_needed_15d_kwh': metrics['battery_needed_15d_kwh'],
                })

    return pd.DataFrame(rows).sort_values(
        'autonomous_days_pct', ascending=False
    ).reset_index(drop=True)


def battery_sizing_table(
    all_data: Dict[str, pd.DataFrame],
    demand_df: pd.DataFrame,
    params: dict,
    sites_config: list,
    best_scenarios: Optional[Dict[str, Tuple[float, int]]] = None,
) -> pd.DataFrame:
    """
    Battery sizing table for best scenario per site.

    Parameters
    ----------
    best_scenarios : dict {site: (area_m2, n_turbines)} — if None uses
                     200m² + 3 turbines for all sites

    Returns DataFrame showing battery needed for 3/7/15 day autonomy
    per site with and without battery simulation.
    """
    if best_scenarios is None:
        best_scenarios = {
            loc: (200, 3) for loc in all_data
        }

    rows = []
    for loc, df in all_data.items():
        area, n = best_scenarios.get(loc, (200, 3))

        # No battery baseline
        base = analyse_site_autonomy(
            df, demand_df, params, area_m2=area, n_turbines=n
        )

        # Simulate with 7-day battery
        bat_7d = base['battery_needed_7d_kwh']
        with_bat = analyse_site_autonomy(
            df, demand_df, params,
            area_m2=area, n_turbines=n,
            battery_kwh=bat_7d,
        )

        rows.append({
            'location': loc,
            'panel_area_m2': area,
            'n_turbines': n,
            'auto_days_pct_no_battery': base['autonomous_days_pct'],
            'max_consecutive_no_battery': base['max_consecutive_days'],
            'battery_3d_kwh': base['battery_needed_3d_kwh'],
            'battery_7d_kwh': base['battery_needed_7d_kwh'],
            'battery_15d_kwh': base['battery_needed_15d_kwh'],
            'auto_pct_with_7d_battery': with_bat.get(
                'autonomous_pct_with_battery', 'N/A'
            ),
            'diesel_days_with_7d_battery': with_bat.get(
                'diesel_days_with_battery', 'N/A'
            ),
        })

    return pd.DataFrame(rows).sort_values(
        'auto_days_pct_no_battery', ascending=False
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quantile autonomy analysis
# ---------------------------------------------------------------------------

def quantile_autonomy_table(
    all_data: Dict[str, pd.DataFrame],
    demand_df: pd.DataFrame,
    params: dict,
    sites_config: list,
    solar_forecasters: dict,
    wind_forecasters: dict,
) -> pd.DataFrame:
    """
    Autonomy analysis using Q10/Q50/Q90 historical quantile series.

    Gives pessimistic (Q10), expected (Q50), and optimistic (Q90)
    autonomous days estimates per site.

    Uses the same day-of-year quantile reconstruction as
    diesel_quantile_scenario_table in diesel_model.py.
    """
    from src.forecaster import QUANTILE_LABELS
    from src.solar_model import run_solar_model_quantiles
    from src.wind_model import run_wind_model_quantiles

    site_types = {s['name']: s['type'] for s in sites_config}
    rows = []

    for loc, df in all_data.items():
        sol_dc = solar_forecasters[loc]._decomposers[loc]
        wnd_dc = wind_forecasters[loc]._decomposers[loc]
        sol_emp = solar_forecasters[loc]._emp_forecasters[loc]
        wnd_emp = wind_forecasters[loc]._emp_forecasters[loc]

        sol_series = solar_forecasters[loc]._get_site_series(loc)
        dates = sol_series.index
        sol_trend = sol_dc.trend_at(dates)
        sol_seasonal = sol_dc.seasonal_at(dates)
        wnd_trend = wnd_dc.trend_at(dates)
        wnd_seasonal = wnd_dc.seasonal_at(dates)

        sol_q = {}
        wnd_q = {}
        for q, lbl in zip([0.10, 0.25, 0.50, 0.75, 0.90], QUANTILE_LABELS):
            sol_resid_q = np.array([
                sol_emp._doy_quantiles[d][q]
                for d in dates.dayofyear
            ])
            wnd_resid_q = np.array([
                wnd_emp._doy_quantiles[d][q]
                for d in dates.dayofyear
            ])
            sol_q[lbl] = pd.Series(
                np.clip(sol_trend + sol_seasonal + sol_resid_q, 0, None),
                index=dates,
            )
            wnd_q[lbl] = pd.Series(
                np.clip(wnd_trend + wnd_seasonal + wnd_resid_q, 0, None),
                index=dates,
            )

        sol_fan = pd.DataFrame(sol_q)
        wnd_fan = pd.DataFrame(wnd_q)
        sol_annual = sol_fan.groupby(
            sol_fan.index.dayofyear
        ).mean().iloc[:365]
        wnd_annual = wnd_fan.groupby(
            wnd_fan.index.dayofyear
        ).mean().iloc[:365]

        demand_annual = pd.Series(demand_df['demand_kwh'].values[:365])

        for area in [200]:   # best scenario only for quantile table
            for n in [3]:
                pv_fan = run_solar_model_quantiles(
                    sol_annual, params, area_m2=area
                )
                wind_fan_kwh = run_wind_model_quantiles(
                    wnd_annual, params, n_turbines=n
                )

                for q_lbl in ['Q10', 'Q50', 'Q90']:
                    pv_col = f'pv_{q_lbl}'
                    wnd_col = f'wind_{q_lbl}'
                    if pv_col not in pv_fan.columns:
                        continue

                    renewable = (
                        pv_fan[pv_col].values
                        + wind_fan_kwh[wnd_col].values
                    )
                    demand = demand_annual.values
                    deficit = np.maximum(0, demand - renewable)
                    auto_mask = renewable >= demand
                    runs = consecutive_runs(auto_mask)

                    rows.append({
                        'location': loc,
                        'panel_area_m2': area,
                        'n_turbines': n,
                        'quantile': q_lbl,
                        'autonomous_days_pct': round(
                            float(auto_mask.sum() / len(auto_mask) * 100), 2
                        ),
                        'max_consecutive_days': max(runs) if runs else 0,
                        'mean_consecutive_days': round(
                            float(np.mean(runs)), 1
                        ) if runs else 0,
                        'battery_needed_7d_kwh': battery_kwh_for_autonomy(
                            deficit, 7
                        ),
                    })

    return pd.DataFrame(rows).sort_values(
        ['location', 'quantile']
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def find_autonomy_breakeven(
    all_data: Dict[str, pd.DataFrame],
    demand_df: pd.DataFrame,
    params: dict,
    sites_config: list,
) -> pd.DataFrame:
    """
    Find the minimum panel area and turbine count needed to achieve
    meaningful autonomous operation (>= 10% of days autonomous).

    Extends scenarios beyond the standard 5 areas x 4 turbine counts
    to include larger installations up to 500m² and 6 turbines.
    """
    from src.solar_model import run_solar_model
    from src.wind_model import run_wind_model

    extended_areas = [25, 50, 100, 150, 200, 300, 400, 500]
    extended_turbines = [0, 1, 2, 3, 4, 5, 6]

    rows = []
    for area in extended_areas:
        for n in extended_turbines:
            for loc, df in all_data.items():
                site_df = run_solar_model(df, params, area_m2=area)
                site_df = run_wind_model(site_df, params, n_turbines=n)
                nd = min(len(site_df), len(demand_df))
                renewable = (
                    site_df["pv_output_kwh"].values[:nd]
                    + site_df["wind_output_kwh"].values[:nd]
                )
                demand = demand_df["demand_kwh"].values[:nd]
                auto_mask = renewable >= demand
                auto_pct = float(auto_mask.sum() / nd * 100)
                runs = consecutive_runs(auto_mask)
                rows.append({
                    "location": loc,
                    "panel_area_m2": area,
                    "n_turbines": n,
                    "autonomous_days_pct": round(auto_pct, 2),
                    "max_consecutive_days": max(runs) if runs else 0,
                    "mean_renewable_coverage_pct": round(
                        float(np.minimum(renewable, demand).sum()
                              / demand.sum() * 100), 2
                    ),
                })

    df_out = pd.DataFrame(rows)
    # Flag scenarios achieving >= 10% autonomous days
    df_out["viable_autonomy"] = df_out["autonomous_days_pct"] >= 10.0
    return df_out.sort_values(
        ["location", "autonomous_days_pct"], ascending=[True, False]
    ).reset_index(drop=True)


def run_autonomy_analysis(
    all_data: Dict[str, pd.DataFrame],
    demand_df: pd.DataFrame,
    params: dict,
    sites_config: list,
    solar_forecasters: Optional[dict] = None,
    wind_forecasters: Optional[dict] = None,
    save_csv: bool = True,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Run complete autonomy analysis.

    Returns dict with keys:
      'scenario_table'  : full scenario × autonomy metrics
      'battery_sizing'  : battery needed per site for 3/7/15 day autonomy
      'quantile_table'  : Q10/Q50/Q90 autonomy (if forecasters provided)
    """
    from pathlib import Path
    out_dir = Path('outputs')
    out_dir.mkdir(exist_ok=True)

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log('\n[1/3] Running scenario autonomy analysis...')
    scenario_table = autonomy_scenario_table(
        all_data, demand_df, params, sites_config
    )
    if save_csv:
        path = out_dir / 'autonomy_scenarios.csv'
        scenario_table.to_csv(path, index=False)
        log(f'    Saved -> {path}')

    log('\n[2/3] Running battery sizing analysis...')
    battery_table = battery_sizing_table(
        all_data, demand_df, params, sites_config
    )
    if save_csv:
        path = out_dir / 'autonomy_battery_sizing.csv'
        battery_table.to_csv(path, index=False)
        log(f'    Saved -> {path}')

    results = {
        'scenario_table': scenario_table,
        'battery_sizing': battery_table,
    }

    if solar_forecasters and wind_forecasters:
        log('\n[3/3] Running quantile autonomy analysis...')
        q_table = quantile_autonomy_table(
            all_data, demand_df, params, sites_config,
            solar_forecasters, wind_forecasters,
        )
        if save_csv:
            path = out_dir / 'autonomy_quantiles.csv'
            q_table.to_csv(path, index=False)
            log(f'    Saved -> {path}')
        results['quantile_table'] = q_table
    else:
        log('\n[3/3] Skipping quantile analysis (no forecasters provided).')

    if verbose:
        _print_summary(scenario_table, battery_table)

    return results


def _print_summary(
    scenario_table: pd.DataFrame,
    battery_table: pd.DataFrame,
) -> None:
    sep = '=' * 70

    print(f'\n{sep}')
    print('TOP 10 SCENARIOS BY AUTONOMOUS DAYS %')
    print(sep)
    cols = [
        'location', 'panel_area_m2', 'n_turbines',
        'autonomous_days_pct', 'max_consecutive_days',
        'mean_consecutive_days', 'battery_needed_7d_kwh',
    ]
    print(scenario_table[cols].head(10).to_string(index=False))

    print(f'\n{sep}')
    print('BATTERY SIZING — BEST SCENARIO PER SITE (200m² + 3T)')
    print(sep)
    cols_b = [
        'location',
        'auto_days_pct_no_battery',
        'max_consecutive_no_battery',
        'battery_3d_kwh',
        'battery_7d_kwh',
        'battery_15d_kwh',
        'auto_pct_with_7d_battery',
    ]
    print(battery_table[cols_b].to_string(index=False))
