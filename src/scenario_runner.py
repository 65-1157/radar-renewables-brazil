"""
src/scenario_runner.py
======================
Full pipeline sweep entrypoint.

Runs every combination of:
  - 6 sites
  - 5 solar areas  (25, 50, 100, 150, 200 m²)
  - 4 turbine counts (0, 1, 2, 3)

Produces four output CSVs in outputs/:
  scenario_solar.csv
  scenario_wind.csv
  scenario_diesel.csv
  scenario_economics.csv

Returns a dict of DataFrames so main.py can consume them directly.

Usage (standalone)
------------------
  cd ~/Desktop/projects/radar-renewables-brazil
  source .venv/bin/activate
  python -m src.scenario_runner
"""

from __future__ import annotations

import time
import yaml
from pathlib import Path
from typing import Dict

import pandas as pd

from src.nasa_loader import (
    load_params,
    load_combined_csv,
    quality_filter,
    gap_fill,
    add_hub_height_wind,
    split_by_site,
    summarise_all,
)
from src.solar_model import solar_scenario_table
from src.wind_model import wind_scenario_table
from src.diesel_model import diesel_scenario_table
from src.economics import economics_scenario_table
from src.load_profile import build_load_series

OUTPUT_DIR = Path("outputs")


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_sites_config(config_path: str = "config/sites.yaml") -> list:
    with open(config_path) as f:
        return yaml.safe_load(f)["sites"]


def _save(df: pd.DataFrame, filename: str, verbose: bool) -> None:
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    if verbose:
        print(f"    Saved -> {path}")


def _print_summary(
    econ: pd.DataFrame,
    diesel: pd.DataFrame,
    site_summary: pd.DataFrame,
) -> None:
    sep = "=" * 65

    print(f"\n{sep}")
    print("SITE DATA SUMMARY")
    print(sep)
    print(site_summary.to_string(index=False))

    print(f"\n{sep}")
    print("TOP 10 SCENARIOS BY NPV")
    print(sep)
    cols_econ = [
        "location", "panel_area_m2", "n_turbines",
        "renewable_pct", "cost_saved_usd_yr",
        "capex_usd", "npv_usd", "payback_yr", "lcoe_usd_kwh",
    ]
    print(econ[cols_econ].head(10).to_string(index=False))

    print(f"\n{sep}")
    print("TOP 10 SCENARIOS BY RENEWABLE FRACTION")
    print(sep)
    cols_diesel = [
        "location", "panel_area_m2", "n_turbines",
        "renewable_pct", "litres_saved", "cost_saved_usd", "co2_saved_t",
    ]
    top_re = diesel.sort_values("renewable_pct", ascending=False).head(10)
    print(top_re[cols_diesel].to_string(index=False))

    print(f"\n{sep}")
    print("TRINDADE ISLAND — ALL SCENARIOS (sorted by NPV)")
    print(sep)
    trindade = econ[econ["location"] == "Ilha da Trindade"].copy()
    if not trindade.empty:
        print(trindade[cols_econ].to_string(index=False))
    else:
        print("  (no Trindade rows found)")


def run_all(
    config_path: str = "config/parameters.yaml",
    sites_config_path: str = "config/sites.yaml",
    save_csv: bool = True,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Execute the full scenario sweep and return a dict of result DataFrames.

    Keys
    ----
    "solar"        : solar_scenario_table output
    "wind"         : wind_scenario_table output
    "diesel"       : diesel_scenario_table output
    "economics"    : economics_scenario_table output
    "site_summary" : per-site data summary from summarise_all

    Parameters
    ----------
    config_path       : path to parameters.yaml
    sites_config_path : path to sites.yaml
    save_csv          : write CSVs to outputs/ (default True)
    verbose           : print progress to stdout (default True)
    """
    t0 = time.time()
    _ensure_output_dir()

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    # ------------------------------------------------------------------
    # 1. Load and prepare data
    # ------------------------------------------------------------------
    log("\n[1/5] Loading NASA POWER data...")
    params = load_params(config_path)
    sites_config = _load_sites_config(sites_config_path)

    df = load_combined_csv(params)
    df = quality_filter(df)
    df = gap_fill(df)
    df = add_hub_height_wind(df, params)
    all_data = split_by_site(df)

    site_summary = summarise_all(all_data)
    log(f"    {len(all_data)} sites, {len(df)} total rows loaded.")

    # ------------------------------------------------------------------
    # 2. Load demand profile
    # ------------------------------------------------------------------
    log("\n[2/5] Building load profile...")
    demand_df = build_load_series(
        params["data"]["date_start"],
        params["data"]["date_end"],
        params,
    )
    annual_demand_mwh = demand_df["demand_kwh"].sum() / 1000
    log(f"    Annual demand: {annual_demand_mwh:.1f} MWh")

    # ------------------------------------------------------------------
    # 3. Solar scenarios
    # ------------------------------------------------------------------
    n_areas = len(params["solar"]["scenarios_area_m2"])
    n_sites = len(all_data)
    log(f"\n[3/5] Running solar scenarios ({n_areas} areas x {n_sites} sites)...")
    solar_table = solar_scenario_table(all_data, params)
    if save_csv:
        _save(solar_table, "scenario_solar.csv", verbose)

    # ------------------------------------------------------------------
    # 4. Wind scenarios
    # ------------------------------------------------------------------
    n_turb = len(params["wind"]["scenarios_n_turbines"])
    log(f"\n[4/5] Running wind scenarios ({n_turb} turbine counts x {n_sites} sites)...")
    wind_table = wind_scenario_table(all_data, params)
    if save_csv:
        _save(wind_table, "scenario_wind.csv", verbose)

    # ------------------------------------------------------------------
    # 5. Diesel displacement + economics
    # ------------------------------------------------------------------
    n_combos = n_areas * n_turb * n_sites
    log(f"\n[5/5] Running diesel + economics sweep ({n_combos} combinations)...")

    diesel_table = diesel_scenario_table(
        all_data, demand_df, params, sites_config
    )
    econ_table = economics_scenario_table(diesel_table, all_data, params)

    if save_csv:
        _save(diesel_table, "scenario_diesel.csv", verbose)
        _save(econ_table, "scenario_economics.csv", verbose)

    elapsed = time.time() - t0
    log(f"\nDone in {elapsed:.1f}s — results in {OUTPUT_DIR.resolve()}/")

    if verbose:
        _print_summary(econ_table, diesel_table, site_summary)

    return {
        "solar": solar_table,
        "wind": wind_table,
        "diesel": diesel_table,
        "economics": econ_table,
        "site_summary": site_summary,
    }


if __name__ == "__main__":
    run_all()


def run_all_quantiles(
    config_path: str = "config/parameters.yaml",
    sites_config_path: str = "config/sites.yaml",
    save_csv: bool = True,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Full quantile sweep — Q10/Q50/Q90 for diesel and economics.

    Requires empirical forecasters for all sites (no LSTM needed).
    Uses SiteCorrelationEstimator for joint solar-wind worst case.

    Keys in returned dict
    ---------------------
    "diesel_q"   : diesel_quantile_scenario_table output
    "economics_q": economics_quantile_table output
    """
    import time
    import yaml
    from src.nasa_loader import (
        load_params, load_combined_csv, quality_filter,
        gap_fill, add_hub_height_wind, split_by_site,
    )
    from src.load_profile import build_load_series
    from src.forecaster import (
        Forecaster, SiteCorrelationEstimator,
    )
    from src.diesel_model import diesel_quantile_scenario_table
    from src.economics import (
        economics_quantile_table, print_quantile_economics_summary
    )

    t0 = time.time()
    _ensure_output_dir()

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log("\n[1/5] Loading NASA POWER data...")
    params = load_params(config_path)
    sites_config = _load_sites_config(sites_config_path)
    df = load_combined_csv(params)
    df = quality_filter(df)
    df = gap_fill(df)
    df = add_hub_height_wind(df, params)
    all_data = split_by_site(df)
    log(f"    {len(all_data)} sites loaded.")

    log("\n[2/5] Building load profile...")
    demand_df = build_load_series(
        params["data"]["date_start"],
        params["data"]["date_end"],
        params,
    )

    log("\n[3/5] Fitting empirical forecasters...")
    solar_fcast = Forecaster(df, variable="solar")
    solar_fcast.fit_empirical()
    wind_fcast = Forecaster(df, variable="wind")
    wind_fcast.fit_empirical()
    solar_forecasters = {site: solar_fcast for site in all_data}
    wind_forecasters = {site: wind_fcast for site in all_data}

    log("\n[4/5] Estimating solar-wind correlations...")
    corr = SiteCorrelationEstimator()
    for site in all_data:
        sol_resid = solar_fcast._decomposers[site].residuals
        wnd_resid = wind_fcast._decomposers[site].residuals
        corr.fit(sol_resid, wnd_resid, site)
        log(f"    {site}: {corr.correlations[site]}")

    n_combos = (
        len(params["solar"]["scenarios_area_m2"])
        * len(params["wind"]["scenarios_n_turbines"])
        * len(all_data)
    )
    log(f"\n[5/5] Running quantile sweep ({n_combos} combinations)...")

    diesel_q = diesel_quantile_scenario_table(
        all_data, demand_df, params, sites_config,
        solar_forecasters, wind_forecasters, corr,
    )
    econ_q = economics_quantile_table(diesel_q, params)

    if save_csv:
        _save(diesel_q, "scenario_diesel_quantiles.csv", verbose)
        _save(econ_q, "scenario_economics_quantiles.csv", verbose)

    elapsed = time.time() - t0
    log(f"\nDone in {elapsed:.1f}s")

    if verbose:
        print_quantile_economics_summary(econ_q, top_n=10)

    return {"diesel_q": diesel_q, "economics_q": econ_q}
