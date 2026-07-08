"""
src/main_pstef_A.py
===================
The master entry point for the Predictive-Strategic Techno-Economic Framework (PSTEF).
Orchestrates data loading, LSTM/N-BEATS integration, Swarm Optimization, and Risk Analysis.

REBUILT from the original scaffold: the original file used MockForecastBundle
and np.random.uniform() dummy data throughout (with a missing `import numpy`,
so it could not actually run). This version wires the REAL pipeline:

  nasa_loader.py  -> real NASA POWER solar/wind data
  forecaster.py   -> real trained Forecaster objects (LSTM + N-BEATS),
                     winner-aware via load_winners() / set_winner(), so
                     this pipeline uses whichever model select_best_model()
                     actually chose per site/variable — NOT always LSTM.
  load_profile.py -> real site demand series

NOTE ON load_profile.build_load_series(): confirmed against the real file —
build_load_series(date_start, date_end, params) -> DataFrame with a
"demand_kwh" column, date-indexed. NOT site-specific (generic profile from
params["load"], varying by calendar month/season only).

STEP 9 (added): economics.py's capex_usd()/npv_usd()/payback_years() were
never called anywhere in the original PSTEF path — dispatch_model_A.py
computes its own simpler inline LCOE for the MOPSO objective function
(reasonable, since it runs ~2000 times per site). Once a single final
blueprint is selected, it's now run through the real economics functions
once, against a diesel-only baseline, surfacing NPV and payback_years —
figures the PSTEF path previously never reported at all.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from src.nasa_loader import (
    load_params,
    load_combined_csv,
    quality_filter,
    gap_fill,
    add_hub_height_wind,
)
from src.forecaster import build_forecasters, ForecastBundle, SiteCorrelationEstimator
from src.mopso_optimizer_A import run_swarm_optimization
from src.strategic_validation_A import execute_decision_engine

try:
    from src.load_profile import build_load_series
except ImportError:
    build_load_series = None  # handled explicitly in run_pstef_pipeline()

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path("/content/drive/MyDrive/radar_renewables_results/models")
DEFAULT_WINNERS_PATH = Path("/content/drive/MyDrive/radar_renewables_results/winners.json")

# NOTE: load_profile.py defines its OWN load_params(), independent of
# nasa_loader.load_params() — both default to reading "config/parameters.yaml".
# This works as long as that one file contains all the sections each module
# needs (nasa_loader's solar/wind/site keys AND load_profile's "load" key
# with radar_kw/hvac_summer_kw/etc.) — but it's two independently-maintained
# functions reading the same file by convention, not by a shared contract.
# Worth consolidating into one loader eventually; not fixed here since it
# works today as long as that assumption holds (checked explicitly below).


def load_real_data() -> pd.DataFrame:
    """Same loading sequence used throughout the forecasting notebook (CELL 05)."""
    params = load_params()
    df = load_combined_csv(params)
    df = quality_filter(df)
    df = gap_fill(df)
    df = add_hub_height_wind(df, params)
    return df


def run_pstef_pipeline(
    site_name: str,
    params: dict,
    model_dir: Optional[Path] = None,
    winners_path: Optional[Path] = None,
) -> None:
    print(f"--- Initiating PSTEF Framework for {site_name} ---")
    model_dir = model_dir or DEFAULT_MODEL_DIR
    winners_path = winners_path or DEFAULT_WINNERS_PATH

    # 1. LOAD REAL DATA — replaces the original np.random.uniform() mocks
    df = load_real_data()
    if site_name not in df["location"].unique():
        raise ValueError(
            f"'{site_name}' not found in loaded data. "
            f"Available sites: {sorted(df['location'].unique())}"
        )

    # 2. BUILD REAL FORECASTERS (solar + wind), loading trained model
    #    checkpoints from the same model_dir the CELL 10/11 real run wrote to.
    forecasters = build_forecasters(df, model_dir=model_dir, fit_empirical=True)

    for variable, fcast in forecasters.items():
        # Load LSTM/N-BEATS checkpoints for this site so forecast_lstm()/
        # forecast_nbeats() are actually callable, not just empirical.
        lstm_ckpt = model_dir / f"lstm_{variable}_{site_name.replace(' ', '_')}.pt"
        if lstm_ckpt.exists():
            from src.forecaster import _QuantileLSTMModel
            model = _QuantileLSTMModel()
            model.load(lstm_ckpt)
            fcast._lstm_models[site_name] = model
        else:
            logger.warning("No saved LSTM checkpoint for %s/%s at %s", variable, site_name, lstm_ckpt)

        nbeats_dir = model_dir / f"nbeats_{variable}_{site_name.replace(' ', '_')}"
        if nbeats_dir.exists():
            from src.forecaster import _NBEATSModel
            from neuralforecast import NeuralForecast
            nb_model = _NBEATSModel()
            nb_model._nf = NeuralForecast.load(path=str(nbeats_dir))
            nb_model._site = site_name
            nb_model._is_trained = True
            fcast._nbeats_models[site_name] = nb_model
        else:
            logger.warning("No saved N-BEATS checkpoint for %s/%s at %s", variable, site_name, nbeats_dir)

        # 3. APPLY THE ACTUAL WINNER — this is the critical fix. Without
        #    this, forecast() defaults to LSTM-if-trained regardless of
        #    what select_best_model() determined in CELL 12.
        if winners_path.exists():
            fcast.load_winners(winners_path)
        else:
            logger.warning(
                "winners.json not found at %s — forecast() will fall back to "
                "LSTM-if-trained, else empirical, rather than the actual "
                "comparison-selected model. Run CELL 12 first.", winners_path,
            )

    solar_fc, wind_fc = forecasters["solar"], forecasters["wind"]

    # 3b. Determine site_type from sites.yaml (coastal / coastal_island /
    #     remote_island) — needed so dispatch_model_A.py charges the
    #     correct diesel price (trindade_price_usd_per_litre for
    #     "remote_island", not the flat rate). Previously this lookup
    #     didn't exist anywhere in the PSTEF path at all.
    import yaml
    with open("config/sites.yaml") as f:
        sites_cfg = yaml.safe_load(f)["sites"]
    site_type_map = {s["name"]: s["type"] for s in sites_cfg}
    site_type = site_type_map.get(site_name, "coastal")
    if site_name not in site_type_map:
        logger.warning(
            "site_type not found for '%s' in sites.yaml — defaulting to "
            "'coastal'. If this is Ilha da Trindade, diesel pricing will "
            "be WRONG (flat rate instead of the higher remote-island rate).",
            site_name,
        )
    print(f"Site type for '{site_name}': {site_type}")

    # 4. REAL historical series for MOPSO (actual_solar/actual_wind), not
    #    random noise — these are the raw observed columns for the site.
    site_df = df[df["location"] == site_name].set_index("date")
    actual_solar = site_df["solar_irradiance_kwh_m2_day"]
    actual_wind = site_df["wind_speed_hub_m_s"]

    # 5. REAL DEMAND SERIES. load_profile.build_load_series() is NOT
    #    site-specific — it computes a generic daily demand profile from
    #    fixed load parameters (radar_kw, hvac_summer_kw, etc. under
    #    params["load"]) by calendar month/season, independent of which
    #    station is being evaluated. Confirmed against the actual
    #    load_profile.py signature: build_load_series(date_start, date_end,
    #    params) -> DataFrame with a "demand_kwh" column, date-indexed.
    if build_load_series is None:
        raise ImportError(
            "src.load_profile.build_load_series could not be imported."
        )
    if "load" not in params:
        raise KeyError(
            "params has no 'load' section — build_load_series() requires "
            "params['load'] with radar_kw, processing_kw, comms_kw, "
            "hvac_summer_kw, hvac_winter_kw, lighting_kw, ups_kw, "
            "night_hvac_factor, night_light_factor, day_hours, night_hours, "
            "summer_months, winter_months. If using nasa_loader.load_params(), "
            "confirm config/parameters.yaml actually contains a 'load' section — "
            "this has not been independently verified."
        )
    demand_df = build_load_series(
        date_start=str(site_df.index.min().date()),
        date_end=str(site_df.index.max().date()),
        params=params,
    )
    demand = demand_df["demand_kwh"]

    # 6. REAL SITE CORRELATION (solar-wind), replacing the mock bundle
    solar_resid = solar_fc.get_residuals(site_name)
    wind_resid = wind_fc.get_residuals(site_name)
    corr = SiteCorrelationEstimator().fit(solar_resid, wind_resid, site_name)
    forecast_bundle = ForecastBundle(solar_fc, wind_fc, corr)

    # 7. RUN MOPSO (Layer 1 & Layer 2) — unchanged interface, real inputs now
    pareto_front = run_swarm_optimization(
        actual_solar, actual_wind, demand, forecast_bundle, site_name, params,
        site_type=site_type,
    )

    # 7b. Site-level constants needed by BOTH the Monte Carlo (step 8) and
    #     the real economics (step 9) — computed once here, not duplicated.
    demand_arr_full = demand.values
    years_simulated = max(len(demand_arr_full) / 365.25, 1e-6)
    annual_demand_kwh = np.sum(demand_arr_full) / years_simulated
    diesel_price = (
        params["diesel"]["trindade_price_usd_per_litre"]
        if site_type == "remote_island"
        else params["diesel"]["price_usd_per_litre"]
    )

    # 8. STRATEGIC VALIDATION (Layer 3) — AHP-TOPSIS (real AHP-derived
    #    weight) + Monte Carlo (perturbs solar, wind, AND diesel price;
    #    FIXED dimensional bug — the prior formula guaranteed budget-cap
    #    failure on every iteration regardless of randomness, verified
    #    via direct reproduction).
    final_blueprint, risk_score = execute_decision_engine(
        pareto_front, params, years_simulated, annual_demand_kwh, diesel_price,
    )

    # 9. REAL ECONOMICS — economics.py's capex_usd()/npv_usd()/payback_years()
    #    were never called anywhere in the PSTEF path; dispatch_model_A.py
    #    computes its own simpler inline LCOE for the MOPSO objective
    #    function (reasonable — it runs ~2000 times per site, too
    #    expensive to route through the full economics module each time).
    #    Now that a single final blueprint has been selected, run it
    #    through the REAL economics functions once, using a diesel-only
    #    baseline (same simulated period, same dispatch engine) to get a
    #    genuine annual-saving figure for NPV/payback — surfacing NPV and
    #    payback_years, which the PSTEF path previously never reported at all.
    from src.economics import capex_usd, npv_usd, payback_years, annual_opex_usd
    from src.dispatch_model_A import evaluate_predictive_dispatch

    pv_arr_full = (actual_solar * params["solar"]["efficiency"] * params["solar"]["performance_ratio"]).values
    from src.wind_model import turbine_power_kw
    p_wind = params["wind"]
    wind_arr_full = actual_wind.apply(
        lambda ws: turbine_power_kw(
            ws, cut_in=p_wind["cut_in_ms"], cut_out=p_wind["cut_out_ms"],
            rated_speed=p_wind["rated_speed_ms"], rated_power=p_wind["rated_power_kw"],
        )
    ).values
    zero_lookahead = np.zeros(len(demand_arr_full))

    # Diesel-only baseline: zero PV area, zero turbines, nominal battery
    # (avoids a literal division-by-zero in the dispatch loop's battery
    # cycle accounting; with zero generation the battery never actually
    # gets used, so this has no effect on the baseline diesel figure).
    _, baseline_diesel_litres = evaluate_predictive_dispatch(
        0.0, 0.0, 1.0, pv_arr_full, wind_arr_full, demand_arr_full,
        zero_lookahead, zero_lookahead, params, site_type=site_type,
    )
    baseline_annual_diesel_cost = (
        (baseline_diesel_litres / years_simulated) * diesel_price
    )
    blueprint_annual_diesel_cost = (
        (final_blueprint["Diesel_Litres"] / years_simulated) * diesel_price
    )
    annual_saving = baseline_annual_diesel_cost - blueprint_annual_diesel_cost

    real_capex = capex_usd(
        final_blueprint["Area_m2"], final_blueprint["n_Turbines"],
        final_blueprint["Battery_kWh"], params,
    )
    real_npv = npv_usd(annual_saving, real_capex, params)
    real_payback = payback_years(annual_saving, real_capex, params)

    print("\n" + "=" * 50)
    print("REAL ECONOMICS (economics.py — previously never called by PSTEF)")
    print("=" * 50)
    print(f"CAPEX (economics.py rates): ${real_capex:,.0f}")
    print(f"Baseline (diesel-only) annual cost: ${baseline_annual_diesel_cost:,.0f}")
    print(f"Blueprint annual diesel cost:        ${blueprint_annual_diesel_cost:,.0f}")
    print(f"Annual saving:                       ${annual_saving:,.0f}")
    print(f"NPV ({params['economics']['project_life_years']}yr, "
          f"{params['economics']['discount_rate']*100:.0f}% discount): ${real_npv:,.0f}")
    print(f"Payback period: {real_payback} years")
    print("=" * 50)

    final_blueprint["real_capex_usd"] = real_capex
    final_blueprint["real_npv_usd"] = real_npv
    final_blueprint["real_payback_years"] = real_payback
    final_blueprint["annual_saving_usd"] = annual_saving

    print(f"--- PSTEF complete for {site_name}: risk_score={risk_score} ---")
    return final_blueprint, risk_score


if __name__ == "__main__":
    # Uses the REAL config loader now, instead of a hardcoded mock dict —
    # this also guarantees params has the "load" section build_load_series()
    # requires, as long as config/parameters.yaml actually defines one
    # (see the module-level NOTE above re: load_profile's own load_params()).
    real_params = load_params()
    run_pstef_pipeline("Ilha da Trindade", real_params)
