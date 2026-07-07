# diesel.py -- updated 07JUL26

import pandas as pd
import numpy as np
import yaml
from typing import Dict


def load_params(config_path: str = "config/parameters.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_residual_load(
    demand_kwh: pd.Series,
    pv_kwh: pd.Series,
    wind_kwh: pd.Series,
) -> pd.Series:
    residual = demand_kwh - pv_kwh - wind_kwh
    return residual.clip(lower=0).round(3)


def kwh_to_litres(kwh: float, efficiency: float, lhv: float) -> float:
    return kwh / (efficiency * lhv)


def compute_diesel_metrics(
    residual_kwh: pd.Series,
    params: dict,
    site_type: str = "coastal",
) -> dict:
    p = params["diesel"]
    price = (
        p["trindade_price_usd_per_litre"]
        if site_type == "remote_island"
        else p["price_usd_per_litre"]
    )
    total_residual_kwh = residual_kwh.sum()
    litres = kwh_to_litres(total_residual_kwh, p["efficiency"], p["lhv_kwh_per_litre"])
    co2_kg = litres * p["co2_kg_per_litre"]
    cost_usd = litres * price

    return {
        "residual_kwh": round(total_residual_kwh, 1),
        "diesel_litres": round(litres, 1),
        "diesel_cost_usd": round(cost_usd, 2),
        "co2_tonnes": round(co2_kg / 1000, 3),
        "price_usd_per_litre": price,
    }


def baseline_diesel(demand_kwh: pd.Series, params: dict,
                    site_type: str = "coastal") -> dict:
    zero = pd.Series(np.zeros(len(demand_kwh)), index=demand_kwh.index)
    return compute_diesel_metrics(
        compute_residual_load(demand_kwh, zero, zero),
        params, site_type,
    )


def run_diesel_model(
    demand_df: pd.DataFrame,
    site_df: pd.DataFrame,
    params: dict,
    site_type: str = "coastal",
    area_m2: float = None,
    n_turbines: int = None,
) -> dict:
    from src.solar_model import run_solar_model
    from src.wind_model import run_wind_model

    site_df = run_solar_model(site_df, params, area_m2=area_m2)
    site_df = run_wind_model(site_df, params, n_turbines=n_turbines)

    # FIX: demand_df (built from config date_start/date_end) and site_df
    # (real NASA data, may end earlier — confirmed config date_end
    # 2026-06-01 vs actual data through ~2026-03-31) can have DIFFERENT
    # lengths. The original code sliced pv/wind to len(demand) but left
    # demand itself un-truncated to the SHORTER of the two — if site_df
    # was shorter, pv/wind arrays came out shorter than demand, and the
    # subsequent Series arithmetic (different-length RangeIndex Series)
    # silently produced NaN for the non-overlapping tail, which
    # residual_kwh.sum() then silently drops (skipna=True by default).
    # Net effect: the last ~2 months of demand quietly vanished from
    # every diesel-litres/cost/CO2 number in diesel_scenario_table(),
    # with no error or warning. Now explicitly aligned to the shortest
    # of the three before any arithmetic — same pattern already used
    # correctly in compute_diesel_metrics_quantiles() below.
    n = min(len(demand_df), len(site_df))
    if len(demand_df) != len(site_df):
        import logging
        logging.getLogger(__name__).warning(
            "run_diesel_model: demand_df (%d rows) and site_df (%d rows) "
            "have different lengths — truncating both to %d rows. This "
            "usually means config date_start/date_end doesn't match the "
            "real data's actual date range.",
            len(demand_df), len(site_df), n,
        )

    demand = demand_df["demand_kwh"].values[:n]
    pv = site_df["pv_output_kwh"].values[:n]
    wind = site_df["wind_output_kwh"].values[:n]

    demand_s = pd.Series(demand)
    pv_s = pd.Series(pv)
    wind_s = pd.Series(wind)

    residual = compute_residual_load(demand_s, pv_s, wind_s)
    metrics = compute_diesel_metrics(residual, params, site_type)

    base = baseline_diesel(demand_s, params, site_type)
    metrics["baseline_litres"] = base["diesel_litres"]
    metrics["baseline_cost_usd"] = base["diesel_cost_usd"]
    metrics["baseline_co2_tonnes"] = base["baseline_co2_tonnes"] if "baseline_co2_tonnes" in base else base["co2_tonnes"]
    metrics["litres_saved"] = round(base["diesel_litres"] - metrics["diesel_litres"], 1)
    metrics["cost_saved_usd"] = round(base["diesel_cost_usd"] - metrics["diesel_cost_usd"], 2)
    metrics["co2_saved_tonnes"] = round(base["co2_tonnes"] - metrics["co2_tonnes"], 3)
    metrics["renewable_fraction_pct"] = round(
        (1 - residual.sum() / demand_s.sum()) * 100, 2
    )
    return metrics


def diesel_scenario_table(
    all_data: Dict[str, pd.DataFrame],
    demand_df: pd.DataFrame,
    params: dict,
    sites_config: list,
) -> pd.DataFrame:
    site_types = {s["name"]: s["type"] for s in sites_config}
    rows = []
    for area in params["solar"]["scenarios_area_m2"]:
        for n in params["wind"]["scenarios_n_turbines"]:
            for loc, df in all_data.items():
                stype = site_types.get(loc, "coastal")
                m = run_diesel_model(
                    demand_df, df, params,
                    site_type=stype,
                    area_m2=area,
                    n_turbines=n,
                )
                rows.append({
                    "location": loc,
                    "panel_area_m2": area,
                    "n_turbines": n,
                    "renewable_pct": m["renewable_fraction_pct"],
                    "litres_saved": m["litres_saved"],
                    "cost_saved_usd": m["cost_saved_usd"],
                    "co2_saved_t": m["co2_saved_tonnes"],
                    "baseline_litres": m["baseline_litres"],
                })
    return pd.DataFrame(rows).sort_values(
        "renewable_pct", ascending=False
    ).reset_index(drop=True)


if __name__ == "__main__":
    from src.nasa_loader import (load_combined_csv, quality_filter,
                                  gap_fill, add_hub_height_wind, split_by_site)
    from src.load_profile import load_params as lp, build_load_series
    import yaml

    params = load_params()
    df = load_combined_csv(params)
    df = quality_filter(df)
    df = gap_fill(df)
    df = add_hub_height_wind(df, params)
    all_data = split_by_site(df)

    with open("config/sites.yaml") as f:
        sites_config = yaml.safe_load(f)["sites"]

    demand_df = build_load_series(
        params["data"]["date_start"],
        params["data"]["date_end"],
        params,
    )

    print("\n=== Diesel displacement scenario table ===")
    table = diesel_scenario_table(all_data, demand_df, params, sites_config)
    print(table.head(20).to_string(index=False))


# ---------------------------------------------------------------------------
# Quantile extensions
# ---------------------------------------------------------------------------

def compute_diesel_metrics_quantiles(
    pv_fan: pd.DataFrame,
    wind_fan: pd.DataFrame,
    demand_kwh: pd.Series,
    params: dict,
    rho: float = 0.0,
    site_type: str = "coastal",
) -> dict:
    """
    Compute diesel metrics for Q10, Q50, Q90 scenarios, accounting for
    solar-wind correlation rho when combining worst-case quantiles.

    Correlation logic
    -----------------
    Q50 (expected):   pv_Q50 + wind_Q50  — independent medians
    Q90 (best case):  pv_Q90 + wind_Q90  — both resources high
    Q10 (worst case): depends on rho
      rho > 0.3  (correlated):     pv_Q10 + wind_Q10
      rho < -0.3 (anti-correlated): pv_Q10 + wind_Q90 combined via
                                    variance formula, use Q25 wind
      |rho| <= 0.3 (independent):  pv_Q10 + wind_Q25 (conservative)

    Parameters
    ----------
    pv_fan    : DataFrame with columns pv_Q10..pv_Q90 (kWh/day)
    wind_fan  : DataFrame with columns wind_Q10..wind_Q90 (kWh/day)
    demand_kwh: pd.Series of daily demand
    params    : parameters dict
    rho       : solar-wind correlation coefficient [-1, 1]
    site_type : "coastal" or "remote_island"

    Returns
    -------
    dict with keys:
      diesel_litres_Q10, _Q50, _Q90
      diesel_cost_usd_Q10, _Q50, _Q90
      co2_tonnes_Q10, _Q50, _Q90
      cost_saved_usd_Q10, _Q50, _Q90
      co2_saved_tonnes_Q10, _Q50, _Q90
      renewable_pct_Q10, _Q50, _Q90
    """
    # Align lengths — this pattern was already correct here; run_diesel_model()
    # above now follows the same approach.
    n = min(len(demand_kwh), len(pv_fan), len(wind_fan))
    demand_s = pd.Series(demand_kwh.values[:n])

    def _pv(q):
        return pd.Series(pv_fan[f"pv_{q}"].values[:n])

    def _wind(q):
        return pd.Series(wind_fan[f"wind_{q}"].values[:n])

    # Determine wind quantile for Q10 worst case based on rho
    if rho > 0.3:
        wind_q10_partner = "Q10"   # correlated — both bad together
    elif rho < -0.3:
        wind_q10_partner = "Q90"   # anti-correlated — wind good when solar bad
    else:
        wind_q10_partner = "Q25"   # independent — conservative estimate

    scenarios = {
        "Q10": (_pv("Q10"), _wind(wind_q10_partner)),
        "Q50": (_pv("Q50"), _wind("Q50")),
        "Q90": (_pv("Q90"), _wind("Q90")),
    }

    base = baseline_diesel(demand_s, params, site_type)
    results = {}

    for q_lbl, (pv_s, wind_s) in scenarios.items():
        residual = compute_residual_load(demand_s, pv_s, wind_s)
        m = compute_diesel_metrics(residual, params, site_type)
        results[f"diesel_litres_{q_lbl}"] = m["diesel_litres"]
        results[f"diesel_cost_usd_{q_lbl}"] = m["diesel_cost_usd"]
        results[f"co2_tonnes_{q_lbl}"] = m["co2_tonnes"]
        results[f"cost_saved_usd_{q_lbl}"] = round(
            base["diesel_cost_usd"] - m["diesel_cost_usd"], 2
        )
        results[f"co2_saved_tonnes_{q_lbl}"] = round(
            base["co2_tonnes"] - m["co2_tonnes"], 3
        )
        results[f"renewable_pct_{q_lbl}"] = round(
            (1 - residual.sum() / demand_s.sum()) * 100, 2
        )

    results["baseline_litres"] = base["diesel_litres"]
    results["baseline_cost_usd"] = base["diesel_cost_usd"]
    results["baseline_co2_tonnes"] = base["co2_tonnes"]
    return results


def diesel_quantile_scenario_table(
    all_data: Dict[str, pd.DataFrame],
    demand_df: pd.DataFrame,
    params: dict,
    sites_config: list,
    solar_forecasters: dict,
    wind_forecasters: dict,
    correlation_estimator,
) -> pd.DataFrame:
    """
    Full combinatorial sweep producing Q10/Q50/Q90 diesel metrics
    for every site x solar area x n_turbines combination.

    Parameters
    ----------
    solar_forecasters : dict {site: Forecaster(variable="solar")}
    wind_forecasters  : dict {site: Forecaster(variable="wind")}
    correlation_estimator : SiteCorrelationEstimator fitted for all sites

    Returns
    -------
    DataFrame with all scenario rows including Q10/Q50/Q90 columns.
    """
    from src.solar_model import run_solar_model_quantiles
    from src.wind_model import run_wind_model_quantiles
    from src.forecaster import QUANTILE_LABELS

    site_types = {s["name"]: s["type"] for s in sites_config}
    rows = []

    for loc, df in all_data.items():
        stype = site_types.get(loc, "coastal")

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

        sol_fan_hist = pd.DataFrame(sol_q)
        wnd_fan_hist = pd.DataFrame(wnd_q)
        sol_annual = sol_fan_hist.groupby(
            sol_fan_hist.index.dayofyear
        ).mean().iloc[:365]
        wnd_annual = wnd_fan_hist.groupby(
            wnd_fan_hist.index.dayofyear
        ).mean().iloc[:365]

        demand_annual = pd.Series(
            demand_df["demand_kwh"].values[:365]
        )

        rho = correlation_estimator.get(loc, 6)

        for area in params["solar"]["scenarios_area_m2"]:
            for n in params["wind"]["scenarios_n_turbines"]:

                pv_fan = run_solar_model_quantiles(
                    sol_annual, params, area_m2=area
                )
                wind_fan_kwh = run_wind_model_quantiles(
                    wnd_annual, params, n_turbines=n
                )

                m = compute_diesel_metrics_quantiles(
                    pv_fan, wind_fan_kwh, demand_annual,
                    params, rho=rho, site_type=stype,
                )

                rows.append({
                    "location": loc,
                    "panel_area_m2": area,
                    "n_turbines": n,
                    "rho": round(rho, 3),
                    **{k: v for k, v in m.items()},
                })

    return pd.DataFrame(rows).sort_values(
        "renewable_pct_Q50", ascending=False
    ).reset_index(drop=True)
