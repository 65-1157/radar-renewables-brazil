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

    demand = demand_df["demand_kwh"].values
    pv = site_df["pv_output_kwh"].values[:len(demand)]
    wind = site_df["wind_output_kwh"].values[:len(demand)]

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
