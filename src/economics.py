import pandas as pd
import numpy as np
import yaml
from typing import Dict


def load_params(config_path: str = "config/parameters.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def capex_usd(area_m2: float, n_turbines: int,
              battery_kwh: float, params: dict) -> float:
    p = params["economics"]
    pw = params["wind"]
    ps = params["solar"]
    solar_kw = area_m2 * ps["efficiency"]
    wind_kw = n_turbines * pw["rated_power_kw"]
    return (
        solar_kw * p["solar_capex_usd_per_kw"]
        + wind_kw * p["wind_capex_usd_per_kw"]
        + battery_kwh * p["battery_capex_usd_per_kwh"]
    )


def annual_opex_usd(capex: float, params: dict) -> float:
    return capex * params["economics"]["opex_pct_capex"]


def npv_usd(annual_saving: float, capex: float,
            params: dict) -> float:
    r = params["economics"]["discount_rate"]
    n = params["economics"]["project_life_years"]
    annuity = annual_saving * (1 - (1 + r) ** -n) / r
    opex_pv = annual_opex_usd(capex, params) * (1 - (1 + r) ** -n) / r
    return round(annuity - capex - opex_pv, 2)


def payback_years(annual_saving: float, capex: float,
                  params: dict) -> float:
    opex = annual_opex_usd(capex, params)
    net_annual = annual_saving - opex
    if net_annual <= 0:
        return float("inf")
    return round(capex / net_annual, 1)


def lcoe_usd_per_kwh(capex: float, annual_kwh: float,
                     params: dict) -> float:
    p = params["economics"]
    r = p["discount_rate"]
    n = p["project_life_years"]
    opex = annual_opex_usd(capex, params)
    cost_pv = capex + opex * (1 - (1 + r) ** -n) / r
    energy_pv = annual_kwh * (1 - (1 + r) ** -n) / r
    if energy_pv <= 0:
        return float("inf")
    return round(cost_pv / energy_pv, 4)


def economics_scenario_table(
    diesel_table: pd.DataFrame,
    all_data: Dict[str, pd.DataFrame],
    params: dict,
) -> pd.DataFrame:
    rows = []
    for _, row in diesel_table.iterrows():
        area = row["panel_area_m2"]
        n = row["n_turbines"]
        loc = row["location"]
        battery_kwh = params["simulation"]["battery_capacity_kwh"][1]

        cap = capex_usd(area, n, battery_kwh, params)
        saving = row["cost_saved_usd"]
        annual_re_kwh = (
            row["baseline_litres"] - (row["baseline_litres"] - row["litres_saved"])
        ) * params["diesel"]["efficiency"] * params["diesel"]["lhv_kwh_per_litre"]

        rows.append({
            "location": loc,
            "panel_area_m2": area,
            "n_turbines": n,
            "renewable_pct": row["renewable_pct"],
            "cost_saved_usd_yr": round(saving, 0),
            "co2_saved_t_yr": row["co2_saved_t"],
            "capex_usd": round(cap, 0),
            "npv_usd": npv_usd(saving, cap, params),
            "payback_yr": payback_years(saving, cap, params),
            "lcoe_usd_kwh": lcoe_usd_per_kwh(cap, annual_re_kwh, params),
        })

    return pd.DataFrame(rows).sort_values(
        "npv_usd", ascending=False
    ).reset_index(drop=True)


if __name__ == "__main__":
    from src.nasa_loader import (load_combined_csv, quality_filter,
                                  gap_fill, add_hub_height_wind, split_by_site)
    from src.load_profile import build_load_series
    from src.diesel_model import diesel_scenario_table
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

    diesel_table = diesel_scenario_table(
        all_data, demand_df, params, sites_config
    )

    print("\n=== Economic evaluation (top 15 scenarios by NPV) ===")
    econ = economics_scenario_table(diesel_table, all_data, params)
    cols = ["location", "panel_area_m2", "n_turbines", "renewable_pct",
            "cost_saved_usd_yr", "capex_usd", "npv_usd", "payback_yr", "lcoe_usd_kwh"]
    print(econ[cols].head(15).to_string(index=False))
