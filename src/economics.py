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


# ---------------------------------------------------------------------------
# Quantile extensions
# ---------------------------------------------------------------------------

def economics_quantile_table(
    diesel_q_table: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """
    Compute NPV, payback, and LCOE for Q10, Q50, Q90 scenarios.

    Parameters
    ----------
    diesel_q_table : output of diesel_quantile_scenario_table()
                     must contain cost_saved_usd_Q10/Q50/Q90,
                     baseline_litres, litres_saved_Q10/Q50/Q90
    params         : parameters dict

    Returns
    -------
    DataFrame adding columns:
      npv_usd_Q10, npv_usd_Q50, npv_usd_Q90
      payback_yr_Q10, payback_yr_Q50, payback_yr_Q90
      lcoe_usd_kwh_Q10, lcoe_usd_kwh_Q50, lcoe_usd_kwh_Q90
    """
    battery_kwh = params["simulation"]["battery_capacity_kwh"][1]
    rows = []

    for _, row in diesel_q_table.iterrows():
        area = row["panel_area_m2"]
        n = row["n_turbines"]
        cap = capex_usd(area, n, battery_kwh, params)

        new_row = row.to_dict()

        for q_lbl in ["Q10", "Q50", "Q90"]:
            saving = float(row.get(f"cost_saved_usd_{q_lbl}", 0.0))

            # Annual RE kWh from diesel litres saved
            litres_saved = float(
                row["baseline_litres"]
                - row.get(f"diesel_litres_{q_lbl}", row["baseline_litres"])
            )
            annual_re_kwh = (
                litres_saved
                * params["diesel"]["efficiency"]
                * params["diesel"]["lhv_kwh_per_litre"]
            )

            new_row[f"npv_usd_{q_lbl}"] = npv_usd(saving, cap, params)
            new_row[f"payback_yr_{q_lbl}"] = payback_years(
                saving, cap, params
            )
            new_row[f"lcoe_usd_kwh_{q_lbl}"] = lcoe_usd_per_kwh(
                cap, annual_re_kwh, params
            )

        rows.append(new_row)

    result = pd.DataFrame(rows).sort_values(
        "npv_usd_Q50", ascending=False
    ).reset_index(drop=True)

    return result


def print_quantile_economics_summary(
    econ_q: pd.DataFrame,
    top_n: int = 10,
) -> None:
    """
    Print a formatted summary of the top N scenarios by Q50 NPV,
    showing Q10/Q50/Q90 uncertainty intervals.
    """
    sep = "=" * 75
    print(f"\n{sep}")
    print(f"TOP {top_n} SCENARIOS — NPV UNCERTAINTY (Q10 / Q50 / Q90)")
    print(sep)

    cols = [
        "location", "panel_area_m2", "n_turbines",
        "renewable_pct_Q50",
        "npv_usd_Q10", "npv_usd_Q50", "npv_usd_Q90",
        "payback_yr_Q10", "payback_yr_Q50", "payback_yr_Q90",
    ]
    available = [c for c in cols if c in econ_q.columns]
    # Only show viable scenarios (positive Q50 NPV)
    viable = econ_q[econ_q["npv_usd_Q50"] > 0] if "npv_usd_Q50" in econ_q.columns else econ_q
    if viable.empty:
        print("No scenarios with positive NPV found.")
        print("Showing top 10 by Q50 NPV regardless:")
        print(econ_q[available].head(top_n).to_string(index=False))
    else:
        print(viable[available].head(top_n).to_string(index=False))
