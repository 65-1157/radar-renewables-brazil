import pandas as pd
import numpy as np
import yaml
from typing import Dict


def load_params(config_path: str = "config/parameters.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def turbine_power_kw(
    wind_speed: float,
    cut_in: float,
    cut_out: float,
    rated_speed: float,
    rated_power: float,
) -> float:
    if wind_speed < cut_in or wind_speed >= cut_out:
        return 0.0
    if wind_speed >= rated_speed:
        return rated_power
    return rated_power * ((wind_speed - cut_in) / (rated_speed - cut_in)) ** 3


def daily_wind_output(
    wind_series: pd.Series,
    cut_in: float,
    cut_out: float,
    rated_speed: float,
    rated_power: float,
    n_turbines: int,
    hours_per_day: float = 24.0,
) -> pd.Series:
    power = wind_series.apply(
        lambda ws: turbine_power_kw(ws, cut_in, cut_out, rated_speed, rated_power)
    )
    return (power * n_turbines * hours_per_day).round(3)


def run_wind_model(
    df: pd.DataFrame,
    params: dict,
    n_turbines: int = None,
) -> pd.DataFrame:
    p = params["wind"]
    n = n_turbines if n_turbines is not None else p["n_turbines"]
    df = df.copy()
    df["wind_output_kwh"] = daily_wind_output(
        df["wind_speed_hub_m_s"],
        cut_in=p["cut_in_ms"],
        cut_out=p["cut_out_ms"],
        rated_speed=p["rated_speed_ms"],
        rated_power=p["rated_power_kw"],
        n_turbines=n,
    )
    df["wind_generating"] = (
        (df["wind_speed_hub_m_s"] >= p["cut_in_ms"]) &
        (df["wind_speed_hub_m_s"] < p["cut_out_ms"])
    )
    return df


def wind_summary(df: pd.DataFrame, params: dict) -> dict:
    p = params["wind"]
    total_days = len(df)
    generating_days = df["wind_generating"].sum()
    return {
        "annual_wind_kwh": round(df["wind_output_kwh"].sum(), 1),
        "annual_wind_mwh": round(df["wind_output_kwh"].sum() / 1000, 3),
        "daily_mean_kwh": round(df["wind_output_kwh"].mean(), 2),
        "generating_days": int(generating_days),
        "capacity_factor_pct": round(generating_days / total_days * 100, 1),
        "days_below_cutin": int((df["wind_speed_hub_m_s"] < p["cut_in_ms"]).sum()),
        "days_above_cutout": int((df["wind_speed_hub_m_s"] >= p["cut_out_ms"]).sum()),
    }


def wind_scenario_table(
    all_data: Dict[str, pd.DataFrame],
    params: dict,
) -> pd.DataFrame:
    rows = []
    for n in params["wind"]["scenarios_n_turbines"]:
        for loc, df in all_data.items():
            df_w = run_wind_model(df, params, n_turbines=n)
            s = wind_summary(df_w, params)
            rows.append({
                "location": loc,
                "n_turbines": n,
                "annual_wind_mwh": s["annual_wind_mwh"],
                "daily_mean_kwh": s["daily_mean_kwh"],
                "capacity_factor_pct": s["capacity_factor_pct"],
                "days_below_cutin": s["days_below_cutin"],
            })
    return pd.DataFrame(rows).sort_values(
        ["n_turbines", "annual_wind_mwh"], ascending=[True, False]
    )


if __name__ == "__main__":
    from src.nasa_loader import (load_combined_csv, quality_filter,
                                  gap_fill, add_hub_height_wind, split_by_site)

    params = load_params()
    df = load_combined_csv(params)
    df = quality_filter(df)
    df = gap_fill(df)
    df = add_hub_height_wind(df, params)
    all_data = split_by_site(df)

    print("\n=== Wind scenario table (n_turbines x site) ===")
    table = wind_scenario_table(all_data, params)
    print(table.to_string(index=False))


# ---------------------------------------------------------------------------
# Quantile extensions
# ---------------------------------------------------------------------------

def run_wind_model_quantiles(
    fan_df: pd.DataFrame,
    params: dict,
    n_turbines: int = None,
) -> pd.DataFrame:
    """
    Apply turbine power curve to each quantile column of a wind fan DataFrame.

    The power curve is nonlinear (cubic between cut-in and rated speed),
    so quantiles must be propagated through the actual curve — not scaled
    linearly from Q50.

    Parameters
    ----------
    fan_df     : DataFrame with columns Q10, Q25, Q50, Q75, Q90
                 containing wind speed values in m/s
    params     : parameters dict
    n_turbines : number of turbines (overrides params default)

    Returns
    -------
    DataFrame with columns wind_Q10, wind_Q25, wind_Q50, wind_Q75, wind_Q90
    in kWh/day.
    """
    p = params["wind"]
    n = n_turbines if n_turbines is not None else p["n_turbines"]

    result = pd.DataFrame(index=fan_df.index)
    for q_lbl in ["Q10", "Q25", "Q50", "Q75", "Q90"]:
        if q_lbl in fan_df.columns:
            result[f"wind_{q_lbl}"] = daily_wind_output(
                fan_df[q_lbl],
                cut_in=p["cut_in_ms"],
                cut_out=p["cut_out_ms"],
                rated_speed=p["rated_speed_ms"],
                rated_power=p["rated_power_kw"],
                n_turbines=n,
            )
    return result


def wind_quantile_summary(wind_fan: pd.DataFrame) -> dict:
    """
    Annual summary from a wind quantile fan DataFrame.

    Returns dict with annual_wind_mwh_Q10, _Q50, _Q90 and
    daily_mean_kwh_Q10, _Q50, _Q90.
    """
    summary = {}
    for q_lbl in ["Q10", "Q50", "Q90"]:
        col = f"wind_{q_lbl}"
        if col in wind_fan.columns:
            annual = wind_fan[col].sum()
            summary[f"annual_wind_mwh_{q_lbl}"] = round(annual / 1000, 3)
            summary[f"daily_mean_kwh_{q_lbl}"] = round(
                wind_fan[col].mean(), 2
            )
    return summary
