import pandas as pd
import numpy as np
import yaml
from typing import Dict


def load_params(config_path: str = "config/parameters.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def daily_pv_output(
    solar_kwh_m2_day: pd.Series,
    area_m2: float,
    efficiency: float,
    performance_ratio: float,
) -> pd.Series:
    return (solar_kwh_m2_day * area_m2 * efficiency * performance_ratio).round(3)


def run_solar_model(
    df: pd.DataFrame,
    params: dict,
    area_m2: float = None,
) -> pd.DataFrame:
    p = params["solar"]
    area = area_m2 if area_m2 is not None else p["panel_area_m2"]
    df = df.copy()
    df["pv_output_kwh"] = daily_pv_output(
        df["solar_irradiance_kwh_m2_day"],
        area,
        p["efficiency"],
        p["performance_ratio"],
    )
    return df


def solar_summary(df: pd.DataFrame) -> dict:
    return {
        "annual_pv_kwh": round(df["pv_output_kwh"].sum(), 1),
        "annual_pv_mwh": round(df["pv_output_kwh"].sum() / 1000, 3),
        "daily_mean_kwh": round(df["pv_output_kwh"].mean(), 2),
        "daily_max_kwh": round(df["pv_output_kwh"].max(), 2),
        "daily_min_kwh": round(df["pv_output_kwh"].min(), 2),
    }


def run_all_sites_solar(
    all_data: Dict[str, pd.DataFrame],
    params: dict,
    area_m2: float = None,
) -> Dict[str, pd.DataFrame]:
    results = {}
    for loc, df in all_data.items():
        results[loc] = run_solar_model(df, params, area_m2)
    return results


def solar_scenario_table(
    all_data: Dict[str, pd.DataFrame],
    params: dict,
) -> pd.DataFrame:
    rows = []
    for area in params["solar"]["scenarios_area_m2"]:
        for loc, df in all_data.items():
            df_s = run_solar_model(df, params, area_m2=area)
            s = solar_summary(df_s)
            rows.append({
                "location": loc,
                "panel_area_m2": area,
                "annual_pv_mwh": s["annual_pv_mwh"],
                "daily_mean_kwh": s["daily_mean_kwh"],
                "daily_max_kwh": s["daily_max_kwh"],
            })
    return pd.DataFrame(rows).sort_values(
        ["panel_area_m2", "annual_pv_mwh"], ascending=[True, False]
    )


if __name__ == "__main__":
    from src.nasa_loader import load_combined_csv, quality_filter
    from src.nasa_loader import gap_fill, add_hub_height_wind, split_by_site

    params = load_params()
    df = load_combined_csv(params)
    df = quality_filter(df)
    df = gap_fill(df)
    df = add_hub_height_wind(df, params)
    all_data = split_by_site(df)

    print("\n=== Solar PV scenario table ===")
    table = solar_scenario_table(all_data, params)
    print(table.to_string(index=False))


# ---------------------------------------------------------------------------
# Quantile extensions
# ---------------------------------------------------------------------------

def run_solar_model_quantiles(
    fan_df: pd.DataFrame,
    params: dict,
    area_m2: float = None,
) -> pd.DataFrame:
    """
    Apply PV formula to each quantile column of a solar fan DataFrame.

    Parameters
    ----------
    fan_df  : DataFrame with columns Q10, Q25, Q50, Q75, Q90 and DatetimeIndex
              (output of Forecaster.forecast_empirical / forecast_lstm)
    params  : parameters dict
    area_m2 : panel area in m² (overrides params default)

    Returns
    -------
    DataFrame with columns pv_Q10, pv_Q25, pv_Q50, pv_Q75, pv_Q90
    """
    p = params["solar"]
    area = area_m2 if area_m2 is not None else p["panel_area_m2"]
    eff = p["efficiency"]
    pr = p["performance_ratio"]

    result = pd.DataFrame(index=fan_df.index)
    for q_lbl in ["Q10", "Q25", "Q50", "Q75", "Q90"]:
        if q_lbl in fan_df.columns:
            result[f"pv_{q_lbl}"] = (
                fan_df[q_lbl] * area * eff * pr
            ).round(3)
    return result


def solar_quantile_summary(pv_fan: pd.DataFrame) -> dict:
    """
    Annual summary statistics from a PV quantile fan DataFrame.

    Returns dict with annual_pv_mwh_Q10, _Q50, _Q90 and
    daily_mean_kwh_Q10, _Q50, _Q90.
    """
    summary = {}
    for q_lbl in ["Q10", "Q50", "Q90"]:
        col = f"pv_{q_lbl}"
        if col in pv_fan.columns:
            annual = pv_fan[col].sum()
            summary[f"annual_pv_mwh_{q_lbl}"] = round(annual / 1000, 3)
            summary[f"daily_mean_kwh_{q_lbl}"] = round(
                pv_fan[col].mean(), 2
            )
    return summary
