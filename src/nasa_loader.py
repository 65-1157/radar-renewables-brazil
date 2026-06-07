import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List


def load_params(config_path: str = "config/parameters.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_sites(config_path: str = "config/sites.yaml") -> list:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)["sites"]


def load_combined_csv(params: dict) -> pd.DataFrame:
    filepath = Path(params["data"]["raw_path"])
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df.sort_values(["location", "date"]).reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {filepath.name}")
    print(f"Locations: {sorted(df['location'].unique())}")
    print(f"Period: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    if "quality_flag" in df.columns:
        df = df[df["quality_flag"] == "OK"].copy()
    df["solar_irradiance_kwh_m2_day"] = df[
        "solar_irradiance_kwh_m2_day"
    ].clip(lower=0, upper=12)
    df["wind_speed_10m_m_s"] = df["wind_speed_10m_m_s"].clip(lower=0, upper=50)
    df["wind_speed_50m_m_s"] = df["wind_speed_50m_m_s"].clip(lower=0, upper=50)
    after = len(df)
    print(f"Quality filter: {before - after} rows removed, {after} kept")
    return df


def gap_fill(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    filled = 0
    for col in numeric_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            df[col] = df[col].interpolate(method="linear")
            filled += missing
    if filled > 0:
        print(f"Gap-filled {filled} missing values across all columns")
    return df


def add_hub_height_wind(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    hub_height = params["wind"]["hub_height_m"]
    roughness = params["wind"]["roughness_length"]
    if "wind_speed_50m_m_s" in df.columns:
        df["wind_speed_hub_m_s"] = df["wind_speed_50m_m_s"]
        print(f"Hub wind speed: using 50m column directly")
    else:
        log_correction = np.log(hub_height / roughness) / np.log(10.0 / roughness)
        df["wind_speed_hub_m_s"] = (
            df["wind_speed_10m_m_s"] * log_correction
        ).round(3)
        print(f"Hub wind speed: log-corrected from 10m to {hub_height}m")
    return df


def split_by_site(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {loc: grp.reset_index(drop=True)
            for loc, grp in df.groupby("location")}


def summarise_all(all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for loc, df in all_data.items():
        rows.append({
            "location": loc,
            "n_days": len(df),
            "solar_mean_kwh_m2_day": round(
                df["solar_irradiance_kwh_m2_day"].mean(), 3),
            "solar_max": round(df["solar_irradiance_kwh_m2_day"].max(), 3),
            "wind10_mean_ms": round(df["wind_speed_10m_m_s"].mean(), 3),
            "wind50_mean_ms": round(df["wind_speed_50m_m_s"].mean(), 3),
            "wind_hub_mean_ms": round(df["wind_speed_hub_m_s"].mean(), 3),
            "temp_mean_c": round(df["temperature_2m_c"].mean(), 2),
        })
    return pd.DataFrame(rows).sort_values("solar_mean_kwh_m2_day",
                                          ascending=False)


if __name__ == "__main__":
    params = load_params()
    df = load_combined_csv(params)
    df = quality_filter(df)
    df = gap_fill(df)
    df = add_hub_height_wind(df, params)
    all_data = split_by_site(df)

    print("\n=== Site data summary ===")
    summary = summarise_all(all_data)
    print(summary.to_string(index=False))
