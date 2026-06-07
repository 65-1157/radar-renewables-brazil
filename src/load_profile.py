import pandas as pd
import numpy as np
import yaml
from pathlib import Path


def load_params(config_path: str = "config/parameters.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_season(month: int, params: dict) -> str:
    if month in params["load"]["summer_months"]:
        return "summer"
    elif month in params["load"]["winter_months"]:
        return "winter"
    else:
        return "shoulder"


def hvac_for_season(season: str, params: dict) -> float:
    p = params["load"]
    if season == "summer":
        return p["hvac_summer_kw"]
    elif season == "winter":
        return p["hvac_winter_kw"]
    else:
        return (p["hvac_summer_kw"] + p["hvac_winter_kw"]) / 2.0


def daily_demand_kwh(month: int, params: dict) -> float:
    p = params["load"]
    season = get_season(month, params)
    hvac = hvac_for_season(season, params)

    base_kw = p["radar_kw"] + p["processing_kw"] + p["comms_kw"]

    day_kw = base_kw + hvac + p["lighting_kw"] + p["ups_kw"]
    night_kw = (
        base_kw
        + hvac * p["night_hvac_factor"]
        + p["lighting_kw"] * p["night_light_factor"]
        + p["ups_kw"]
    )

    return day_kw * p["day_hours"] + night_kw * p["night_hours"]


def build_load_series(
    date_start: str, date_end: str, params: dict
) -> pd.DataFrame:
    dates = pd.date_range(start=date_start, end=date_end, freq="D")
    records = []
    for d in dates:
        season = get_season(d.month, params)
        demand = daily_demand_kwh(d.month, params)
        records.append(
            {
                "date": d,
                "month": d.month,
                "season": season,
                "demand_kwh": round(demand, 2),
            }
        )
    return pd.DataFrame(records).set_index("date")


def summarise_load(df: pd.DataFrame) -> dict:
    annual_mwh = df["demand_kwh"].sum() / 1000
    summer = df[df["season"] == "summer"]["demand_kwh"].mean()
    winter = df[df["season"] == "winter"]["demand_kwh"].mean()
    shoulder = df[df["season"] == "shoulder"]["demand_kwh"].mean()
    return {
        "annual_mwh": round(annual_mwh, 2),
        "summer_avg_kwh_day": round(summer, 2),
        "winter_avg_kwh_day": round(winter, 2),
        "shoulder_avg_kwh_day": round(shoulder, 2),
    }


if __name__ == "__main__":
    params = load_params()
    df = build_load_series(
        params["data"]["date_start"],
        params["data"]["date_end"],
        params,
    )
    summary = summarise_load(df)
    print("\n=== Load profile summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nFirst 5 rows:\n{df.head()}")
