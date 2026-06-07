import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from typing import Dict


SITE_COLORS = {
    "Natal": "#1f77b4",
    "Fortaleza": "#ff7f0e",
    "Salvador": "#2ca02c",
    "Cabo Frio": "#d62728",
    "Ilha Grande": "#9467bd",
    "Ilha da Trindade": "#8c564b",
}

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_monthly_solar(all_data: Dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for loc, df in all_data.items():
        monthly = df.groupby("month")["solar_irradiance_kwh_m2_day"].mean()
        ax.plot(monthly.index, monthly.values, marker="o",
                label=loc, color=SITE_COLORS.get(loc))
    ax.set_xlabel("Month")
    ax.set_ylabel("Solar irradiance (kWh/m²/day)")
    ax.set_title("Monthly mean solar irradiance — all sites")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save(fig, "01_monthly_solar.png")


def plot_monthly_wind(all_data: Dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for loc, df in all_data.items():
        monthly = df.groupby("month")["wind_speed_hub_m_s"].mean()
        ax.plot(monthly.index, monthly.values, marker="s",
                label=loc, color=SITE_COLORS.get(loc))
    ax.axhline(y=2.0, color="gray", linestyle="--",
               linewidth=0.8, label="Cut-in (2 m/s)")
    ax.axhline(y=15.0, color="red", linestyle="--",
               linewidth=0.8, label="Cut-out (15 m/s)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Wind speed at hub height (m/s)")
    ax.set_title("Monthly mean wind speed — all sites")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save(fig, "02_monthly_wind.png")


def plot_renewable_fraction(diesel_table: pd.DataFrame) -> None:
    top = diesel_table.sort_values(
        "renewable_pct", ascending=False
    ).head(20).copy()
    top["label"] = (top["location"] + "\n"
                    + top["panel_area_m2"].astype(str) + "m² / "
                    + top["n_turbines"].astype(str) + "T")
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = [SITE_COLORS.get(loc, "#333333") for loc in top["location"]]
    bars = ax.barh(top["label"], top["renewable_pct"], color=colors, alpha=0.85)
    ax.set_xlabel("Renewable energy fraction (%)")
    ax.set_title("Top 20 scenarios — renewable fraction")
    ax.axvline(x=50, color="green", linestyle="--",
               linewidth=0.8, label="50% threshold")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    save(fig, "03_renewable_fraction.png")


def plot_co2_savings(diesel_table: pd.DataFrame) -> None:
    summary = diesel_table.groupby("location")["co2_saved_t"].max().sort_values(
        ascending=False
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [SITE_COLORS.get(loc, "#333333") for loc in summary.index]
    ax.bar(summary.index, summary.values, color=colors, alpha=0.85)
    ax.set_ylabel("Max CO₂ avoided (tonnes/yr)")
    ax.set_title("Maximum CO₂ avoided per site (best scenario)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    save(fig, "04_co2_savings.png")


def plot_npv_heatmap(econ_table: pd.DataFrame) -> None:
    locations = econ_table["location"].unique()
    fig, axes = plt.subplots(1, len(locations),
                             figsize=(4 * len(locations), 5),
                             sharey=False)
    if len(locations) == 1:
        axes = [axes]
    for ax, loc in zip(axes, locations):
        sub = econ_table[econ_table["location"] == loc].copy()
        pivot = sub.pivot_table(
            index="panel_area_m2", columns="n_turbines", values="npv_usd"
        )
        im = ax.imshow(pivot.values / 1e6, aspect="auto", cmap="YlGn")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("N turbines")
        ax.set_ylabel("Panel area m²")
        ax.set_title(loc, fontsize=9)
        plt.colorbar(im, ax=ax, label="NPV $M")
    fig.suptitle("NPV heatmap by site, panel area and turbines", y=1.02)
    save(fig, "05_npv_heatmap.png")


def plot_site_ranking(econ_table: pd.DataFrame) -> None:
    best = econ_table.sort_values("npv_usd", ascending=False).groupby(
        "location"
    ).first().reset_index()
    best = best.sort_values("npv_usd", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [SITE_COLORS.get(loc, "#333333") for loc in best["location"]]
    ax.barh(best["location"], best["npv_usd"] / 1e6, color=colors, alpha=0.85)
    ax.set_xlabel("Best-case NPV (USD millions)")
    ax.set_title("Site ranking — best scenario NPV over 20 years")
    ax.grid(True, axis="x", alpha=0.3)
    save(fig, "06_site_ranking.png")


if __name__ == "__main__":
    from src.nasa_loader import (load_combined_csv, quality_filter,
                                  gap_fill, add_hub_height_wind, split_by_site)
    from src.load_profile import load_params, build_load_series
    from src.diesel_model import diesel_scenario_table
    from src.economics import economics_scenario_table
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
    econ_table = economics_scenario_table(diesel_table, all_data, params)

    print("\nGenerating all plots...")
    plot_monthly_solar(all_data)
    plot_monthly_wind(all_data)
    plot_renewable_fraction(diesel_table)
    plot_co2_savings(diesel_table)
    plot_npv_heatmap(econ_table)
    plot_site_ranking(econ_table)
    print("\nAll plots saved to outputs/")
