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
    fig.savefig(path, dpi=600, bbox_inches="tight")
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


# ---------------------------------------------------------------------------
# Forecast visualisations — quantile fan charts and evaluation plots
# ---------------------------------------------------------------------------

def plot_forecast_fan(
    fan: pd.DataFrame,
    site: str,
    variable: str,
    actual: pd.Series = None,
) -> None:
    """
    Fan chart for a single site + variable forecast.

    Parameters
    ----------
    fan      : DataFrame with columns Q10, Q25, Q50, Q75, Q90 and DatetimeIndex
    site     : site name (used in title and filename)
    variable : "solar" or "wind"
    actual   : optional pd.Series of observed values over the same period
    """
    color = SITE_COLORS.get(site, "#333333")
    units = "kWh/m²/day" if variable == "solar" else "m/s"
    label = "Solar irradiance" if variable == "solar" else "Wind speed (hub)"

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(fan.index, fan["Q10"], fan["Q90"],
                    alpha=0.15, color=color, label="Q10–Q90")
    ax.fill_between(fan.index, fan["Q25"], fan["Q75"],
                    alpha=0.30, color=color, label="Q25–Q75")
    ax.plot(fan.index, fan["Q50"], color=color,
            linewidth=2, label="Q50 (median)")

    if actual is not None:
        ax.plot(actual.index, actual.values, color="black",
                linewidth=1, linestyle="--", label="Observed")

    ax.set_xlabel("Date")
    ax.set_ylabel(f"{label} ({units})")
    ax.set_title(f"15-day probabilistic forecast — {site} / {variable}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=15, ha="right")

    safe_site = site.replace(" ", "_").replace("/", "_")
    save(fig, f"07_forecast_fan_{safe_site}_{variable}.png")


def plot_forecast_fan_all_sites(
    fans: Dict[str, pd.DataFrame],
    variable: str,
) -> None:
    """
    One subplot per site, all fan charts on a single figure.

    Parameters
    ----------
    fans     : dict mapping site name -> fan DataFrame (Q10..Q90, DatetimeIndex)
    variable : "solar" or "wind"
    """
    sites = list(fans.keys())
    n = len(sites)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    units = "kWh/m²/day" if variable == "solar" else "m/s"

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 4 * nrows),
                             sharey=False)
    axes = np.array(axes).flatten()

    for i, site in enumerate(sites):
        ax = axes[i]
        fan = fans[site]
        color = SITE_COLORS.get(site, "#333333")

        ax.fill_between(fan.index, fan["Q10"], fan["Q90"],
                        alpha=0.15, color=color)
        ax.fill_between(fan.index, fan["Q25"], fan["Q75"],
                        alpha=0.30, color=color)
        ax.plot(fan.index, fan["Q50"], color=color, linewidth=2)

        ax.set_title(site, fontsize=9)
        ax.set_ylabel(units)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", labelrotation=15, labelsize=7)

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"15-day probabilistic forecast — {variable} — all sites", y=1.01
    )
    plt.tight_layout()
    save(fig, f"08_forecast_fan_all_{variable}.png")


def plot_forecast_boxplot(
    fans: Dict[str, pd.DataFrame],
    variable: str,
    day: int = 7,
) -> None:
    """
    Box plot comparing Q10/Q25/Q50/Q75/Q90 across sites for forecast day N.

    Parameters
    ----------
    fans     : dict mapping site name -> fan DataFrame
    variable : "solar" or "wind"
    day      : which forecast day to plot (1-based, default 7 = day ahead)
    """
    units = "kWh/m²/day" if variable == "solar" else "m/s"
    sites = list(fans.keys())

    # Build box stats manually from quantiles
    box_data = []
    for site in sites:
        fan = fans[site]
        if len(fan) < day:
            continue
        row = fan.iloc[day - 1]
        box_data.append({
            "site": site,
            "q10": row["Q10"],
            "q25": row["Q25"],
            "q50": row["Q50"],
            "q75": row["Q75"],
            "q90": row["Q90"],
        })

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, d in enumerate(box_data):
        color = SITE_COLORS.get(d["site"], "#333333")
        # Draw box from Q25 to Q75
        ax.bar(i, d["q75"] - d["q25"], bottom=d["q25"],
               color=color, alpha=0.6, width=0.5)
        # Median line
        ax.plot([i - 0.25, i + 0.25], [d["q50"], d["q50"]],
                color=color, linewidth=2)
        # Whiskers Q10–Q90
        ax.plot([i, i], [d["q10"], d["q25"]], color=color,
                linewidth=1, linestyle="--")
        ax.plot([i, i], [d["q75"], d["q90"]], color=color,
                linewidth=1, linestyle="--")
        # Caps
        ax.plot([i - 0.15, i + 0.15], [d["q10"], d["q10"]],
                color=color, linewidth=1)
        ax.plot([i - 0.15, i + 0.15], [d["q90"], d["q90"]],
                color=color, linewidth=1)

    ax.set_xticks(range(len(box_data)))
    ax.set_xticklabels([d["site"] for d in box_data],
                       rotation=15, ha="right", fontsize=8)
    ax.set_ylabel(f"{units}")
    ax.set_title(
        f"Forecast uncertainty by site — {variable} — day {day} ahead"
    )
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, f"09_forecast_boxplot_{variable}_day{day:02d}.png")


def plot_eval_pinball(eval_df: pd.DataFrame, site: str, variable: str) -> None:
    """
    Bar chart of mean pinball loss per quantile from walk-forward evaluation.

    Parameters
    ----------
    eval_df  : DataFrame from Forecaster.evaluate() with columns
               label, pinball_loss
    site     : site name
    variable : "solar" or "wind"
    """
    mean_loss = eval_df.groupby("label")["pinball_loss"].mean()
    color = SITE_COLORS.get(site, "#333333")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(mean_loss.index, mean_loss.values, color=color, alpha=0.8)
    ax.set_xlabel("Quantile")
    ax.set_ylabel("Mean pinball loss")
    ax.set_title(
        f"Walk-forward evaluation — {site} / {variable}\n"
        f"(lower is better)"
    )
    ax.grid(True, axis="y", alpha=0.3)

    safe_site = site.replace(" ", "_").replace("/", "_")
    save(fig, f"10_eval_pinball_{safe_site}_{variable}.png")


def plot_train_val_loss(
    train_losses: list,
    val_losses: list,
    site: str,
    variable: str,
) -> None:
    """
    Train vs validation pinball loss per epoch.
    The gap between curves diagnoses overfitting.
    """
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, train_losses, label="Train", color="#1f77b4")
    ax.plot(epochs, val_losses, label="Validation", color="#d62728")
    ax.fill_between(
        epochs,
        train_losses,
        val_losses,
        alpha=0.10,
        color="#d62728",
        label="Gap",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean pinball loss")
    ax.set_title(f"Train vs validation loss — {site} / {variable}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    safe_site = site.replace(" ", "_").replace("/", "_")
    save(fig, f"11_train_val_loss_{safe_site}_{variable}.png")


def plot_shap_importance(
    shap_importance: np.ndarray,
    site: str,
    variable: str,
    top_n: int = 20,
) -> None:
    """
    Bar chart of mean absolute SHAP value per input lag.
    Shows which days in the 60-day window drive the forecast most.
    """
    n = len(shap_importance)
    lags = [f"lag_{i+1}d" for i in range(n)]
    idx = np.argsort(shap_importance)[::-1][:top_n]
    top_lags = [lags[i] for i in idx]
    top_vals = shap_importance[idx]

    color = SITE_COLORS.get(site, "#333333")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top_lags[::-1], top_vals[::-1], color=color, alpha=0.85)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(
        f"SHAP feature importance (top {top_n} lags) — {site} / {variable}"
    )
    ax.grid(True, axis="x", alpha=0.3)
    safe_site = site.replace(" ", "_").replace("/", "_")
    save(fig, f"12_shap_importance_{safe_site}_{variable}.png")


def plot_comparison_heatmap(
    comparison_df: pd.DataFrame,
    variable: str,
) -> None:
    """
    Heatmap of mean pinball loss per site per method.
    Rows = sites, columns = methods, cells = mean pinball loss.
    Lower is better — green = good, red = poor.

    Parameters
    ----------
    comparison_df : DataFrame with columns site, quantile, + one col per method
    variable      : "solar" or "wind"
    """
    import matplotlib.colors as mcolors

    methods = [c for c in comparison_df.columns if c not in ("site", "quantile")]
    sites = comparison_df["site"].unique().tolist()

    # Mean across quantiles per site per method
    pivot_rows = []
    for site in sites:
        sub = comparison_df[comparison_df["site"] == site]
        row = {"site": site}
        for m in methods:
            if m in sub.columns:
                row[m] = round(sub[m].mean(), 5)
        pivot_rows.append(row)

    import pandas as pd
    pivot = pd.DataFrame(pivot_rows).set_index("site")[methods]

    fig, ax = plt.subplots(figsize=(4 * len(methods), 6))
    im = ax.imshow(
        pivot.values, aspect="auto", cmap="RdYlGn_r",
        vmin=pivot.values.min(), vmax=pivot.values.max(),
    )

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.upper() for m in methods], fontsize=10)
    ax.set_yticks(range(len(sites)))
    ax.set_yticklabels(sites, fontsize=9)

    # Annotate cells
    for i in range(len(sites)):
        for j in range(len(methods)):
            val = pivot.values[i, j]
            ax.text(
                j, i, f"{val:.4f}",
                ha="center", va="center",
                fontsize=8,
                color="black",
            )

    plt.colorbar(im, ax=ax, label="Mean pinball loss (lower = better)")
    ax.set_title(
        f"Forecast method comparison — {variable} — all sites\n"
        f"(mean pinball loss across Q10/Q25/Q50/Q75/Q90)"
    )
    plt.tight_layout()
    save(fig, f"13_comparison_heatmap_{variable}.png")


def plot_dm_heatmap(
    dm_df: pd.DataFrame,
    variable: str,
) -> None:
    """
    Heatmap of Diebold-Mariano p-values.
    Rows = sites, columns = methods compared against reference.
    Green = significant improvement, red = not significant.
    """
    if dm_df.empty:
        return

    sites = dm_df["site"].unique().tolist()
    methods = dm_df["method_b"].unique().tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, metric, label in zip(
        axes,
        ["p_value", "dm_statistic"],
        ["p-value (green < 0.05 = significant)", "DM statistic (negative = LSTM better)"],
    ):
        pivot_rows = []
        for site in sites:
            row = {"site": site}
            for method in methods:
                sub = dm_df[
                    (dm_df["site"] == site) &
                    (dm_df["method_b"] == method)
                ]
                row[method] = round(float(sub[metric].values[0]), 4) if len(sub) else np.nan
            pivot_rows.append(row)

        pivot = pd.DataFrame(pivot_rows).set_index("site")[methods]

        cmap = "RdYlGn_r" if metric == "p_value" else "RdYlGn"
        im = ax.imshow(pivot.values.astype(float), aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.upper() for m in methods], fontsize=8, rotation=15)
        ax.set_yticks(range(len(sites)))
        ax.set_yticklabels(sites, fontsize=8)
        for i in range(len(sites)):
            for j in range(len(methods)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax)
        ax.set_title(label, fontsize=9)

    fig.suptitle(
        f"Diebold-Mariano test — LSTM vs others — {variable} — Q50",
        y=1.02,
    )
    plt.tight_layout()
    save(fig, f"14_dm_heatmap_{variable}.png")


# ---------------------------------------------------------------------------
# Autonomy visualisations
# ---------------------------------------------------------------------------

def plot_autonomy_bars(
    scenario_table: pd.DataFrame,
    top_n: int = 10,
) -> None:
    """
    Horizontal bar chart of top N scenarios by autonomous days %.
    Saves 15_autonomy_bars.png
    """
    top = scenario_table.head(top_n).copy()
    top["label"] = (
        top["location"] + "\n"
        + top["panel_area_m2"].astype(str) + "m² / "
        + top["n_turbines"].astype(str) + "T"
    )
    colors = [SITE_COLORS.get(loc, "#333333") for loc in top["location"]]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(top["label"], top["autonomous_days_pct"], color=colors, alpha=0.85)
    ax.set_xlabel("Autonomous days (%)")
    ax.set_title(f"Top {top_n} scenarios — autonomous operation without diesel")
    ax.axvline(x=10, color="green", linestyle="--", linewidth=0.8,
               label="10% threshold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    save(fig, "15_autonomy_bars.png")


def plot_autonomy_consecutive(
    scenario_table: pd.DataFrame,
    top_n: int = 10,
) -> None:
    """
    Bar chart of max consecutive autonomous days for top N scenarios.
    Saves 16_autonomy_consecutive.png
    """
    top = scenario_table.head(top_n).copy()
    top["label"] = (
        top["location"] + "\n"
        + top["panel_area_m2"].astype(str) + "m² / "
        + top["n_turbines"].astype(str) + "T"
    )
    colors = [SITE_COLORS.get(loc, "#333333") for loc in top["location"]]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(
        top["label"], top["max_consecutive_days"],
        color=colors, alpha=0.85
    )
    ax.set_xlabel("Maximum consecutive autonomous days")
    ax.set_title(
        f"Top {top_n} scenarios — longest autonomous run (no diesel)"
    )
    for bar, val in zip(bars, top["max_consecutive_days"]):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{int(val)}d", va="center", fontsize=8,
        )
    ax.grid(True, axis="x", alpha=0.3)
    save(fig, "16_autonomy_consecutive.png")


def plot_battery_sizing(battery_table: pd.DataFrame) -> None:
    """
    Grouped bar chart: battery kWh needed for 3/7/15 day autonomy per site.
    Saves 17_battery_sizing.png
    """
    sites = battery_table["location"].tolist()
    x = np.arange(len(sites))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, battery_table["battery_3d_kwh"],
           width, label="3-day autonomy", alpha=0.85, color="#1f77b4")
    ax.bar(x, battery_table["battery_7d_kwh"],
           width, label="7-day autonomy", alpha=0.85, color="#ff7f0e")
    ax.bar(x + width, battery_table["battery_15d_kwh"],
           width, label="15-day autonomy", alpha=0.85, color="#d62728")

    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Battery capacity required (kWh)")
    ax.set_title(
        "Battery sizing for 3 / 7 / 15-day autonomous operation\n"
        "Best scenario per site (200m² + 3 turbines)"
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, "17_battery_sizing.png")


def plot_quantile_autonomy(quantile_table: pd.DataFrame) -> None:
    """
    Grouped bar chart: autonomous days % for Q10/Q50/Q90 per site.
    Shows uncertainty in autonomy estimates.
    Saves 18_quantile_autonomy.png
    """
    sites = quantile_table["location"].unique().tolist()
    x = np.arange(len(sites))
    width = 0.25

    q10 = quantile_table[quantile_table["quantile"] == "Q10"]
    q50 = quantile_table[quantile_table["quantile"] == "Q50"]
    q90 = quantile_table[quantile_table["quantile"] == "Q90"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width,
           [float(q10[q10["location"] == s]["autonomous_days_pct"].values[0])
            if len(q10[q10["location"] == s]) > 0 else 0 for s in sites],
           width, label="Q10 (pessimistic)", alpha=0.85, color="#d62728")
    ax.bar(x,
           [float(q50[q50["location"] == s]["autonomous_days_pct"].values[0])
            if len(q50[q50["location"] == s]) > 0 else 0 for s in sites],
           width, label="Q50 (expected)", alpha=0.85, color="#1f77b4")
    ax.bar(x + width,
           [float(q90[q90["location"] == s]["autonomous_days_pct"].values[0])
            if len(q90[q90["location"] == s]) > 0 else 0 for s in sites],
           width, label="Q90 (optimistic)", alpha=0.85, color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Autonomous days (%)")
    ax.set_title(
        "Autonomous operation uncertainty — Q10 / Q50 / Q90\n"
        "Best scenario per site (200m² + 3 turbines)"
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, "18_quantile_autonomy.png")


# ---------------------------------------------------------------------------
# Dispatch strategy visualisations
# ---------------------------------------------------------------------------

def plot_dispatch_comparison(dispatch_table: pd.DataFrame) -> None:
    """
    Grouped bar chart comparing annual diesel litres per strategy per site.
    Saves 19_dispatch_comparison.png
    """
    sites = dispatch_table["location"].unique().tolist()
    strategies = ["base_load", "load_following", "forecast_driven"]
    strategy_labels = ["Base Load (S2)", "Load Following (S1)",
                       "Forecast-Driven (S3)"]
    strategy_colors = ["#d62728", "#1f77b4", "#2ca02c"]

    x = np.arange(len(sites))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (strat, label, color) in enumerate(
        zip(strategies, strategy_labels, strategy_colors)
    ):
        vals = []
        for site in sites:
            sub = dispatch_table[
                (dispatch_table["location"] == site) &
                (dispatch_table["strategy"] == strat)
            ]
            vals.append(float(sub["annual_diesel_litres"].values[0])
                        if len(sub) > 0 else 0.0)
        offset = (i - 1) * width
        ax.bar(x + offset, vals, width, label=label,
               color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Annual diesel consumption (litres)")
    ax.set_title("Dispatch strategy comparison — annual diesel consumption")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, "19_dispatch_comparison.png")


def plot_dispatch_improvement(improvement_table: pd.DataFrame) -> None:
    """
    Bar chart: Strategy 3 vs Strategy 1 diesel saving % per site.
    Saves 20_dispatch_improvement.png
    """
    imp = improvement_table[
        improvement_table["diesel_saving_pct"] > 0
    ].copy()

    if imp.empty:
        print("  No improvement to plot — all sites show 0% saving.")
        return

    colors = [SITE_COLORS.get(loc, "#333333") for loc in imp["location"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: diesel saving %
    axes[0].barh(imp["location"], imp["diesel_saving_pct"],
                 color=colors, alpha=0.85)
    axes[0].set_xlabel("Diesel saving (%)")
    axes[0].set_title("Forecast-driven vs load-following\ndiesel saving %")
    axes[0].grid(True, axis="x", alpha=0.3)

    # Right: cost saving USD/yr
    axes[1].barh(imp["location"], imp["cost_saving_usd_yr"],
                 color=colors, alpha=0.85)
    axes[1].set_xlabel("Annual cost saving (USD)")
    axes[1].set_title("Forecast-driven vs load-following\nannual cost saving")
    axes[1].grid(True, axis="x", alpha=0.3)

    fig.suptitle(
        "Strategy 3 (forecast-driven) improvement over Strategy 1 (load-following)",
        y=1.02,
    )
    plt.tight_layout()
    save(fig, "20_dispatch_improvement.png")


def plot_dispatch_renewable_fraction(dispatch_table: pd.DataFrame) -> None:
    """
    Stacked view: renewable fraction and autonomous days per strategy per site.
    Saves 21_dispatch_renewable.png
    """
    sites = dispatch_table["location"].unique().tolist()
    strategies = ["load_following", "forecast_driven"]
    strategy_labels = ["Load Following (S1)", "Forecast-Driven (S3)"]
    strategy_colors = ["#1f77b4", "#2ca02c"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, ylabel, title in zip(
        axes,
        ["renewable_fraction_pct", "autonomous_days_pct"],
        ["Renewable fraction (%)", "Autonomous days (%)"],
        ["Renewable energy fraction", "Autonomous operation days"],
    ):
        x = np.arange(len(sites))
        width = 0.35
        for i, (strat, label, color) in enumerate(
            zip(strategies, strategy_labels, strategy_colors)
        ):
            vals = []
            for site in sites:
                sub = dispatch_table[
                    (dispatch_table["location"] == site) &
                    (dispatch_table["strategy"] == strat)
                ]
                vals.append(
                    float(sub[metric].values[0]) if len(sub) > 0 else 0.0
                )
            offset = (i - 0.5) * width
            ax.bar(x + offset, vals, width, label=label,
                   color=color, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(sites, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("S1 vs S3 — renewable utilisation and autonomous operation",
                 y=1.02)
    plt.tight_layout()
    save(fig, "21_dispatch_renewable.png")

"""
Additions for visualiser.py
=============================
Appends MOPSO Pareto Front and Monte Carlo Risk visualizations to the existing 
high-resolution (600 DPI) plotting pipeline.
"""

def plot_pareto_front(pareto_df: pd.DataFrame, site: str, best_system: pd.Series = None) -> None:
    """
    Generates the MOPSO Pareto Front scatter plot.
    This is the core figure required by IEEE optimization reviewers.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot all optimized swarm particles
    ax.scatter(
        pareto_df['Diesel_Litres'], 
        pareto_df['LCOE_USD'], 
        color=SITE_COLORS.get(site, "#1f77b4"), 
        alpha=0.6, 
        edgecolors='w',
        s=60,
        label="Pareto Optimal Solutions"
    )
    
    # Highlight the single system chosen by AHP-TOPSIS
    if best_system is not None:
        ax.scatter(
            best_system['Diesel_Litres'], 
            best_system['LCOE_USD'], 
            color='red', 
            marker='*', 
            s=200, 
            edgecolors='black',
            label="AHP-TOPSIS Strategic Choice"
        )
        
    ax.set_title(f"MOPSO Pareto Front: {site}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Lifecycle Diesel Consumption (Litres)", fontsize=12)
    ax.set_ylabel("Levelized Cost of Energy (USD/kWh)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)
    
    save(fig, f"pareto_front_{site.replace(' ', '_')}.png")

def plot_monte_carlo_risk(sim_lcoe_results: np.ndarray, budget_cap: float, site: str) -> None:
    """
    Generates a histogram of the Monte Carlo simulations.
    Proves to the reviewers that the chosen system survives stochastic uncertainties.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the distribution of the 10,000 simulated project costs
    n, bins, patches = ax.hist(
        sim_lcoe_results, 
        bins=50, 
        color='gray', 
        alpha=0.7, 
        edgecolor='black'
    )
    
    # Color code the histogram based on success (under budget) vs failure (over budget)
    for i in range(len(patches)):
        if bins[i] > budget_cap:
            patches[i].set_facecolor('#d62728') # Red for failure
        else:
            patches[i].set_facecolor('#2ca02c') # Green for success
            
    # Add vertical line for the budget cap
    ax.axvline(budget_cap, color='black', linestyle='dashed', linewidth=2, label=f'Budget Cap (${budget_cap:.4f})')
    
    ax.set_title(f"Monte Carlo Economic Resilience: {site}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Simulated LCOE under Uncertainty (USD/kWh)", fontsize=12)
    ax.set_ylabel("Frequency (10,000 Scenarios)", fontsize=12)
    ax.legend(fontsize=10)
    
    save(fig, f"monte_carlo_risk_{site.replace(' ', '_')}.png")
