"""
visualiser_fixes.py
====================
Corrections and additions to visualiser.py, matching the REAL, verified
column names produced by this project's actual pipeline (not the legacy
scenario-sweep pipeline the original 21-function file mostly targeted).

Import alongside the original visualiser.py and use these versions
instead of the two originals they replace:
    from src.visualiser import SITE_COLORS, save
    from src.visualiser_fixes import (
        plot_comparison_heatmap, plot_dm_heatmap, plot_final_ranking
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from src.visualiser import SITE_COLORS, save
except ImportError:
    SITE_COLORS = {}
    OUTPUT_DIR = Path("/content/drive/MyDrive/radar_renewables_results/figures")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        path = OUTPUT_DIR / name
        fig.savefig(path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


def plot_comparison_heatmap(comparison_df: pd.DataFrame, variable: str) -> None:
    """
    FIXED: original expected a wide-format DataFrame (site, quantile, one
    column per method). Real model_comparison.csv is long-format:
    columns variable, site, method, mape, rmse, error_std.

    Heatmap of MAPE per site per method (lower = better).
    """
    sub = comparison_df[comparison_df["variable"] == variable]
    pivot = sub.pivot_table(index="site", columns="method", values="mape")

    fig, ax = plt.subplots(figsize=(4 * len(pivot.columns), 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([m.upper() for m in pivot.columns], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.2f}%", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, label="MAPE (%) — lower is better")
    ax.set_title(f"Forecast method comparison — {variable} — all sites")
    plt.tight_layout()
    save(fig, f"13_comparison_heatmap_{variable}.png")


def plot_dm_heatmap(dm_df: pd.DataFrame, variable: str) -> None:
    """
    FIXED: original assumed a single reference method ("LSTM vs others").
    Real diebold_mariano.csv compares against whichever method actually
    won each site (overall_winner column, per select_best_model()) — not
    always LSTM. Title updated to reflect this rather than hardcode LSTM.
    """
    sub = dm_df[dm_df["variable"] == variable]
    if sub.empty:
        print(f"  No DM data for variable={variable}, skipping.")
        return

    sites = sub["site"].unique().tolist()
    methods = sub["method_b"].unique().tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, metric, label in zip(
        axes, ["p_value", "dm_statistic"],
        ["p-value (green < 0.05 = significant)", "DM statistic"],
    ):
        pivot = sub.pivot_table(index="site", columns="method_b", values=metric)
        pivot = pivot.reindex(index=sites, columns=methods)
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
        f"Diebold-Mariano test vs. selected winner — {variable} — Q50", y=1.02
    )
    plt.tight_layout()
    save(fig, f"14_dm_heatmap_{variable}.png")


def plot_final_ranking(blueprints_df: pd.DataFrame) -> None:
    """
    NEW: no equivalent existed anywhere in the original file — the
    original plot_site_ranking() depended on the never-run legacy
    economics_scenario_table(). This uses the REAL, verified
    final_blueprints_all_sites.csv (CELL 16's output).

    Two-panel figure: NPV ranking (left) and LCOE vs. diesel litres
    (right), across all sites, colour-coded by site_type.
    """
    df = blueprints_df.sort_values("real_npv_usd", ascending=True)
    type_colors = {"coastal": "#1f77b4", "coastal_island": "#9467bd", "remote_island": "#8c564b"}
    colors = [type_colors.get(t, "#333333") for t in df["site_type"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].barh(df["site"], df["real_npv_usd"] / 1000, color=colors, alpha=0.85)
    axes[0].set_xlabel("NPV, 20-year (USD thousands)")
    axes[0].set_title("Site ranking — real NPV (economics.py)")
    axes[0].grid(True, axis="x", alpha=0.3)

    axes[1].scatter(
        blueprints_df["Diesel_Litres"], blueprints_df["LCOE_USD"],
        c=[type_colors.get(t, "#333333") for t in blueprints_df["site_type"]],
        s=100, edgecolors="black",
    )
    for _, row in blueprints_df.iterrows():
        axes[1].annotate(row["site"], (row["Diesel_Litres"], row["LCOE_USD"]),
                          fontsize=8, xytext=(5, 5), textcoords="offset points")
    axes[1].set_xlabel("Residual diesel consumption (litres, simulated period)")
    axes[1].set_ylabel("LCOE (USD/kWh)")
    axes[1].set_title("Selected blueprint — LCOE vs. residual diesel, by site")
    axes[1].grid(True, alpha=0.3)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in type_colors.values()]
    fig.legend(handles, type_colors.keys(), loc="lower center", ncol=3, fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, "22_final_ranking.png")
