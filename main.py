"""
main.py
=======
CLI entrypoint for Radar Renewables Brazil.

Subcommands
-----------
  run-pipeline    Load and validate NASA POWER data, print site summary
  run-scenarios   Full scenario sweep -> outputs/scenario_*.csv
  run-forecast    Probabilistic 15-day fan chart for one site + variable
  run-comparison  Pinball loss table: persistence vs empirical vs LSTM
                  across all 6 sites and both variables

Usage
-----
  python main.py run-pipeline
  python main.py run-scenarios
  python main.py run-forecast --site "Natal" --variable solar
  python main.py run-forecast --site "Natal" --variable solar --lstm --epochs 50
  python main.py run-forecast --site "Natal" --variable solar --lstm --tune --n-trials 20
  python main.py run-forecast --site "Natal" --variable solar --lstm --tune --evaluate
  python main.py run-comparison
  python main.py run-comparison --variable solar
  python main.py run-comparison --test-days 90
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Shared data loader
# ---------------------------------------------------------------------------

def _load_df(config: str):
    from src.nasa_loader import (
        load_params, load_combined_csv, quality_filter,
        gap_fill, add_hub_height_wind,
    )
    params = load_params(config)
    df = load_combined_csv(params)
    df = quality_filter(df)
    df = gap_fill(df)
    df = add_hub_height_wind(df, params)
    return df


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_pipeline(args: argparse.Namespace) -> None:
    from src.nasa_loader import split_by_site, summarise_all
    df = _load_df(args.config)
    all_data = split_by_site(df)
    print("\n=== Site data summary ===")
    print(summarise_all(all_data).to_string(index=False))
    print(f"\nTotal rows: {len(df)}")


def cmd_scenarios(args: argparse.Namespace) -> None:
    from src.scenario_runner import run_all
    run_all(
        config_path=args.config,
        sites_config_path=args.sites,
        save_csv=not args.no_save,
        verbose=True,
    )


def cmd_forecast(args: argparse.Namespace) -> None:
    from src.forecaster import Forecaster

    df = _load_df(args.config)
    site = args.site
    variable = args.variable
    n_days = args.days

    print(f"\nFitting forecaster — site: {site}, variable: {variable}")
    fcast = Forecaster(df, variable=variable)
    fcast.fit_empirical(sites=[site])

    if args.lstm:
        try:
            if args.tune:
                print(f"Running Optuna search ({args.n_trials} trials)...")
                best = fcast.tune_lstm(
                    site=site,
                    n_trials=args.n_trials,
                    verbose=True,
                    force_retune=args.retrain,
                )
                print(f"Best params: {best}")

            print(f"Training LSTM (epochs={args.epochs})...")
            train_losses, val_losses = fcast.fit_lstm(
                site=site,
                epochs=args.epochs,
                verbose=True,
                force_retrain=args.retrain,
            )
            if val_losses:
                print(
                    f"Final train={train_losses[-1]:.5f}  "
                    f"val={val_losses[-1]:.5f}  "
                    f"gap={val_losses[-1]-train_losses[-1]:.5f}"
                )
                from src.visualiser import plot_train_val_loss
                plot_train_val_loss(train_losses, val_losses, site, variable)

        except ImportError as e:
            print(f"WARNING: {e} — falling back to empirical.")

    fan = fcast.forecast(site=site, n_days=n_days)
    print(f"\n=== {n_days}-day forecast: {site} / {variable} ===")
    print(fan.to_string())

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    safe_site = site.replace(" ", "_").replace("/", "_")
    out_path = out_dir / f"forecast_{safe_site}_{variable}.csv"
    fan.to_csv(out_path)
    print(f"\nSaved -> {out_path}")

    if args.evaluate:
        print("\nRunning walk-forward evaluation (last 90 days)...")
        comp = fcast.compare_methods(site=site, test_days=90)
        print("\nMean pinball loss — method comparison:")
        print(comp.to_string(index=False))
        comp_path = out_dir / f"compare_{safe_site}_{variable}.csv"
        comp.to_csv(comp_path, index=False)
        print(f"Saved -> {comp_path}")

    if args.shap and args.lstm:
        print("\nComputing SHAP importance...")
        importance = fcast.compute_shap(site=site)
        from src.visualiser import plot_shap_importance
        plot_shap_importance(importance, site, variable)
        print("SHAP plot saved.")

    if args.lime and args.lstm:
        print("\nComputing LIME explanation for last forecast, Q50, day+1...")
        exp = fcast.explain_forecast_lime(
            site=site, quantile_idx=2, horizon_idx=0
        )
        print("LIME top-10 features:")
        for feat, weight in exp.as_list():
            print(f"  {feat:20s}  {weight:+.5f}")


def cmd_scenarios_quantiles(args: argparse.Namespace) -> None:
    """Run Q10/Q50/Q90 scenario sweep."""
    from src.scenario_runner import run_all_quantiles
    run_all_quantiles(
        config_path=args.config,
        sites_config_path=args.sites,
        save_csv=not args.no_save,
        verbose=True,
    )


def cmd_comparison(args: argparse.Namespace) -> None:
    """
    Run walk-forward pinball loss comparison across all sites.
    Produces:
      outputs/comparison_<variable>.csv   — full table
      outputs/13_comparison_<variable>.png — heatmap
    """
    import pandas as pd
    from src.forecaster import Forecaster, SITES
    from src.visualiser import plot_comparison_heatmap

    df = _load_df(args.config)
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    variables = (
        [args.variable] if args.variable
        else ["solar", "wind"]
    )

    for variable in variables:
        print(f"\n{'='*60}")
        print(f"METHOD COMPARISON — {variable.upper()} — all sites")
        print(f"{'='*60}")

        fcast = Forecaster(df, variable=variable)
        fcast.fit_empirical()

        all_rows = []
        for site in SITES:
            print(f"  Evaluating {site}...", end=" ", flush=True)
            try:
                comp = fcast.compare_methods(
                    site=site, test_days=args.test_days
                )
                comp.insert(0, "site", site)
                all_rows.append(comp)
                print("done")
            except Exception as e:
                print(f"SKIPPED ({e})")

        if not all_rows:
            print("No results — check data.")
            continue

        full_table = pd.concat(all_rows, ignore_index=True)

        # Print summary
        print(f"\nMean pinball loss — {variable} — all sites:")
        print(full_table.to_string(index=False))

        # Save CSV
        csv_path = out_dir / f"comparison_{variable}.csv"
        full_table.to_csv(csv_path, index=False)
        print(f"\nSaved -> {csv_path}")

        # Save heatmap
        plot_comparison_heatmap(full_table, variable)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Radar Renewables Brazil — feasibility & forecasting toolkit",
    )
    parser.add_argument(
        "--config", default="config/parameters.yaml",
        help="Path to parameters.yaml",
    )
    parser.add_argument(
        "--sites", default="config/sites.yaml",
        help="Path to sites.yaml",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # -- run-pipeline -------------------------------------------------------
    sub.add_parser("run-pipeline", help="Load data and print site summary")

    # -- run-scenarios ------------------------------------------------------
    p_scen = sub.add_parser("run-scenarios", help="Full scenario sweep")
    p_scen.add_argument("--no-save", action="store_true")

    # -- run-forecast -------------------------------------------------------
    p_fc = sub.add_parser(
        "run-forecast", help="15-day probabilistic forecast for one site"
    )
    p_fc.add_argument(
        "--site", required=True,
        choices=[
            "Salvador", "Natal", "Fortaleza",
            "Cabo Frio", "Ilha Grande", "Ilha da Trindade",
        ],
    )
    p_fc.add_argument(
        "--variable", required=True, choices=["solar", "wind"]
    )
    p_fc.add_argument("--days", type=int, default=15, metavar="N")
    p_fc.add_argument(
        "--lstm", action="store_true",
        help="Train/use LSTM (requires: pip install torch)",
    )
    p_fc.add_argument(
        "--tune", action="store_true",
        help="Run Optuna search before training (requires --lstm)",
    )
    p_fc.add_argument(
        "--n-trials", type=int, default=20, metavar="N",
        help="Number of Optuna trials (default: 20)",
    )
    p_fc.add_argument("--epochs", type=int, default=100, metavar="N")
    p_fc.add_argument(
        "--retrain", action="store_true",
        help="Force retrain even if checkpoint exists",
    )
    p_fc.add_argument(
        "--evaluate", action="store_true",
        help="Walk-forward pinball loss + method comparison table",
    )
    p_fc.add_argument(
        "--shap", action="store_true",
        help="Compute and plot SHAP feature importance (requires --lstm)",
    )
    p_fc.add_argument(
        "--lime", action="store_true",
        help="Print LIME explanation for last forecast (requires --lstm)",
    )

    # -- run-scenarios-quantiles --------------------------------------------
    p_scen_q = sub.add_parser(
        "run-scenarios-quantiles",
        help="Q10/Q50/Q90 diesel + economics sweep -> outputs/scenario_*_quantiles.csv",
    )
    p_scen_q.add_argument("--no-save", action="store_true")

    # -- run-comparison -----------------------------------------------------
    p_cmp = sub.add_parser(
        "run-comparison",
        help=(
            "Pinball loss table: persistence vs empirical vs LSTM "
            "across all 6 sites"
        ),
    )
    p_cmp.add_argument(
        "--variable",
        choices=["solar", "wind"],
        default=None,
        help="Limit to one variable (default: both)",
    )
    p_cmp.add_argument(
        "--test-days",
        type=int,
        default=90,
        metavar="N",
        help="Walk-forward test window in days (default: 90)",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    dispatch = {
        "run-pipeline": cmd_pipeline,
        "run-scenarios": cmd_scenarios,
        "run-forecast": cmd_forecast,
        "run-scenarios-quantiles": cmd_scenarios_quantiles,
        "run-comparison": cmd_comparison,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
