"""
main.py
=======
CLI entrypoint for Radar Renewables Brazil.

Subcommands
-----------
  run-pipeline   Load and validate NASA POWER data, print site summary
  run-scenarios  Full scenario sweep -> outputs/scenario_*.csv
  run-forecast   Probabilistic 15-day fan chart for one site + variable

Usage
-----
  cd ~/Desktop/projects/radar-renewables-brazil
  source .venv/bin/activate

  python main.py run-pipeline
  python main.py run-scenarios
  python main.py run-forecast --site "Natal" --variable solar
  python main.py run-forecast --site "Ilha da Trindade" --variable wind --days 15
  python main.py run-forecast --site "Natal" --variable solar --lstm --epochs 50
  python main.py run-forecast --site "Natal" --variable solar --lstm --evaluate
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_pipeline(args: argparse.Namespace) -> None:
    """Load data, run quality filter, print site summary."""
    from src.nasa_loader import (
        load_params, load_combined_csv, quality_filter,
        gap_fill, add_hub_height_wind, split_by_site, summarise_all,
    )

    params = load_params(args.config)
    df = load_combined_csv(params)
    df = quality_filter(df)
    df = gap_fill(df)
    df = add_hub_height_wind(df, params)
    all_data = split_by_site(df)

    print("\n=== Site data summary ===")
    summary = summarise_all(all_data)
    print(summary.to_string(index=False))
    print(f"\nTotal rows: {len(df)}")


def cmd_scenarios(args: argparse.Namespace) -> None:
    """Run full scenario sweep and save CSVs to outputs/."""
    from src.scenario_runner import run_all
    run_all(
        config_path=args.config,
        sites_config_path=args.sites,
        save_csv=not args.no_save,
        verbose=True,
    )


def cmd_forecast(args: argparse.Namespace) -> None:
    """
    Run probabilistic 15-day forecast for one site and variable.
    Prints the fan chart table and saves to outputs/forecast_<site>_<var>.csv
    """
    from src.nasa_loader import (
        load_params, load_combined_csv, quality_filter,
        gap_fill, add_hub_height_wind,
    )
    from src.forecaster import Forecaster

    params = load_params(args.config)
    df = load_combined_csv(params)
    df = quality_filter(df)
    df = gap_fill(df)
    df = add_hub_height_wind(df, params)

    site = args.site
    variable = args.variable
    n_days = args.days

    print(f"\nFitting forecaster — site: {site}, variable: {variable}")
    fcast = Forecaster(df, variable=variable)
    fcast.fit_empirical(sites=[site])

    if args.lstm:
        try:
            print(f"Training LSTM (epochs={args.epochs})...")
            val_losses = fcast.fit_lstm(
                site=site,
                epochs=args.epochs,
                verbose=True,
                force_retrain=args.retrain,
            )
            if val_losses:
                print(f"Final val pinball loss: {val_losses[-1]:.5f}")
        except ImportError:
            print(
                "WARNING: PyTorch not installed — falling back to empirical.\n"
                "Install with: pip install torch"
            )

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
        method = (
            "lstm"
            if args.lstm and site in fcast._lstm_fitted_sites
            else "empirical"
        )
        eval_df = fcast.evaluate(site=site, method=method, test_days=90)
        mean_loss = eval_df.groupby("label")["pinball_loss"].mean()
        print("\nMean pinball loss by quantile:")
        print(mean_loss.to_string())
        eval_path = out_dir / f"eval_{safe_site}_{variable}.csv"
        eval_df.to_csv(eval_path, index=False)
        print(f"Saved -> {eval_path}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Radar Renewables Brazil — feasibility & forecasting toolkit",
    )
    parser.add_argument(
        "--config",
        default="config/parameters.yaml",
        help="Path to parameters.yaml (default: config/parameters.yaml)",
    )
    parser.add_argument(
        "--sites",
        default="config/sites.yaml",
        help="Path to sites.yaml (default: config/sites.yaml)",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # -- run-pipeline -------------------------------------------------------
    sub.add_parser(
        "run-pipeline",
        help="Load NASA POWER data and print site summary",
    )

    # -- run-scenarios ------------------------------------------------------
    p_scen = sub.add_parser(
        "run-scenarios",
        help="Full solar/wind/diesel/economics scenario sweep",
    )
    p_scen.add_argument(
        "--no-save",
        action="store_true",
        help="Print results without saving CSVs",
    )

    # -- run-forecast -------------------------------------------------------
    p_fc = sub.add_parser(
        "run-forecast",
        help="Probabilistic 15-day fan chart for one site",
    )
    p_fc.add_argument(
        "--site",
        required=True,
        choices=[
            "Salvador",
            "Natal",
            "Fortaleza",
            "Cabo Frio",
            "Ilha Grande",
            "Ilha da Trindade",
        ],
        help="Site name",
    )
    p_fc.add_argument(
        "--variable",
        required=True,
        choices=["solar", "wind"],
        help="Variable to forecast",
    )
    p_fc.add_argument(
        "--days",
        type=int,
        default=15,
        metavar="N",
        help="Forecast horizon in days (default: 15)",
    )
    p_fc.add_argument(
        "--lstm",
        action="store_true",
        help="Train/use LSTM model (requires: pip install torch)",
    )
    p_fc.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="LSTM training epochs (default: 100, used with --lstm)",
    )
    p_fc.add_argument(
        "--retrain",
        action="store_true",
        help="Force LSTM retraining even if a saved checkpoint exists",
    )
    p_fc.add_argument(
        "--evaluate",
        action="store_true",
        help="Walk-forward pinball-loss evaluation on last 90 days",
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
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
