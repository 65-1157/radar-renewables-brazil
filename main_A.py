"""
main_A.py
=========
Upgraded CLI entrypoint for Radar Renewables Brazil.
Preserves all legacy scenario testing while introducing the
Predictive-Strategic Techno-Economic Framework (PSTEF).

New Subcommand:
---------------
  run-pstef       Executes the full Tri-Layer Architecture:
                  Forecaster (LSTM/N-BEATS, winner-aware) -> MOPSO Swarm
                  Sizing -> AHP-TOPSIS & Monte Carlo Risk

Usage:
------
  python main_A.py run-pstef --site "Ilha da Trindade"

FIX APPLIED: this file previously used a hardcoded mock `params` dict
with only solar/wind/diesel/economics(partial) keys — a leftover from
the original scaffold, never replaced despite every other file in this
project having its mock/placeholder logic fixed already. Running
`python main_A.py run-pstef` as-written would have crashed the moment
it reached build_load_series() (params["load"] missing entirely) or
the economics.py wiring added later (params["economics"] missing the
solar_capex_usd_per_kw/wind_capex_usd_per_kw/battery_capex_usd_per_kwh/
opex_pct_capex keys). Now uses the real config loader, matching
src/main_pstef_A.py's own __main__ block and everything else in this
pipeline.
"""

import argparse
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", category=Warning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*converge.*")
warnings.filterwarnings("ignore", message=".*mle_retvals.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Import legacy commands from your existing codebase
# from main import (cmd_pipeline, cmd_scenarios, cmd_forecast,
#                   cmd_dispatch, cmd_autonomy, cmd_scenarios_quantiles, cmd_comparison)

from src.main_pstef_A import run_pstef_pipeline
from src.nasa_loader import load_params


def cmd_pstef(args: argparse.Namespace) -> None:
    """
    Executes the PSTEF MOPSO optimization and risk assessment pipeline
    for one site, using the REAL project config (config/parameters.yaml)
    — not a hardcoded mock dict.
    """
    print(f"Starting PSTEF Optimization Pipeline for {args.site}...")

    params = load_params()

    final_blueprint, risk_score = run_pstef_pipeline(args.site, params)
    return final_blueprint, risk_score


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Radar Renewables Brazil - PSTEF Upgraded")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- NEW ARCHITECTURE COMMAND ---
    p_pstef = sub.add_parser(
        "run-pstef",
        help="Run the MOPSO Swarm Optimization and Monte Carlo Risk Validation",
    )
    p_pstef.add_argument(
        "--site",
        type=str,
        default="Ilha da Trindade",
        help="Target naval radar station to optimize",
    )

    # --- LEGACY COMMANDS ---
    # NOTE: these subcommands are ACCEPTED by argparse below (so
    # `python main_A.py run-pipeline` won't error at the CLI-parsing
    # stage), but their handler functions remain commented out above and
    # are NOT in the `dispatch` dict in main(). Running them currently
    # falls through to parser.print_help() silently — misleading, since
    # the subcommand looks valid but does nothing. If the legacy
    # pipeline is still needed standalone (outside PSTEF), wire these
    # commands to the real functions in src/scenario_runner.py /
    # src/main.py; otherwise consider removing these subparsers
    # entirely so an unimplemented command fails loudly (argparse error)
    # instead of silently printing help.
    p_pipe = sub.add_parser("run-pipeline", help="Load NASA POWER data and print summary")
    p_scen = sub.add_parser("run-scenarios", help="Run legacy discrete scenario sweep")

    # ... (Keep all other sub.add_parser definitions from your original main.py here) ...

    return parser


def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "run-pstef": cmd_pstef,
        # "run-pipeline": cmd_pipeline,
        # "run-scenarios": cmd_scenarios,
        # "run-forecast": cmd_forecast,
        # "run-dispatch": cmd_dispatch,
        # "run-autonomy": cmd_autonomy,
        # "run-scenarios-quantiles": cmd_scenarios_quantiles,
        # "run-comparison": cmd_comparison,
    }

    func = dispatch.get(args.command)
    if func:
        func(args)
    else:
        print(
            f"\n'{args.command}' is accepted by the CLI parser but has no "
            f"wired handler yet (legacy command, not part of PSTEF). "
            f"Showing help instead:\n"
        )
        parser.print_help()


if __name__ == "__main__":
    main()
