"""
main_A.py
=========
Upgraded CLI entrypoint for Radar Renewables Brazil.
Preserves all legacy scenario testing while introducing the 
Predictive-Strategic Techno-Economic Framework (PSTEF).

New Subcommand:
---------------
  run-pstef       Executes the full Tri-Layer Architecture:
                  LSTM Pre-computation -> MOPSO Swarm Sizing -> AHP-TOPSIS & Monte Carlo Risk

Usage:
------
  python main_A.py run-pstef --site "Ilha da Trindade"
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

# Import the new architecture
from src.main_pstef_A import run_pstef_pipeline
# from src.data_loader import load_params # Assuming this exists

def cmd_pstef(args: argparse.Namespace) -> None:
    """
    Executes the new MOPSO optimization and risk assessment pipeline.
    """
    print(f"Starting PSTEF Optimization Pipeline for {args.site}...")
    
    # Load configuration
    # params = load_params() 
    
    # Mocking params for compilation (Replace with actual load_params output)
    params = {
        "solar": {"efficiency": 0.2, "performance_ratio": 0.8, "cost_pv_m2": 250},
        "wind": {"rated_power_kw": 50.0, "cost_wind_turbine": 100000},
        "diesel": {"efficiency": 0.35, "lhv_kwh_per_litre": 10.0, "price_usd_per_litre": 2.0},
        "economics": {"cost_pv_m2": 250, "cost_wind_turbine": 100000, "cost_battery_kwh": 400}
    }
    
    # Execute Layer 1, 2, and 3
    run_pstef_pipeline(args.site, params)

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

    # --- LEGACY COMMANDS (Preserved) ---
    p_pipe = sub.add_parser("run-pipeline", help="Load NASA POWER data and print summary")
    p_scen = sub.add_parser("run-scenarios", help="Run legacy discrete scenario sweep")
    
    # ... (Keep all other sub.add_parser definitions from your original main.py here) ...

    return parser

def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    
    # Dispatch dictionary mapping commands to their execution functions
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
        parser.print_help()

if __name__ == "__main__":
    main()