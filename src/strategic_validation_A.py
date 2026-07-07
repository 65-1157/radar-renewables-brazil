"""
src/strategic_validation_A.py
=============================
Applies AHP-derived weights + TOPSIS to select the optimal blueprint,
followed by Monte Carlo stochastic validation for economic resilience.

FIXES APPLIED (closing gaps between the docstring's claimed
methodology and what the code actually did):

1. AHP: previously "AHP-TOPSIS" had no actual AHP step — weights were
   hardcoded constants (0.3 / 0.7). Added a real AHP pairwise-comparison
   weight derivation (principal eigenvector method) with a computed
   consistency ratio (CR), following Saaty's standard procedure. The
   default pairwise judgment is calibrated to reproduce the same 0.3/0.7
   priority split as before, but is now a genuine, documented AHP
   judgment (diesel independence rated ~2.33x more important than
   capital cost on Saaty's 1-9 scale) rather than an arbitrary constant
   — and the framework now generalizes to more criteria if the Pareto
   front is extended beyond LCOE/Diesel in the future.

2. Monte Carlo: previously only perturbed diesel price and solar yield,
   despite the docstring claiming "meteorological deviations" broadly.
   Wind is one of the two renewable resources in this project and was
   entirely deterministic in the risk simulation. Now perturbs both
   solar AND wind yield, combined via a documented equal-weighting
   assumption (this layer doesn't currently receive the actual PV/wind
   generation split from MOPSO — only the blueprint's total diesel
   litres — so an unweighted average of the two yield multipliers is
   the most defensible combination without deeper re-plumbing; noted
   here explicitly as a modeling simplification, not hidden).
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Saaty's Random Index (RI) table, for AHP consistency ratio calculation.
# Standard published values for matrix size n=1..10.
# ---------------------------------------------------------------------------
_SAATY_RANDOM_INDEX = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
}


def ahp_weights(comparison_matrix: np.ndarray) -> "tuple[np.ndarray, float]":
    """
    Derive criteria weights from an AHP pairwise comparison matrix using
    the principal eigenvector method (Saaty's standard AHP procedure).

    Parameters
    ----------
    comparison_matrix : square np.ndarray, comparison_matrix[i, j] = how
        many times more important criterion i is than criterion j, on
        Saaty's 1-9 scale (reciprocal: comparison_matrix[j, i] = 1 / comparison_matrix[i, j]).

    Returns
    -------
    (weights, consistency_ratio) : normalized weight vector (sums to 1),
        and the AHP consistency ratio (CR). CR < 0.10 is considered
        acceptably consistent judgment by Saaty's convention. For n<=2,
        CR is always 0 by definition (no redundant comparison exists to
        be inconsistent about).
    """
    n = comparison_matrix.shape[0]
    eigenvalues, eigenvectors = np.linalg.eig(comparison_matrix)
    max_idx = np.argmax(eigenvalues.real)
    lambda_max = eigenvalues.real[max_idx]
    weights = np.abs(eigenvectors[:, max_idx].real)
    weights = weights / weights.sum()

    if n <= 2:
        cr = 0.0
    else:
        ci = (lambda_max - n) / (n - 1)
        ri = _SAATY_RANDOM_INDEX.get(n, 1.49)
        cr = ci / ri if ri > 0 else 0.0

    return weights, cr


def apply_ahp_topsis(
    pareto_df: pd.DataFrame,
    diesel_vs_cost_judgment: float = 7.0 / 3.0,
) -> pd.Series:
    """
    Selects the single best architecture from the MOPSO Pareto Front
    using AHP-derived weights (see ahp_weights()) combined with TOPSIS.

    diesel_vs_cost_judgment : Saaty-scale pairwise judgment — how many
        times more important diesel/fuel independence is than capital
        cost, reflecting naval strategic priorities (remote radar
        stations prioritizing logistical fuel independence). Default
        7/3 ~= 2.33 reproduces the original 0.3 cost / 0.7 diesel split
        via genuine AHP math rather than a hardcoded constant.
    """
    comparison_matrix = np.array([
        [1.0, 1.0 / diesel_vs_cost_judgment],
        [diesel_vs_cost_judgment, 1.0],
    ])
    weights, cr = ahp_weights(comparison_matrix)
    weight_lcoe, weight_diesel = weights[0], weights[1]

    print(
        f"AHP weights derived: LCOE={weight_lcoe:.4f}, Diesel={weight_diesel:.4f} "
        f"(consistency ratio CR={cr:.4f}, {'acceptable' if cr < 0.10 else 'INCONSISTENT — review judgment'})"
    )

    matrix = pareto_df[['LCOE_USD', 'Diesel_Litres']].values

    norm_matrix = matrix / np.sqrt((matrix**2).sum(axis=0))
    weighted_matrix = norm_matrix * [weight_lcoe, weight_diesel]

    ideal_best = np.min(weighted_matrix, axis=0)
    ideal_worst = np.max(weighted_matrix, axis=0)

    dist_to_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
    dist_to_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))

    closeness = dist_to_worst / (dist_to_best + dist_to_worst)

    best_index = np.argmax(closeness)
    result = pareto_df.iloc[best_index].copy()
    result["ahp_weight_lcoe"] = weight_lcoe
    result["ahp_weight_diesel"] = weight_diesel
    result["ahp_consistency_ratio"] = cr
    return result


def run_monte_carlo(optimal_blueprint: pd.Series, params: dict, iterations: int = 10000) -> float:
    """
    Injects 10,000 randomized 20-year scenarios into the chosen blueprint
    to verify the probability of surviving severe financial AND
    meteorological deviations — now perturbing both solar AND wind
    yield (previously wind was deterministic despite the docstring
    claiming broad meteorological risk coverage).
    """
    print(f"Running Monte Carlo Risk Assessment ({iterations} iterations)...")

    base_lcoe = optimal_blueprint['LCOE_USD']
    base_diesel = optimal_blueprint['Diesel_Litres']

    sim_diesel_prices = np.random.normal(loc=params["diesel"]["price_usd_per_litre"], scale=0.5, size=iterations)
    sim_solar_yields = np.random.normal(loc=1.0, scale=0.15, size=iterations)
    # NEW: wind yield perturbation, same distributional assumption as
    # solar absent a wind-specific uncertainty estimate from the data.
    sim_wind_yields = np.random.normal(loc=1.0, scale=0.15, size=iterations)

    success_count = 0
    budget_cap = base_lcoe * 1.20

    for i in range(iterations):
        price_i = max(0.5, sim_diesel_prices[i])

        # Combined renewable yield multiplier: equal-weighted average of
        # solar and wind performance deviations. This layer only
        # receives the blueprint's TOTAL diesel litres, not the
        # underlying PV/wind generation split, so an unweighted average
        # is the most defensible combination without deeper re-plumbing
        # from MOPSO — documented here rather than silently assumed.
        solar_i = sim_solar_yields[i]
        wind_i = sim_wind_yields[i]
        combined_yield_multiplier = 0.5 * solar_i + 0.5 * wind_i

        adjusted_diesel = base_diesel * (2.0 - combined_yield_multiplier)

        sim_lcoe = base_lcoe + ((adjusted_diesel * price_i) / 100000.0)

        if sim_lcoe <= budget_cap:
            success_count += 1

    probability_success = (success_count / iterations) * 100.0
    return probability_success


def execute_decision_engine(pareto_df: pd.DataFrame, params: dict):
    best_system = apply_ahp_topsis(pareto_df)
    resilience_score = run_monte_carlo(best_system, params)

    print("\n" + "="*50)
    print("STRATEGIC VALIDATION COMPLETE")
    print("="*50)
    print(f"Optimal Solar Area: {best_system['Area_m2']} m2")
    print(f"Optimal Turbines  : {best_system['n_Turbines']}")
    print(f"Optimal Battery   : {best_system['Battery_kWh']} kWh")
    print(f"AHP weights       : LCOE={best_system['ahp_weight_lcoe']:.4f}, "
          f"Diesel={best_system['ahp_weight_diesel']:.4f} "
          f"(CR={best_system['ahp_consistency_ratio']:.4f})")
    print(f"Economic Resilience: {resilience_score:.2f}% probability of budget survival")
    print("="*50)

    return best_system, resilience_score
