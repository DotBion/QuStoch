"""Quantum-assisted Monte Carlo simulation of stock prices under geometric Brownian motion.

The terminal log-return is discretised onto a fixed grid of ``2 ** n_disc`` points with
Gaussian weights.  That same (grid, weight) pair drives two independent calculations:

* a classical expectation, computed as a probability-weighted sum, and
* a quantum expectation, computed by amplitude estimation on a PennyLane simulator.

Because both consume the identical discretised distribution, the two numbers are directly
comparable and are both reported to the caller.
"""

from pathlib import Path
from time import time

import matplotlib

matplotlib.use("Agg")  # headless: the web server has no display

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

# Resolve output relative to this file so the app works from any working directory.
GRAPH_DIR = Path(__file__).resolve().parent / "static" / "images" / "graphs"

TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_WEEK = 5


def gaussian_grid(n_qubits, cutoff_factor=4.0):
    """Return a deterministic standard-normal grid and its renormalised probabilities.

    The grid has ``2 ** n_qubits`` points evenly spaced over ``[-cutoff, +cutoff]`` standard
    deviations.  A deterministic grid (rather than random draws) is what makes the classical
    and quantum expectations comparable: the quantum circuit encodes exactly these weights,
    so the classical side must average against them too.
    """
    dim = 2**n_qubits
    points = np.linspace(-cutoff_factor, cutoff_factor, dim)
    weights = np.exp(-0.5 * points**2)
    weights /= weights.sum()
    return points, weights


def terminal_prices(spot, rate, volatility, horizon, z):
    """Terminal GBM price for standard-normal draws ``z`` over ``horizon`` years.

    ``S_T = S_0 * exp(sigma * sqrt(T) * z + (r - sigma^2 / 2) * T)``

    Both the diffusion and the drift scale with the same horizon, which is what keeps the
    simulated spread consistent with the annualised volatility that was fed in.
    """
    drift = (rate - 0.5 * volatility**2) * horizon
    diffusion = volatility * np.sqrt(horizon) * z
    return spot * np.exp(drift + diffusion)


def brownian_bridge_paths(z, horizon, num_steps, seed=0):
    """Build GBM sample paths whose terminal Wiener values are exactly ``z * sqrt(horizon)``.

    Intermediate points come from a Brownian bridge, so each path is a genuine Brownian
    trajectory *and* lands on the discretised grid point it belongs to.  Path-dependent
    statistics (drawdown, strike crossings, autocorrelation) are therefore measured on real
    multi-step paths rather than on a straight line between two points.

    Returns ``(times, wiener_paths)`` with shapes ``(num_steps + 1,)`` and
    ``(len(z), num_steps + 1)``.
    """
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, horizon, num_steps + 1)
    increments = rng.normal(0.0, np.sqrt(np.diff(times)), size=(len(z), num_steps))
    unconditioned = np.concatenate(
        [np.zeros((len(z), 1)), np.cumsum(increments, axis=1)], axis=1
    )

    fraction = times / horizon
    endpoint = z[:, None] * np.sqrt(horizon)
    return times, unconditioned - fraction * unconditioned[:, -1:] + fraction * endpoint


def price_paths(spot, rate, volatility, times, wiener_paths):
    """Convert Wiener paths into GBM price paths."""
    drift = (rate - 0.5 * volatility**2) * times
    return spot * np.exp(drift + volatility * wiener_paths)


def weighted_moments(values, weights):
    """Return ``(mean, std, skewness, excess_kurtosis)`` under a probability weighting."""
    mean = float(np.sum(weights * values))
    variance = float(np.sum(weights * (values - mean) ** 2))
    std = float(np.sqrt(variance))
    if std == 0.0:
        return mean, 0.0, 0.0, 0.0
    standardised = (values - mean) / std
    skewness = float(np.sum(weights * standardised**3))
    kurtosis = float(np.sum(weights * standardised**4) - 3.0)
    return mean, std, skewness, kurtosis


def weighted_quantile(values, weights, q):
    """Quantile of a discrete weighted distribution."""
    order = np.argsort(values)
    sorted_values = np.asarray(values)[order]
    cumulative = np.cumsum(np.asarray(weights)[order])
    index = min(int(np.searchsorted(cumulative, q, side="left")), len(sorted_values) - 1)
    return float(sorted_values[index])


def call_price(spot, strike, rate, volatility, horizon, z, weights):
    """Discounted expected European call payoff over the discretised distribution."""
    prices = terminal_prices(spot, rate, volatility, horizon, z)
    payoff = np.maximum(prices - strike, 0.0)
    return float(np.exp(-rate * horizon) * np.sum(weights * payoff))


def option_greeks(spot, strike, rate, volatility, horizon, z, weights):
    """Finite-difference Delta, Gamma and Theta of the European call.

    The grid is deterministic, so bumping an input produces a clean derivative instead of
    the Monte Carlo noise that made these values collapse to zero previously.
    """
    bump = 0.01 * spot
    base = call_price(spot, strike, rate, volatility, horizon, z, weights)
    up = call_price(spot + bump, strike, rate, volatility, horizon, z, weights)
    down = call_price(spot - bump, strike, rate, volatility, horizon, z, weights)

    delta = (up - down) / (2 * bump)
    gamma = (up - 2 * base + down) / bump**2

    day = 1 / TRADING_DAYS_PER_YEAR
    if horizon > day:
        shortened = call_price(spot, strike, rate, volatility, horizon - day, z, weights)
        theta = shortened - base  # value lost over one trading day
    else:
        theta = None

    return delta, gamma, theta


def quantum_expectation(weights, values, n_disc, n_pe):
    """Estimate ``sum(weights * values)`` by quantum amplitude estimation.

    Phase estimation resolves the amplitude only up to the symmetry ``phase -> 1 - phase``,
    and reading the larger of the two peaks is ambiguous, so we take the peak in the lower
    half of the spectrum.  That is only the correct branch while ``E[f] <= 1/2``, which is
    guaranteed by normalising the payoff by *twice* its maximum: every scaled value then
    lies in ``[0, 1/2]``, and so does their mean.  Normalising by the plain maximum instead
    silently returns ``1 - E[f]`` for any payoff averaging above one half.
    """
    peak = float(np.max(values))
    if peak <= 0:
        return 0.0

    normalisation = 2.0 * peak
    scaled = values / normalisation

    target_wires = range(n_disc + 1)
    estimation_wires = range(n_disc + 1, n_disc + n_pe + 1)
    n_estimation_states = 2**n_pe

    dev = qml.device("lightning.qubit", wires=(n_disc + n_pe + 1))

    @qml.qnode(dev)
    def circuit():
        qml.templates.QuantumMonteCarlo(
            weights,
            lambda i: scaled[i],
            target_wires=target_wires,
            estimation_wires=estimation_wires,
        )
        return qml.probs(estimation_wires)

    probabilities = circuit()
    phase = np.argmax(probabilities[: n_estimation_states // 2]) / n_estimation_states
    return float((1 - np.cos(np.pi * phase)) / 2 * normalisation)


def path_statistics(paths, strike, weights):
    """Drawdown, strike-crossing and volatility-clustering statistics over price paths."""
    running_peak = np.maximum.accumulate(paths, axis=1)
    max_drawdown = np.max((running_peak - paths) / running_peak, axis=1)

    above_strike = paths > strike
    crossings = np.sum(above_strike[:, 1:] != above_strike[:, :-1], axis=1)

    log_returns = np.diff(np.log(paths), axis=1)
    if log_returns.shape[1] > 1:
        magnitudes = np.abs(log_returns)
        correlations = [
            np.corrcoef(row[:-1], row[1:])[0, 1]
            for row in magnitudes
            if np.std(row[:-1]) > 0 and np.std(row[1:]) > 0
        ]
        autocorrelation = float(np.mean(correlations)) if correlations else 0.0
    else:
        autocorrelation = 0.0

    return {
        "average_max_drawdown": float(np.sum(weights * max_drawdown)),
        "average_strike_crossings": float(np.sum(weights * crossings)),
        "maximum_strike_crossings": float(np.max(crossings)),
        "volatility_clustering": autocorrelation,
    }


def render_paths(times, paths, weights, spot, strike, mean_price):
    """Write the simulation chart to ``GRAPH_DIR`` and return the filename."""
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    mean_path = np.sum(weights[:, None] * paths, axis=0)
    std_path = np.sqrt(np.sum(weights[:, None] * (paths - mean_path) ** 2, axis=0))

    plt.figure(figsize=(10, 6))
    plt.plot(times, paths.T, color="steelblue", alpha=0.15, linewidth=0.8)
    plt.plot(times, mean_path, "r-", linewidth=2, label="Mean Path")
    plt.fill_between(
        times,
        mean_path - 2 * std_path,
        mean_path + 2 * std_path,
        alpha=0.2,
        color="gray",
        label="95% Confidence Band",
    )
    plt.axhline(y=strike, color="g", linestyle="--", alpha=0.5, label="Strike Price")

    plt.title("Quantum Monte Carlo Simulation Paths", fontsize=12, pad=15)
    plt.xlabel("Time (years)", fontsize=10)
    plt.ylabel("Stock Price ($)", fontsize=10)
    plt.legend(loc="upper left")

    plt.annotate(f"Strike: ${strike:.2f}", xy=(times[-1], strike), fontsize=8, color="g")
    plt.annotate(f"Start: ${spot:.2f}", xy=(times[0], spot), fontsize=8)
    plt.annotate(f"Mean End: ${mean_price:.2f}", xy=(times[-1], mean_path[-1]), fontsize=8)
    plt.tight_layout()

    filename = f"QMC_{round(time())}.svg"
    plt.savefig(GRAPH_DIR / filename, format="svg", bbox_inches="tight")
    plt.close()
    return filename


def Quantum_Monte_Carlo(
    horizon,
    spot=100.0,
    strike=100.0,
    rate=0.05,
    volatility=0.2,
    cutoff_factor=4.0,
    n_disc=6,
    n_pe=12,
    num_steps=64,
    seed=0,
    run_quantum=True,
    make_plot=True,
):
    """Run the full hybrid simulation over ``horizon`` years.

    Returns ``(quantum_estimate, filename, stats)`` where ``filename`` is ``None`` when
    plotting is disabled.
    """
    z, weights = gaussian_grid(n_disc, cutoff_factor)
    stock_prices = terminal_prices(spot, rate, volatility, horizon, z)

    mean_price, std_dev, skewness, kurtosis = weighted_moments(stock_prices, weights)
    predictive_interval = (
        weighted_quantile(stock_prices, weights, 0.025),
        weighted_quantile(stock_prices, weights, 0.975),
    )
    prob_increase = float(np.sum(weights[stock_prices > spot]))

    downside = spot - float(np.min(stock_prices))
    upside = float(np.max(stock_prices)) - spot
    risk_reward_ratio = upside / downside if downside > 0 else float("inf")

    times, wiener_paths = brownian_bridge_paths(z, horizon, num_steps, seed=seed)
    paths = price_paths(spot, rate, volatility, times, wiener_paths)
    path_stats = path_statistics(paths, strike, weights)

    option_value = call_price(spot, strike, rate, volatility, horizon, z, weights)
    delta, gamma, theta = option_greeks(
        spot, strike, rate, volatility, horizon, z, weights
    )

    stress_scenarios = {
        "High Volatility": {"volatility": volatility * 1.5},
        "Low Volatility": {"volatility": volatility * 0.5},
        "Market Crash": {"spot": spot * 0.8, "volatility": volatility * 1.3},
        "Market Boom": {"spot": spot * 1.2},
        "Low Interest Rate": {"rate": rate * 0.5},
        "High Interest Rate": {"rate": rate * 1.5},
    }
    stress_results = {}
    for scenario, overrides in stress_scenarios.items():
        stressed = terminal_prices(
            overrides.get("spot", spot),
            overrides.get("rate", rate),
            overrides.get("volatility", volatility),
            horizon,
            z,
        )
        stress_results[scenario] = float(np.sum(weights * stressed))

    quantum_estimate = (
        quantum_expectation(weights, stock_prices, n_disc, n_pe) if run_quantum else None
    )

    filename = (
        render_paths(times, paths, weights, spot, strike, mean_price)
        if make_plot
        else None
    )

    stats = {
        "inference_statistics": {
            "estimated_price": round(mean_price, 2),
            "quantum_estimated_price": (
                round(quantum_estimate, 2) if quantum_estimate is not None else None
            ),
            "quantum_vs_classical_error": (
                round(abs(quantum_estimate - mean_price), 4)
                if quantum_estimate is not None
                else None
            ),
            "predictive_interval_95": {
                "lower": round(predictive_interval[0], 2),
                "upper": round(predictive_interval[1], 2),
            },
            "standard_deviation": std_dev,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "probability_of_increase": prob_increase,
            "risk_reward_ratio": risk_reward_ratio,
        },
        "option": {
            "strike": strike,
            "call_price": option_value,
            "horizon_years": horizon,
        },
        "greeks": {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
        },
        "stress_test_results": {
            scenario: {
                "price": price,
                "percent_change": (price / mean_price - 1) * 100,
            }
            for scenario, price in stress_results.items()
        },
        "path_dependent_stats": path_stats,
    }

    return quantum_estimate, filename, stats


def Q_sim_start(S0, K, T_weeks, r, sigma, steps_per_day=8):
    """Entry point used by the web app.

    ``T_weeks`` is the maturity in weeks, as collected by the dashboard form.  ``r`` and
    ``sigma`` come from live market data and are passed through to the simulation.
    """
    horizon = T_weeks * TRADING_DAYS_PER_WEEK / TRADING_DAYS_PER_YEAR
    if horizon <= 0:
        raise ValueError("maturity must be a positive number of weeks")

    num_steps = max(8, int(round(horizon * TRADING_DAYS_PER_YEAR * steps_per_day)))

    _, filename, stats = Quantum_Monte_Carlo(
        horizon=horizon,
        spot=S0,
        strike=K,
        rate=r,
        volatility=sigma,
        num_steps=num_steps,
    )
    return filename, stats


if __name__ == "__main__":
    from pprint import pprint

    name, summary = Q_sim_start(100, 105, 4, 0.05, 0.2)
    print(f"chart: {name}")
    pprint(summary)
