import numpy as np
import pytest
from scipy.stats import norm

from quantumMonteCarloStochastic import (
    Q_sim_start,
    Quantum_Monte_Carlo,
    brownian_bridge_paths,
    call_price,
    gaussian_grid,
    option_greeks,
    path_statistics,
    price_paths,
    quantum_expectation,
    terminal_prices,
    weighted_moments,
    weighted_quantile,
)

SPOT = 100.0
STRIKE = 105.0
RATE = 0.05
VOLATILITY = 0.2
HORIZON = 0.5  # years


def black_scholes_call(spot, strike, rate, volatility, horizon):
    """Closed-form European call price, used as an independent reference."""
    d1 = (np.log(spot / strike) + (rate + 0.5 * volatility**2) * horizon) / (
        volatility * np.sqrt(horizon)
    )
    d2 = d1 - volatility * np.sqrt(horizon)
    return spot * norm.cdf(d1) - strike * np.exp(-rate * horizon) * norm.cdf(d2)


class TestGaussianGrid:
    def test_shape_and_normalisation(self):
        points, weights = gaussian_grid(6)
        assert len(points) == 64
        assert len(weights) == 64
        assert weights.sum() == pytest.approx(1.0)
        assert np.all(weights > 0)

    def test_symmetric_about_zero(self):
        points, weights = gaussian_grid(5)
        assert points[0] == pytest.approx(-points[-1])
        assert weights == pytest.approx(weights[::-1])
        assert np.sum(weights * points) == pytest.approx(0.0, abs=1e-12)

    def test_is_deterministic(self):
        """Determinism is what makes the classical and quantum sides comparable."""
        assert gaussian_grid(5)[0] == pytest.approx(gaussian_grid(5)[0])

    def test_approximates_unit_variance(self):
        points, weights = gaussian_grid(8)
        assert np.sum(weights * points**2) == pytest.approx(1.0, rel=0.02)


class TestTerminalPrices:
    def test_martingale_property(self):
        """The discounted expected price must equal today's price."""
        z, weights = gaussian_grid(8)
        prices = terminal_prices(SPOT, RATE, VOLATILITY, HORIZON, z)
        expected = np.sum(weights * prices)
        assert expected == pytest.approx(SPOT * np.exp(RATE * HORIZON), rel=1e-3)

    def test_spread_matches_annualised_volatility(self):
        """Diffusion scales with sqrt(T); a mismatch here was the original scaling bug."""
        z, weights = gaussian_grid(8)
        for horizon in (0.08, 0.25, 1.0):
            prices = terminal_prices(SPOT, RATE, VOLATILITY, horizon, z)
            _, std, _, _ = weighted_moments(prices, weights)
            assert std == pytest.approx(
                SPOT * VOLATILITY * np.sqrt(horizon), rel=0.1
            )

    def test_zero_volatility_is_deterministic_growth(self):
        z, _ = gaussian_grid(4)
        prices = terminal_prices(SPOT, RATE, 0.0, HORIZON, z)
        assert prices == pytest.approx(SPOT * np.exp(RATE * HORIZON))

    def test_prices_stay_positive(self):
        z, _ = gaussian_grid(6)
        prices = terminal_prices(SPOT, RATE, 1.5, 2.0, z)
        assert np.all(prices > 0)


class TestWeightedStatistics:
    def test_uniform_weights_match_numpy(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        weights = np.full(4, 0.25)
        mean, std, _, _ = weighted_moments(values, weights)
        assert mean == pytest.approx(values.mean())
        assert std == pytest.approx(values.std())

    def test_symmetric_distribution_has_zero_skew(self):
        z, weights = gaussian_grid(7)
        _, _, skew, kurtosis = weighted_moments(z, weights)
        assert skew == pytest.approx(0.0, abs=1e-9)
        assert kurtosis == pytest.approx(0.0, abs=0.1)

    def test_constant_values_have_no_spread(self):
        values = np.full(8, 7.0)
        mean, std, skew, kurtosis = weighted_moments(values, np.full(8, 0.125))
        assert (mean, std, skew, kurtosis) == (7.0, 0.0, 0.0, 0.0)

    def test_quantiles_are_ordered_and_bounded(self):
        z, weights = gaussian_grid(8)
        low = weighted_quantile(z, weights, 0.025)
        median = weighted_quantile(z, weights, 0.5)
        high = weighted_quantile(z, weights, 0.975)
        assert low < median < high
        assert median == pytest.approx(0.0, abs=0.1)
        assert low == pytest.approx(-1.96, abs=0.2)
        assert high == pytest.approx(1.96, abs=0.2)

    def test_extreme_quantiles_stay_in_range(self):
        z, weights = gaussian_grid(5)
        assert weighted_quantile(z, weights, 1.0) == pytest.approx(z[-1])
        assert weighted_quantile(z, weights, 0.0) == pytest.approx(z[0])


class TestBrownianBridge:
    def test_shape_and_endpoints(self):
        z, _ = gaussian_grid(5)
        times, wiener = brownian_bridge_paths(z, HORIZON, num_steps=32)
        assert times.shape == (33,)
        assert wiener.shape == (32, 33)
        assert np.all(wiener[:, 0] == 0.0)

    def test_terminal_value_hits_the_grid_point(self):
        """Each path must land on the grid point the quantum circuit encoded."""
        z, _ = gaussian_grid(5)
        _, wiener = brownian_bridge_paths(z, HORIZON, num_steps=48)
        assert wiener[:, -1] == pytest.approx(z * np.sqrt(HORIZON))

    def test_paths_are_not_straight_lines(self):
        """The original code emitted two-point paths, making path statistics degenerate."""
        z, _ = gaussian_grid(5)
        times, wiener = brownian_bridge_paths(z, HORIZON, num_steps=64)
        straight = np.outer(wiener[:, -1], times / HORIZON)
        assert np.max(np.abs(wiener - straight)) > 1e-6

    def test_reproducible_for_a_given_seed(self):
        z, _ = gaussian_grid(4)
        first = brownian_bridge_paths(z, HORIZON, 16, seed=7)[1]
        second = brownian_bridge_paths(z, HORIZON, 16, seed=7)[1]
        assert first == pytest.approx(second)

    def test_price_paths_start_at_spot(self):
        z, _ = gaussian_grid(4)
        times, wiener = brownian_bridge_paths(z, HORIZON, 16)
        paths = price_paths(SPOT, RATE, VOLATILITY, times, wiener)
        assert paths[:, 0] == pytest.approx(SPOT)
        assert np.all(paths > 0)


class TestOptionPricing:
    def test_matches_black_scholes(self):
        z, weights = gaussian_grid(10)
        priced = call_price(SPOT, STRIKE, RATE, VOLATILITY, HORIZON, z, weights)
        assert priced == pytest.approx(
            black_scholes_call(SPOT, STRIKE, RATE, VOLATILITY, HORIZON), rel=0.01
        )

    @pytest.mark.parametrize("strike", [70.0, 100.0, 130.0])
    def test_matches_black_scholes_across_moneyness(self, strike):
        z, weights = gaussian_grid(10)
        priced = call_price(SPOT, strike, RATE, VOLATILITY, HORIZON, z, weights)
        reference = black_scholes_call(SPOT, strike, RATE, VOLATILITY, HORIZON)
        assert priced == pytest.approx(reference, rel=0.02, abs=0.05)

    def test_deep_out_of_the_money_is_worthless(self):
        z, weights = gaussian_grid(8)
        assert call_price(SPOT, 10_000.0, RATE, VOLATILITY, HORIZON, z, weights) == 0.0

    def test_price_increases_with_volatility(self):
        z, weights = gaussian_grid(8)
        cheap = call_price(SPOT, STRIKE, RATE, 0.1, HORIZON, z, weights)
        dear = call_price(SPOT, STRIKE, RATE, 0.4, HORIZON, z, weights)
        assert dear > cheap


class TestGreeks:
    def test_delta_matches_black_scholes(self):
        z, weights = gaussian_grid(10)
        delta, _, _ = option_greeks(SPOT, STRIKE, RATE, VOLATILITY, HORIZON, z, weights)
        d1 = (np.log(SPOT / STRIKE) + (RATE + 0.5 * VOLATILITY**2) * HORIZON) / (
            VOLATILITY * np.sqrt(HORIZON)
        )
        assert delta == pytest.approx(norm.cdf(d1), abs=0.02)

    def test_greeks_are_not_degenerate(self):
        """Delta, Gamma and Theta all collapsed to ~0 before the payoff had curvature."""
        z, weights = gaussian_grid(8)
        delta, gamma, theta = option_greeks(
            SPOT, STRIKE, RATE, VOLATILITY, HORIZON, z, weights
        )
        assert 0.0 < delta < 1.0
        assert gamma > 0.0
        assert theta is not None and theta < 0.0  # a long call decays

    def test_theta_is_none_for_sub_daily_horizons(self):
        z, weights = gaussian_grid(6)
        _, _, theta = option_greeks(SPOT, STRIKE, RATE, VOLATILITY, 1 / 504, z, weights)
        assert theta is None


class TestQuantumEstimation:
    def test_agrees_with_classical_expectation(self):
        z, weights = gaussian_grid(4)
        prices = terminal_prices(SPOT, RATE, VOLATILITY, HORIZON, z)
        classical = float(np.sum(weights * prices))
        estimate = quantum_expectation(weights, prices, n_disc=4, n_pe=10)
        assert estimate == pytest.approx(classical, abs=0.5)

    def test_handles_payoffs_averaging_above_half_the_maximum(self):
        """Regression: normalising by the plain maximum returned ``1 - E[f]`` here."""
        z, weights = gaussian_grid(4)
        prices = terminal_prices(SPOT, RATE, 0.05, HORIZON, z)
        classical = float(np.sum(weights * prices))
        assert classical > 0.5 * float(np.max(prices))  # the regime that used to break
        estimate = quantum_expectation(weights, prices, n_disc=4, n_pe=10)
        assert estimate == pytest.approx(classical, rel=0.02)

    def test_zero_payoff_returns_zero(self):
        _, weights = gaussian_grid(3)
        assert quantum_expectation(weights, np.zeros(8), n_disc=3, n_pe=6) == 0.0


class TestPathStatistics:
    def _paths(self, num_steps=64):
        z, weights = gaussian_grid(5)
        times, wiener = brownian_bridge_paths(z, HORIZON, num_steps)
        return price_paths(SPOT, RATE, VOLATILITY, times, wiener), weights

    def test_values_are_in_range(self):
        paths, weights = self._paths()
        stats = path_statistics(paths, STRIKE, weights)
        assert 0.0 <= stats["average_max_drawdown"] < 1.0
        assert stats["average_strike_crossings"] >= 0.0
        assert stats["maximum_strike_crossings"] >= stats["average_strike_crossings"]
        assert -1.0 <= stats["volatility_clustering"] <= 1.0

    def test_multi_step_paths_produce_real_drawdowns(self):
        paths, weights = self._paths()
        assert path_statistics(paths, STRIKE, weights)["average_max_drawdown"] > 0.0

    def test_unreachable_strike_is_never_crossed(self):
        paths, weights = self._paths()
        assert path_statistics(paths, 1e9, weights)["maximum_strike_crossings"] == 0.0


class TestEndToEnd:
    def test_simulation_reports_both_estimates(self):
        estimate, filename, stats = Quantum_Monte_Carlo(
            horizon=0.25,
            spot=SPOT,
            strike=STRIKE,
            rate=RATE,
            volatility=VOLATILITY,
            n_disc=4,
            n_pe=10,
            num_steps=32,
            make_plot=False,
        )
        assert filename is None
        inference = stats["inference_statistics"]
        assert inference["quantum_estimated_price"] is not None
        assert estimate == pytest.approx(inference["quantum_estimated_price"], abs=0.01)
        # The quantum and classical routes must agree on the same distribution.
        assert inference["quantum_vs_classical_error"] < 1.0

    def test_quantum_stage_can_be_skipped(self):
        estimate, _, stats = Quantum_Monte_Carlo(
            horizon=0.25, n_disc=4, num_steps=16, run_quantum=False, make_plot=False
        )
        assert estimate is None
        assert stats["inference_statistics"]["quantum_estimated_price"] is None

    def test_market_inputs_reach_the_simulation(self):
        """Q_sim_start used to drop the live rate and volatility on the floor."""
        low = Quantum_Monte_Carlo(
            horizon=0.25, volatility=0.1, num_steps=16, run_quantum=False, make_plot=False
        )[2]
        high = Quantum_Monte_Carlo(
            horizon=0.25, volatility=0.6, num_steps=16, run_quantum=False, make_plot=False
        )[2]
        low_std = low["inference_statistics"]["standard_deviation"]
        high_std = high["inference_statistics"]["standard_deviation"]
        assert high_std > 3 * low_std

    def test_q_sim_start_writes_a_chart(self):
        from quantumMonteCarloStochastic import GRAPH_DIR

        filename, stats = Q_sim_start(SPOT, STRIKE, 4, RATE, VOLATILITY)
        chart = GRAPH_DIR / filename
        try:
            assert chart.exists() and chart.stat().st_size > 0
            assert set(stats) == {
                "inference_statistics",
                "option",
                "greeks",
                "stress_test_results",
                "path_dependent_stats",
            }
            assert stats["option"]["horizon_years"] == pytest.approx(20 / 252)
        finally:
            chart.unlink(missing_ok=True)

    @pytest.mark.parametrize("weeks", [0, -3])
    def test_non_positive_maturity_is_rejected(self, weeks):
        with pytest.raises(ValueError):
            Q_sim_start(SPOT, STRIKE, weeks, RATE, VOLATILITY)
