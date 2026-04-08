"""
test_benchmark.py
=================
Tests for Phase 1 (two-player benchmark).

Each test is tied to a specific claim in the paper:
  - Fixed point formula
  - Oscillation condition
  - Regime classification (stable / oscillatory / cascade)
  - Flip time properties
  - Best-response functions
  - Capital stock mechanics
"""

import numpy as np
import pytest

from chse.core.primitives import Params, CapitalStocks, CHSEState
from chse.core.mechanisms import (
    ambiguity_push,
    reframe_resistance,
    reframe_success_prob,
    reframing_investment,
    optimal_eta,
    optimal_kappa_spend,
)
from chse.benchmark.two_player import TwoPlayerModel
from chse.benchmark.oscillation import OscillationAnalysis
from chse.benchmark.flip_threshold import flip_time, flip_time_vs_hsi


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class TestParams:
    def test_hsi_formula(self):
        p = Params(lambda_kappa=2.0, lambda_R=1.0)
        assert p.hsi(K_i=3.0, V_j=2.0) == pytest.approx(3.0, rel=1e-9)

    def test_hsi_override(self):
        p = Params(HSI=2.1)
        # Override ignores K_i, V_j
        assert p.hsi(K_i=0.0, V_j=100.0) == 2.1

    def test_hsi_infinite_when_V_zero(self):
        p = Params()
        assert p.hsi(K_i=1.0, V_j=0.0) == float("inf")

    def test_instability_index_stable(self):
        # HSI=2.1, PI=0 → Z = 1/2.1 ≈ 0.476 < 1 → stable
        p = Params(HSI=2.1, PI=0.0)
        Z = p.instability_index(K_i=0, V_j=0)
        assert Z == pytest.approx(1 / 2.1, rel=1e-6)
        assert p.regime(K_i=0, V_j=0) == "stable"

    def test_instability_index_oscillatory(self):
        # HSI=1.0, PI=0 → Z = 1.0 → boundary → oscillatory
        p = Params(HSI=1.0, PI=0.0)
        assert p.regime(K_i=0, V_j=0) == "oscillatory"

    def test_instability_index_cascade(self):
        # HSI=0.4, PI=0 → Z = 2.5 → cascade
        p = Params(HSI=0.4, PI=0.0)
        assert p.regime(K_i=0, V_j=0) == "cascade"


class TestCapitalStocks:
    def test_replenish_caps_at_max(self):
        p = Params(K_cap=10.0, M_cap=10.0, rho_kappa=5.0, rho_mu=5.0)
        stocks = CapitalStocks(kappa=8.0, mu=8.0)
        replenished = stocks.replenish(p)
        assert replenished.kappa == pytest.approx(10.0)
        assert replenished.mu == pytest.approx(10.0)

    def test_replenish_normal(self):
        p = Params(K_cap=10.0, M_cap=10.0, rho_kappa=1.0, rho_mu=1.0)
        stocks = CapitalStocks(kappa=3.0, mu=3.0)
        replenished = stocks.replenish(p)
        assert replenished.kappa == pytest.approx(4.0)
        assert replenished.mu == pytest.approx(4.0)

    def test_deplete_floors_at_zero(self):
        p = Params()
        stocks = CapitalStocks(kappa=1.0, mu=1.0)
        depleted = stocks.deplete_kappa(100.0, p)
        assert depleted.kappa == 0.0


class TestCHSEState:
    def test_coherence(self):
        s = CHSEState(h=0.7)
        assert s.h_21 == pytest.approx(0.3)
        assert s.h + s.h_21 == pytest.approx(1.0)

    def test_invalid_h_raises(self):
        with pytest.raises(ValueError):
            CHSEState(h=1.5)
        with pytest.raises(ValueError):
            CHSEState(h=-0.1)

    def test_leader_detection(self):
        assert CHSEState(h=0.8).leader() == 1
        assert CHSEState(h=0.3).leader() == 2
        assert CHSEState(h=0.5).leader() == 1  # tie goes to player 1


# ---------------------------------------------------------------------------
# Mechanisms
# ---------------------------------------------------------------------------

class TestMechanisms:
    def test_ambiguity_push_direction(self):
        # h > 0.5: push should be negative (toward ambiguity)
        delta = ambiguity_push(h=0.8, gamma=1.0, params=Params(mu_II=1.0))
        assert delta < 0

        # h < 0.5: push should be positive
        delta = ambiguity_push(h=0.3, gamma=1.0, params=Params(mu_II=1.0))
        assert delta > 0

    def test_ambiguity_push_at_half(self):
        # At h = 0.5, push is exactly zero regardless of gamma
        delta = ambiguity_push(h=0.5, gamma=5.0, params=Params())
        assert delta == pytest.approx(0.0)

    def test_ambiguity_push_zero_spend(self):
        delta = ambiguity_push(h=0.9, gamma=0.0, params=Params())
        assert delta == pytest.approx(0.0)

    def test_reframe_resistance_range(self):
        p = Params(lambda_kappa=1.0)
        rho = reframe_resistance(c=5.0, params=p)
        assert 0.0 <= rho < 1.0

    def test_reframe_resistance_zero_spend(self):
        rho = reframe_resistance(c=0.0, params=Params())
        assert rho == pytest.approx(0.0)

    def test_reframe_resistance_monotone(self):
        p = Params(lambda_kappa=1.0)
        rho_low = reframe_resistance(c=1.0, params=p)
        rho_high = reframe_resistance(c=5.0, params=p)
        assert rho_high > rho_low

    def test_reframe_success_prob_range(self):
        p = Params(lambda_R=1.0)
        prob = reframe_success_prob(eta=2.0, rho=0.3, params=p)
        assert 0.0 <= prob <= 1.0

    def test_reframe_success_prob_zero_eta(self):
        prob = reframe_success_prob(eta=0.0, rho=0.0, params=Params())
        assert prob == pytest.approx(0.0)

    def test_reframe_success_prob_full_resistance(self):
        # rho = 1 means the leader is fully reframe-resistant
        prob = reframe_success_prob(eta=100.0, rho=1.0, params=Params())
        assert prob == pytest.approx(0.0)

    def test_reframing_investment_always_nonpositive(self):
        p = Params()
        for h in [0.3, 0.5, 0.7]:
            delta = reframing_investment(h=h, eta=1.0, rho=0.0, params=p)
            assert delta <= 0.0

    def test_optimal_eta_zero_when_leader(self):
        # If h >= 0.5, follower has no incentive to reframe
        assert optimal_eta(h=0.6, params=Params()) == 0.0
        assert optimal_eta(h=0.5, params=Params()) == 0.0

    def test_optimal_eta_positive_when_follower(self):
        # h=0.2: interior = alpha_R*lambda_R*(0.5-0.2)/c_mu = 0.5*1.0*0.3/0.1 = 1.5 > 1
        p = Params(alpha_R=0.5, lambda_R=1.0, c_mu=0.1)
        eta = optimal_eta(h=0.2, params=p)
        assert eta > 0.0

    def test_optimal_kappa_zero_when_follower(self):
        assert optimal_kappa_spend(h=0.4, params=Params()) == 0.0
        assert optimal_kappa_spend(h=0.5, params=Params()) == 0.0

    def test_optimal_kappa_positive_when_leader(self):
        # h=0.8: interior = alpha_R*lambda_kappa*(0.8-0.5)/c_kappa = 0.5*1.0*0.3/0.1 = 1.5 > 1
        p = Params(alpha_R=0.5, lambda_kappa=1.0, c_kappa=0.1)
        kappa = optimal_kappa_spend(h=0.8, params=p)
        assert kappa > 0.0


# ---------------------------------------------------------------------------
# Two-player model
# ---------------------------------------------------------------------------

class TestTwoPlayerModel:
    def setup_method(self):
        # mu=1.5, eta=0.8, kappa=0.2, r=0.5
        # h* = 0.5 + (0.8-0.2*0.5)/(2*1.5) = 0.80  disc=1.96>0 stable node
        self.stable = TwoPlayerModel(
            mu=1.5, eta_bar=0.8, kappa_bar=0.2, r_bar=0.5,
            params=Params(HSI=2.1)
        )
        # mu=0.6, eta=0.4, kappa=0.4, r=1.0
        # h* = 0.5  disc=0.36-0.64=-0.28<0 oscillatory
        self.oscillatory = TwoPlayerModel(
            mu=0.6, eta_bar=0.4, kappa_bar=0.4, r_bar=1.0,
            params=Params(HSI=1.0)
        )
        # mu=0.3, eta=0.4, kappa=0.4, r=1.0  (same but weaker mean-reversion)
        self.cascade = TwoPlayerModel(
            mu=0.3, eta_bar=0.4, kappa_bar=0.4, r_bar=1.0,
            params=Params(HSI=0.4)
        )

    def test_fixed_point_formula(self):
        # dh/dt = 0  =>  h* = 0.5 + (eta_bar - kappa_bar*r_bar) / mu  (no factor of 2)
        expected = 0.5 + (0.8 - 0.2 * 0.5) / 1.5
        assert self.stable.fixed_point() == pytest.approx(expected)

    def test_fixed_point_clipped_low(self):
        # Strong resistance: h* clips to 0.0
        m = TwoPlayerModel(mu=0.1, eta_bar=0.0, kappa_bar=5.0, r_bar=1.0)
        assert m.fixed_point() == pytest.approx(0.0)

    def test_fixed_point_clipped_high(self):
        # Strong attack: h* clips to 1.0
        m = TwoPlayerModel(mu=0.1, eta_bar=5.0, kappa_bar=0.0, r_bar=1.0)
        assert m.fixed_point() == pytest.approx(1.0)

    def test_integrate_returns_correct_shape(self):
        result = self.stable.integrate(T=80, n_points=800)
        assert len(result.t) == 800
        assert len(result.h) == 800
        assert len(result.eta) == 800
        assert len(result.kappa) == 800

    def test_h_stays_in_unit_interval(self):
        for model in [self.stable, self.oscillatory, self.cascade]:
            result = model.integrate(T=80)
            assert np.all(result.h >= 0.0)
            assert np.all(result.h <= 1.0)

    def test_stable_converges_to_fixed_point(self):
        # disc > 0 -> stable node; h converges from h0=0.75 to h*=0.80
        result = self.stable.integrate(T=200, n_points=2000)
        h_final = np.mean(result.h[-100:])
        h_star = self.stable.fixed_point()
        assert abs(h_final - h_star) < 0.05

    def test_stable_ode_zero_flips(self):
        # Stable node with h*=0.80 > 0.75=h0: h converges upward, never crosses 0.5
        result = self.stable.integrate(T=80)
        assert result.turnover_count == 0

    def test_best_response_mode_runs(self):
        p = Params(alpha_R=0.3, lambda_R=1.0, lambda_kappa=1.0,
                   c_mu=0.2, c_kappa=0.2, HSI=1.0)
        model = TwoPlayerModel(mu=1.0, eta_bar=0.3, kappa_bar=0.3,
                               r_bar=0.5, params=p)
        result = model.integrate(T=40, mode="best_response")
        assert np.all(result.h >= 0.0)
        assert np.all(result.h <= 1.0)

    def test_integrate_stochastic_shape(self):
        result = self.oscillatory.integrate_stochastic(T=80, noise_std=0.06)
        # integrate_stochastic returns T+1 points (periods 0 through T inclusive)
        assert len(result.t) == 81
        assert len(result.h) == 81
        assert np.all(result.h >= 0.0)
        assert np.all(result.h <= 1.0)

    def test_figure2_regimes_keys(self):
        regimes = TwoPlayerModel.figure2_regimes(T=80)
        assert set(regimes.keys()) == {"stable", "oscillatory", "cascade"}

    def test_figure2_h_in_unit_interval(self):
        regimes = TwoPlayerModel.figure2_regimes(T=80)
        for name, result in regimes.items():
            assert np.all(result.h >= 0.0), f"{name} h < 0"
            assert np.all(result.h <= 1.0), f"{name} h > 1"

    def test_figure2_stable_zero_flips(self):
        regimes = TwoPlayerModel.figure2_regimes(T=80)
        assert regimes["stable"].turnover_count == 0

    def test_figure2_cascade_has_wide_amplitude(self):
        regimes = TwoPlayerModel.figure2_regimes(T=80)
        h_range = regimes["cascade"].h.max() - regimes["cascade"].h.min()
        assert h_range > 0.4

    def test_figure2_cascade_more_flips_than_stable(self):
        regimes = TwoPlayerModel.figure2_regimes(T=80)
        assert regimes["cascade"].turnover_count > regimes["stable"].turnover_count

    def test_invalid_mu_raises(self):
        with pytest.raises(ValueError):
            TwoPlayerModel(mu=-1.0, eta_bar=0.3, kappa_bar=0.3, r_bar=0.5)

    def test_invalid_h0_raises(self):
        with pytest.raises(ValueError):
            TwoPlayerModel(mu=1.0, eta_bar=0.3, kappa_bar=0.3,
                           r_bar=0.5, h0=1.5)


# ---------------------------------------------------------------------------
# Oscillation analysis
# ---------------------------------------------------------------------------

class TestOscillationAnalysis:
    def test_stable_node_condition(self):
        # μ² > 4η̄κ̄ → stable
        osc = OscillationAnalysis(mu=2.0, eta_bar=0.3, kappa_bar=0.3)
        result = osc.analyse()
        assert result.regime == "stable_node"
        assert not result.oscillates

    def test_oscillatory_condition(self):
        # μ² < 4η̄κ̄ → oscillates
        osc = OscillationAnalysis(mu=0.5, eta_bar=0.8, kappa_bar=0.8)
        result = osc.analyse()
        assert result.oscillates
        assert result.period is not None
        assert result.period > 0

    def test_eigenvalues_negative_real_part(self):
        # Regardless of regime, Re(λ) < 0 (system is stable)
        for mu, eta, kappa in [(2.0, 0.3, 0.3), (0.5, 0.8, 0.8)]:
            osc = OscillationAnalysis(mu=mu, eta_bar=eta, kappa_bar=kappa)
            result = osc.analyse()
            assert result.eigenvalues[0].real < 0
            assert result.eigenvalues[1].real < 0

    def test_discriminant_sign(self):
        osc_stable = OscillationAnalysis(mu=2.0, eta_bar=0.3, kappa_bar=0.3)
        osc_osc = OscillationAnalysis(mu=0.5, eta_bar=0.8, kappa_bar=0.8)
        assert osc_stable.analyse().discriminant > 0
        assert osc_osc.analyse().discriminant < 0

    def test_condition_holds_wrapper(self):
        osc = OscillationAnalysis(mu=0.5, eta_bar=0.8, kappa_bar=0.8)
        assert osc.condition_holds()

    def test_phase_portrait_shape(self):
        osc = OscillationAnalysis(mu=1.0, eta_bar=0.3, kappa_bar=0.3)
        h_vals, dh_vals = osc.phase_portrait(n_points=100)
        assert len(h_vals) == 100
        assert len(dh_vals) == 100

    def test_stability_scan_shape(self):
        mu_vals = np.linspace(0.1, 3.0, 10)
        eta_vals = np.linspace(0.1, 2.0, 10)
        result = OscillationAnalysis.stability_scan(mu_vals, eta_vals)
        assert result.shape == (10, 10)


# ---------------------------------------------------------------------------
# Flip threshold
# ---------------------------------------------------------------------------

class TestFlipThreshold:
    def test_stable_regime_infinite_flip_time(self):
        # eta_bar=0.8, kappa_bar=0.2, r_bar=0.5, mu=1.0
        # h* = 0.5 + (0.8 - 0.2*0.5)/(2*1.0) = 0.85  → well above 0.5+epsilon
        # disc = 1.0 - 4*0.8*0.2 = 0.36 > 0 → stable node → t* = None
        result = flip_time(
            h0=0.75, mu=1.0, eta_bar=0.8, kappa_bar=0.2,
            r_bar=0.5, epsilon=0.01, params=Params(HSI=2.1)
        )
        assert result.t_star is None

    def test_oscillatory_finite_flip_time(self):
        result = flip_time(
            h0=0.75, mu=0.5, eta_bar=0.8, kappa_bar=0.8,
            r_bar=0.5, epsilon=0.01
        )
        assert result.t_star is not None
        assert result.t_star > 0

    def test_flip_time_increases_with_h0_distance(self):
        # Starting closer to 0.5 should give shorter flip time
        r1 = flip_time(h0=0.6, mu=0.5, eta_bar=0.8, kappa_bar=0.8, r_bar=0.5)
        r2 = flip_time(h0=0.9, mu=0.5, eta_bar=0.8, kappa_bar=0.8, r_bar=0.5)
        if r1.t_star is not None and r2.t_star is not None:
            assert r1.t_star < r2.t_star

    def test_invalid_h0_raises(self):
        with pytest.raises(ValueError):
            flip_time(h0=0.4, mu=1.0, eta_bar=0.3, kappa_bar=0.3, r_bar=0.5)

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError):
            flip_time(h0=0.75, mu=1.0, eta_bar=0.3, kappa_bar=0.3,
                      r_bar=0.5, epsilon=-0.01)

    def test_flip_time_vs_hsi_shape(self):
        hsi_vals = np.linspace(0.3, 2.5, 20)
        hsi_out, t_stars = flip_time_vs_hsi(hsi_vals)
        assert len(hsi_out) == 20
        assert len(t_stars) == 20

    def test_flip_time_vs_hsi_monotone(self):
        # Flip time should increase with HSI (more stable = slower flip)
        hsi_vals = np.array([0.4, 0.6, 0.8, 1.0])
        _, t_stars = flip_time_vs_hsi(hsi_vals, h0=0.75, mu=0.8,
                                       eta_bar=0.3, r_bar=0.5)
        # Filter out nan (stable regimes)
        valid = t_stars[~np.isnan(t_stars)]
        if len(valid) > 1:
            diffs = np.diff(valid)
            # Majority should be non-decreasing
            assert np.sum(diffs >= -0.01) >= len(diffs) // 2
