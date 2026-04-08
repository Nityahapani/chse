"""
test_phase3.py
==============
Tests for Phase 3: HOE estimation, welfare distortions, paradox, empirical pipeline.

Each test is tied to a specific claim in the paper.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from chse.core.primitives import Params
from chse.core.network import CHSENetwork
from chse.equilibrium.markov import run_chain, MarkovState, CHSETransition
from chse.equilibrium.hoe import (
    HOEEstimator, HOEStatistics, stationarity_test,
    check_ergodicity_conditions,
)
from chse.equilibrium.lyapunov import (
    verify_lyapunov, lyapunov_V, estimate_orbit_support, LyapunovResult,
)
from chse.welfare.distortions import (
    compute_welfare_distortions, total_welfare, reframing_distortion,
    resistance_distortion, clarity_distortion, total_clarity_gap,
    social_optimal_eta, social_optimal_kappa,
)
from chse.welfare.paradox import calibrated_paradox_scan, ParadoxResult
from chse.empirical.fdi import (
    FDIEstimate, build_paper_examples, hoe_statistics_from_series,
    predict_regime, persistence_paradox_test,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_net(n=2, h0=0.7):
    if n == 2:
        return CHSENetwork.two_player(h0=h0)
    return CHSENetwork.complete(n, initial_h=h0)

def make_params(**kwargs):
    defaults = dict(
        alpha_R=0.4, alpha_I=0.1, beta_I=0.05,
        lambda_R=1.0, lambda_kappa=1.0, lambda_sigma=0.5,
        mu_II=0.2, rho_kappa=0.3, rho_mu=0.3,
        K_cap=10.0, M_cap=10.0, delta_kappa=0.2,
        c_mu=0.3, c_kappa=0.3, beta_R=0.1, zeta_II=0.3,
    )
    defaults.update(kwargs)
    return Params(**defaults)


# ============================================================================
# Markov chain
# ============================================================================

class TestMarkovChain:

    def test_run_chain_shape(self):
        net = make_net(2)
        p = make_params()
        r = run_chain(net, p, T=20, seed=42)
        assert r.T == 20
        assert r.h_trajectory.shape == (21, 1)
        assert r.kappa_traj.shape == (21, 2)
        assert r.mu_traj.shape == (21, 2)

    def test_h_stays_in_unit_interval(self):
        net = make_net(3)
        p = make_params()
        r = run_chain(net, p, T=50, seed=42)
        assert np.all(r.h_trajectory >= 0.0)
        assert np.all(r.h_trajectory <= 1.0)

    def test_kappa_bounded(self):
        net = make_net(2)
        p = make_params(K_cap=10.0)
        r = run_chain(net, p, T=50, seed=42)
        assert np.all(r.kappa_traj >= 0.0)
        assert np.all(r.kappa_traj <= p.K_cap + 1e-6)

    def test_mu_bounded(self):
        net = make_net(2)
        p = make_params(M_cap=10.0)
        r = run_chain(net, p, T=50, seed=42)
        assert np.all(r.mu_traj >= 0.0)
        assert np.all(r.mu_traj <= p.M_cap + 1e-6)

    def test_turnover_count_nonnegative(self):
        net = make_net(2)
        p = make_params()
        r = run_chain(net, p, T=50, seed=42)
        assert r.turnover_count() >= 0

    def test_turnover_frequency_in_range(self):
        net = make_net(2)
        p = make_params()
        r = run_chain(net, p, T=50, seed=42)
        assert 0.0 <= r.turnover_frequency() <= 1.0

    def test_state_to_vector_shape(self):
        net = make_net(2)
        p = make_params()
        r = run_chain(net, p, T=10, seed=42)
        vec = r.states[0].to_vector()
        # 1 edge + 2 kappa + 2 mu = 5
        assert len(vec) == 5

    def test_markov_state_copy(self):
        s = MarkovState(h=np.array([0.7]), kappa=np.array([5.0, 5.0]),
                        mu=np.array([5.0, 5.0]))
        s2 = s.copy()
        s2.h[0] = 0.3
        assert s.h[0] == 0.7   # original unchanged

    def test_different_seeds_give_different_paths(self):
        net = make_net(2)
        p = make_params()
        r1 = run_chain(net, p, T=30, seed=1)
        r2 = run_chain(net, p, T=30, seed=2)
        # Paths should differ (different noise)
        assert not np.allclose(r1.h_trajectory, r2.h_trajectory)

    def test_chain_with_3_player_network(self):
        net = make_net(3)
        p = make_params()
        r = run_chain(net, p, T=30, seed=42)
        assert r.n_edges == 3
        assert r.h_trajectory.shape == (31, 3)


# ============================================================================
# HOE estimation
# ============================================================================

class TestHOEEstimation:

    def test_ergodicity_conditions_satisfied(self):
        net = make_net(2)
        p = make_params()
        result = check_ergodicity_conditions(net, p)
        assert result['irreducible'] is True

    def test_ergodicity_irreducibility_checks(self):
        net = make_net(2)
        p = make_params()
        result = check_ergodicity_conditions(net, p)
        checks = result['irreducibility_checks']
        assert checks['lambda_R > 0']
        assert checks['lambda_kappa > 0']
        assert checks['rho_kappa > 0']
        assert checks['rho_mu > 0']
        assert checks['alpha_R > 0']
        assert checks['network connected']

    def test_ergodicity_fails_zero_replenishment(self):
        net = make_net(2)
        p = make_params(rho_kappa=0.0)
        result = check_ergodicity_conditions(net, p)
        assert result['irreducible'] is False

    def test_stationarity_test_structure(self):
        net = make_net(2)
        p = make_params()
        chain = run_chain(net, p, T=60, seed=42)
        st = stationarity_test(chain, burn_in=20)
        assert 'window_means' in st
        assert 'max_mean_diff' in st
        assert 'converged' in st
        assert len(st['window_means']) == 4

    def test_stationarity_max_diff_nonnegative(self):
        net = make_net(2)
        p = make_params()
        chain = run_chain(net, p, T=60, seed=42)
        st = stationarity_test(chain, burn_in=20)
        assert st['max_mean_diff'] >= 0.0

    def test_hoe_estimator_runs(self):
        net = make_net(2)
        p = make_params()
        estimator = HOEEstimator(net, p, T=80, burn_in=20, n_chains=2)
        chains, stats = estimator.run()
        assert len(chains) == 2
        assert isinstance(stats, HOEStatistics)
        assert stats.n_chains == 2

    def test_hoe_statistics_ranges(self):
        net = make_net(2)
        p = make_params()
        estimator = HOEEstimator(net, p, T=80, burn_in=20, n_chains=2)
        _, stats = estimator.run()
        assert 0.0 <= stats.tau_hat <= 1.0
        assert stats.var_h >= 0.0
        assert 0.0 <= stats.mean_h <= 1.0
        assert stats.expected_cascade >= 0.0

    def test_hoe_summary_string(self):
        net = make_net(2)
        p = make_params()
        estimator = HOEEstimator(net, p, T=60, burn_in=10, n_chains=2)
        _, stats = estimator.run()
        summary = stats.summary()
        assert 'tau_hat' in summary
        assert 'Var(h)' in summary


# ============================================================================
# Lyapunov stability
# ============================================================================

class TestLyapunov:

    def test_lyapunov_V_nonnegative(self):
        net = make_net(2)
        p = make_params()
        chain = run_chain(net, p, T=60, seed=42)
        support = estimate_orbit_support(chain, burn_in=20, n_support_points=20)
        V = lyapunov_V(chain.states[40], support, p)
        assert V >= 0.0

    def test_orbit_support_shape(self):
        net = make_net(2)
        p = make_params()
        chain = run_chain(net, p, T=80, seed=42)
        support = estimate_orbit_support(chain, burn_in=20, n_support_points=30)
        assert support.shape[0] == 30
        assert support.shape[1] == chain.states[0].dim

    def test_lyapunov_verify_returns_result(self):
        net = make_net(2)
        p = make_params()
        chain = run_chain(net, p, T=80, seed=42)
        result = verify_lyapunov(chain, p, burn_in=20, n_support_points=20)
        assert isinstance(result, LyapunovResult)
        assert len(result.V_trajectory) > 0
        assert len(result.delta_V) == len(result.V_trajectory) - 1

    def test_lyapunov_stability_condition(self):
        # Gamma=0.3, delta=0.95: (1-0.95)/(1+0.95) = 0.05/1.95 ≈ 0.026
        # 0.3 > 0.026 → not Lyapunov stable by theorem
        net = make_net(2)
        p = make_params(discount=0.95)
        chain = run_chain(net, p, T=80, seed=42)
        result = verify_lyapunov(chain, p, burn_in=20, Gamma=0.3)
        # Check the bound is computed correctly
        assert abs(result.stability_bound - (1 - 0.95) / (1 + 0.95)) < 1e-6

    def test_lyapunov_stable_when_gamma_small(self):
        # Very small Gamma → should be stable by theorem
        net = make_net(2)
        p = make_params(discount=0.5)
        chain = run_chain(net, p, T=80, seed=42)
        # Bound = (1-0.5)/(1+0.5) = 0.333
        result = verify_lyapunov(chain, p, burn_in=20, Gamma=0.1)
        assert result.lyapunov_stable is True

    def test_frac_decreasing_in_range(self):
        net = make_net(2)
        p = make_params()
        chain = run_chain(net, p, T=80, seed=42)
        result = verify_lyapunov(chain, p, burn_in=20)
        assert 0.0 <= result.frac_decreasing <= 1.0


# ============================================================================
# Welfare distortions
# ============================================================================

class TestWelfareDistortions:

    def test_total_welfare_positive(self):
        net = make_net(3, h0=0.7)
        W = total_welfare(net, u_L=10.0, u_F=2.0)
        assert W > 0.0

    def test_total_welfare_depends_on_payoffs(self):
        # Welfare W = Σ_{edges} [h*uL + (1-h)*uF + (1-h)*uL + h*uF]
        #           = Σ_{edges} [uL + uF]  — symmetric, independent of h
        # But costs break the symmetry: lower investment = higher net welfare
        net = make_net(3, h0=0.7)
        low_cost  = np.zeros(3)
        high_cost = np.ones(3) * 2.0
        W_low  = total_welfare(net, u_L=10.0, u_F=2.0, investment_costs=low_cost)
        W_high = total_welfare(net, u_L=10.0, u_F=2.0, investment_costs=high_cost)
        assert W_low > W_high

    def test_reframing_distortion_positive(self):
        net = make_net(3)
        p = make_params(beta_R=0.2)
        excess = reframing_distortion(net, p, eta_eq=0.5, Gamma=0.4)
        assert excess > 0.0

    def test_reframing_distortion_zero_gamma(self):
        net = make_net(3)
        p = make_params(beta_R=0.2)
        excess = reframing_distortion(net, p, eta_eq=0.5, Gamma=0.0)
        assert excess == 0.0

    def test_social_optimal_eta_less_than_eq(self):
        net = make_net(3)
        p = make_params(beta_R=0.2)
        eta_eq = 0.5
        eta_so = social_optimal_eta(net, p, eta_eq=eta_eq, Gamma=0.4)
        # Social planner chooses less reframing (corrects overinvestment)
        assert eta_so < eta_eq

    def test_resistance_distortion_positive(self):
        net = make_net(3)
        p = make_params()
        excess = resistance_distortion(net, p, kappa_eq=0.6)
        assert excess > 0.0

    def test_clarity_distortion_per_player(self):
        net = make_net(3)
        p = make_params(zeta_II=0.3)
        per_player = clarity_distortion(net, p)
        assert len(per_player) == 3
        # All players have positive under-investment
        for v in per_player.values():
            assert v >= 0.0

    def test_clarity_gap_increases_with_ambiguity(self):
        # h=0.5 (maximum ambiguity) should have larger clarity gap
        net_ambig = CHSENetwork.complete(3, initial_h=0.5)
        net_clear = CHSENetwork.complete(3, initial_h=0.9)
        p = make_params(zeta_II=0.3)
        gap_ambig = total_clarity_gap(net_ambig, p)
        gap_clear = total_clarity_gap(net_clear, p)
        assert gap_ambig > gap_clear

    def test_compute_welfare_distortions_runs(self):
        net = make_net(3, h0=0.65)
        p = make_params()
        wd = compute_welfare_distortions(net, p, eta_eq=0.5, kappa_eq=0.6, Gamma=0.4)
        assert wd.reframing_excess >= 0.0
        assert wd.resistance_excess >= 0.0
        assert wd.clarity_gap >= 0.0
        assert len(wd.policy_implications) == 3

    def test_welfare_distortions_summary(self):
        net = make_net(3, h0=0.65)
        p = make_params()
        wd = compute_welfare_distortions(net, p, eta_eq=0.5, kappa_eq=0.6, Gamma=0.4)
        summary = wd.summary()
        assert 'Distortion 1' in summary
        assert 'Distortion 2' in summary
        assert 'Distortion 3' in summary


# ============================================================================
# Hierarchy Persistence Paradox
# ============================================================================

class TestParadox:

    def test_calibrated_scan_runs(self):
        result = calibrated_paradox_scan()
        assert isinstance(result, ParadoxResult)
        assert len(result.hsi_vals) > 0

    def test_acc_increases_with_hsi(self):
        result = calibrated_paradox_scan()
        # Acc_ij should be monotonically increasing in HSI
        diffs = np.diff(result.acc_vals)
        assert np.all(diffs >= 0)

    def test_rho_K_increases_with_hsi(self):
        result = calibrated_paradox_scan()
        diffs = np.diff(result.rho_K_vals)
        assert np.all(diffs >= 0)

    def test_cascade_sizes_increase_with_hsi(self):
        # This is the core paradox claim
        result = calibrated_paradox_scan()
        assert result.derivative_sign is True

    def test_rho_K_in_realistic_range(self):
        # With calibrated params, rho_K should stay well below 1
        result = calibrated_paradox_scan()
        assert result.rho_K_vals.max() < 1.0
        # But should be meaningfully above 0 for high HSI
        assert result.rho_K_vals.max() > 0.1

    def test_cascade_sizes_finite(self):
        result = calibrated_paradox_scan()
        # All cascade sizes should be finite (rho_K < 1)
        assert result.n_below_one == len(result.hsi_vals)

    def test_paradox_summary(self):
        result = calibrated_paradox_scan()
        summary = result.summary()
        assert 'Paradox' in summary
        assert 'd(E[cascade])' in summary

    def test_custom_hsi_range(self):
        hsi_vals = np.array([0.5, 1.0, 1.5, 2.0])
        result = calibrated_paradox_scan(hsi_vals=hsi_vals)
        assert len(result.hsi_vals) == 4
        assert len(result.cascade_sizes) == 4


# ============================================================================
# Empirical pipeline
# ============================================================================

class TestEmpiricalPipeline:

    def test_fdi_formula(self):
        # FDI = V_T * lambda_R / (K_CB * rho_ratio)
        est = FDIEstimate(country='Test', period='2020', V_T=1.0,
                         K_CB=1.0, lambda_R=1.0, rho_ratio=1.0)
        assert abs(est.FDI - 1.0) < 1e-10

    def test_fdi_regime_monetary(self):
        est = FDIEstimate(country='Test', period='2020', V_T=0.2,
                         K_CB=0.9, lambda_R=1.0, rho_ratio=1.0)
        assert est.regime == 'monetary'

    def test_fdi_regime_contested(self):
        est = FDIEstimate(country='Test', period='2020', V_T=0.7,
                         K_CB=0.9, lambda_R=1.0, rho_ratio=1.0)
        assert est.regime == 'contested'

    def test_fdi_regime_fiscal(self):
        est = FDIEstimate(country='Test', period='2020', V_T=1.5,
                         K_CB=0.8, lambda_R=1.0, rho_ratio=1.0)
        assert est.regime == 'fiscal'

    def test_paper_examples_count(self):
        examples = build_paper_examples()
        assert len(examples) == 6

    def test_paper_examples_fdi_ordering(self):
        examples = build_paper_examples()
        fdis = {e.country + '_' + e.period: e.FDI for e in examples}
        # Chile and US 2000-07 should be monetary (FDI < 0.5)
        assert fdis['Chile_2000-22'] < 0.5
        assert fdis['US_2000-07'] < 0.5
        # Turkey should be fiscal (FDI > 1)
        assert fdis['Turkey_2021-23'] > 1.0
        # Zambia should be fiscal
        assert fdis['Zambia_2020-23'] > 1.0

    def test_fdi_to_hsi(self):
        est = FDIEstimate(country='Test', period='2020', V_T=0.5,
                         K_CB=1.0, lambda_R=1.0, rho_ratio=1.0)
        assert abs(est.to_hsi() - 2.0) < 1e-6  # HSI = 1/0.5 = 2.0

    def test_hoe_statistics_from_series_ranges(self):
        rng = np.random.default_rng(42)
        h_series = rng.uniform(0.3, 0.8, 200)
        stats = hoe_statistics_from_series(h_series)
        assert 0.0 <= stats.tau_hat <= 1.0
        assert stats.var_h >= 0.0
        assert 0.0 <= stats.mean_h <= 1.0
        assert 0.0 <= stats.h_above_half <= 1.0
        assert stats.n_obs == 200

    def test_hoe_statistics_strictly_above_half(self):
        # Series always above 0.5 → zero crossings → tau_hat = 0
        h_series = np.linspace(0.6, 1.0, 100)
        stats = hoe_statistics_from_series(h_series)
        assert stats.tau_hat == 0.0

    def test_predict_regime_monetary(self):
        est = FDIEstimate(country='Test', period='2020', V_T=0.2,
                         K_CB=0.9, lambda_R=1.0, rho_ratio=1.0)
        pred = predict_regime(est, pi=0.0)
        assert pred['predicted_regime'] == 'monetary'

    def test_predict_regime_consistency(self):
        examples = build_paper_examples()
        for ex in examples:
            pred = predict_regime(ex, pi=0.0)
            # With FDI-based thresholds, predicted_regime == fdi_regime always
            assert pred['consistent'], f"{ex.country}: predicted={pred['predicted_regime']} fdi={pred['fdi_regime']}"

    def test_persistence_paradox_test_positive_corr(self):
        examples = build_paper_examples()
        result = persistence_paradox_test(examples)
        assert result['correlation'] > 0
        assert result['paradox_confirmed'] is True

    def test_persistence_paradox_test_structure(self):
        examples = build_paper_examples()
        result = persistence_paradox_test(examples)
        assert 'hsi_vals' in result
        assert 'volatility' in result
        assert 'correlation' in result
        assert 'interpretation' in result
