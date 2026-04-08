"""
hoe.py
======
Hierarchy Orbit Equilibrium (HOE) estimation.

The HOE π* is an invariant measure of the induced Markov process:

    π*(B) = ∫_S P(s, B) π*(ds)  for all B ∈ B(S)

Existence is guaranteed by Krylov-Bogolyubov (Theorem 8.1).
Uniqueness and ergodicity hold when the chain is irreducible and
aperiodic (Proposition on Ergodicity).

Estimation strategy (Section 10.2):
    Run the chain from many initial states, discard a burn-in period,
    and pool the remaining states.  The empirical distribution of
    (h, κ, μ) over this pooled sample approximates π*.

    The HOE is identified empirically as the stationary distribution
    over the triple (τ̂, Var(h), E[cascade size]) — Definition 10.2.

Stationarity test:
    Split the post-burn-in trajectory into two halves.  If the chain
    has converged, the mean and variance of h on both halves should be
    close (within tolerance).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..core.primitives import Params
from ..core.network import CHSENetwork
from ..core.kernel import expected_cascade_size, spectral_radius
from .markov import run_chain, ChainResult


# ---------------------------------------------------------------------------
# HOE statistics
# ---------------------------------------------------------------------------

@dataclass
class HOEStatistics:
    """
    The three empirical statistics that characterise the HOE (Definition 10.2).

    Attributes
    ----------
    tau_hat        : float  Leadership turnover frequency τ̂ (flips per period).
    var_h          : float  Variance of h in the stationary distribution.
    mean_h         : float  Mean of h in the stationary distribution.
    expected_cascade: float  E[cascade size] — estimated from chain dynamics.
    n_periods      : int    Number of periods used (post burn-in).
    n_chains       : int    Number of chains pooled.
    converged      : bool   Whether stationarity test passed.
    stationarity_gap: float Difference in means between first and second half.
    edge_idx       : int    Which edge these statistics are computed for.
    """
    tau_hat: float
    var_h: float
    mean_h: float
    expected_cascade: float
    n_periods: int
    n_chains: int
    converged: bool
    stationarity_gap: float
    edge_idx: int = 0

    def summary(self) -> str:
        conv = "YES" if self.converged else "NO (gap={:.4f})".format(self.stationarity_gap)
        return (
            f"HOE Statistics (edge {self.edge_idx})\n"
            f"  tau_hat (turnover/period) : {self.tau_hat:.4f}\n"
            f"  Var(h)                    : {self.var_h:.4f}\n"
            f"  E[h]                      : {self.mean_h:.4f}\n"
            f"  E[cascade size]           : {self.expected_cascade:.4f}\n"
            f"  Periods used              : {self.n_periods}\n"
            f"  Chains pooled             : {self.n_chains}\n"
            f"  Converged                 : {conv}"
        )


# ---------------------------------------------------------------------------
# HOE estimator
# ---------------------------------------------------------------------------

class HOEEstimator:
    """
    Monte Carlo estimator for the Hierarchy Orbit Equilibrium.

    Runs multiple chains from different initial states, discards burn-in,
    and pools the remaining samples to approximate π*.

    Parameters
    ----------
    network   : CHSENetwork
    params    : Params
    T         : int    Total periods per chain.
    burn_in   : int    Periods to discard at start of each chain.
    n_chains  : int    Number of chains to pool.
    """

    def __init__(
        self,
        network: CHSENetwork,
        params: Params,
        T: int = 400,
        burn_in: int = 100,
        n_chains: int = 4,
    ) -> None:
        self.network = network
        self.params = params
        self.T = T
        self.burn_in = burn_in
        self.n_chains = n_chains

    def _initial_states(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate diverse initial states spanning the state space.

        Returns list of (h0, kappa0, mu0) tuples.
        """
        n = self.network.n_players
        n_edges = len(self.network.canon_edges)
        p = self.params

        inits = []
        rng = np.random.default_rng(0)

        for chain_idx in range(self.n_chains):
            # Randomise initial h uniformly over [0.3, 0.9]
            h0 = rng.uniform(0.3, 0.9, size=n_edges)
            # Randomise capitals
            kappa0 = rng.uniform(0.2 * p.K_cap, 0.9 * p.K_cap, size=n)
            mu0 = rng.uniform(0.2 * p.M_cap, 0.9 * p.M_cap, size=n)
            inits.append((h0, kappa0, mu0))

        return inits

    def run(self) -> Tuple[List[ChainResult], HOEStatistics]:
        """
        Run all chains and estimate the HOE statistics.

        Returns
        -------
        chains : list[ChainResult]   One per chain.
        stats  : HOEStatistics       Pooled stationary statistics.
        """
        inits = self._initial_states()
        chains: List[ChainResult] = []

        for idx, (h0, kappa0, mu0) in enumerate(inits):
            net = self.network.copy()
            net.set_belief_vector(h0)
            result = run_chain(
                network=net,
                params=self.params,
                T=self.T,
                h0=h0,
                kappa0=kappa0,
                mu0=mu0,
                seed=42 + idx * 7,
            )
            chains.append(result)

        stats = self._compute_statistics(chains)
        return chains, stats

    def _compute_statistics(
        self,
        chains: List[ChainResult],
        edge_idx: int = 0,
    ) -> HOEStatistics:
        """Compute pooled HOE statistics from post-burn-in samples."""
        all_h: List[np.ndarray] = []
        total_flips = 0
        total_periods = 0

        for chain in chains:
            post = chain.h_trajectory[self.burn_in:, edge_idx]
            all_h.append(post)
            # Count flips
            flips = int(np.sum(np.abs(np.diff((post > 0.5).astype(int)))))
            total_flips += flips
            total_periods += len(post)

        pooled_h = np.concatenate(all_h)

        tau_hat = total_flips / max(total_periods, 1)
        var_h = float(np.var(pooled_h))
        mean_h = float(np.mean(pooled_h))

        # Cascade size: use rho_K from the last chain's final state
        # Simple approximation: use the mean absolute deviation of h as a
        # proxy for cascade potential
        h_dev = float(np.mean(np.abs(pooled_h - 0.5)))
        rho_K_proxy = min(0.8, h_dev * 1.2)
        exp_cascade = expected_cascade_size(rho_K_proxy, self.params.alpha_R)

        # Stationarity test: compare first vs second half of pooled h
        mid = len(pooled_h) // 2
        mean_first = float(np.mean(pooled_h[:mid]))
        mean_second = float(np.mean(pooled_h[mid:]))
        gap = abs(mean_first - mean_second)
        converged = gap < 0.05

        return HOEStatistics(
            tau_hat=tau_hat,
            var_h=var_h,
            mean_h=mean_h,
            expected_cascade=exp_cascade,
            n_periods=total_periods,
            n_chains=len(chains),
            converged=converged,
            stationarity_gap=gap,
            edge_idx=edge_idx,
        )


# ---------------------------------------------------------------------------
# Stationarity test
# ---------------------------------------------------------------------------

def stationarity_test(
    chain: ChainResult,
    burn_in: int = 100,
    edge_idx: int = 0,
    n_windows: int = 4,
) -> Dict:
    """
    Test whether the chain has converged to its stationary distribution.

    Splits the post-burn-in trajectory into n_windows windows and checks
    that the mean h is approximately constant across windows (consistent
    with stationarity).

    Parameters
    ----------
    chain    : ChainResult
    burn_in  : int    Periods to discard.
    edge_idx : int    Edge to test.
    n_windows: int    Number of windows to split into.

    Returns
    -------
    dict with keys:
        window_means   : list of mean h per window
        window_vars    : list of var h per window
        max_mean_diff  : max pairwise difference of window means
        converged      : bool — max_mean_diff < 0.05
    """
    post = chain.h_trajectory[burn_in:, edge_idx]
    n = len(post)
    window_size = n // n_windows

    means = []
    variances = []
    for w in range(n_windows):
        window = post[w * window_size: (w + 1) * window_size]
        means.append(float(np.mean(window)))
        variances.append(float(np.var(window)))

    max_diff = float(max(abs(means[i] - means[j])
                        for i in range(len(means))
                        for j in range(i + 1, len(means))))

    return {
        "window_means": means,
        "window_vars": variances,
        "max_mean_diff": max_diff,
        "converged": max_diff < 0.05,
    }


# ---------------------------------------------------------------------------
# Ergodicity check
# ---------------------------------------------------------------------------

def check_ergodicity_conditions(
    network: CHSENetwork,
    params: Params,
) -> Dict:
    """
    Check the sufficient conditions for ergodicity (Proposition on Ergodicity).

    Conditions:
      1. Irreducibility: all mechanism efficiencies > 0 and replenishment > 0.
      2. Aperiodicity: replenishment rates are incommensurate (rho_k/rho_mu irrational).

    Returns a dict with condition checks and overall verdict.
    """
    p = params

    irred_checks = {
        "lambda_R > 0": p.lambda_R > 0,
        "lambda_kappa > 0": p.lambda_kappa > 0,
        "lambda_sigma > 0": p.lambda_sigma > 0,
        "rho_kappa > 0": p.rho_kappa > 0,
        "rho_mu > 0": p.rho_mu > 0,
        "alpha_R > 0": p.alpha_R > 0,
        "network connected": _is_connected(network),
    }
    irreducible = all(irred_checks.values())

    # Aperiodicity: check rho_kappa / rho_mu is not a simple rational
    ratio = p.rho_kappa / p.rho_mu if p.rho_mu > 0 else 0.0
    # Sufficient condition: ratio is not close to p/q for small p, q
    aperiodic = _is_approximately_irrational(ratio)

    return {
        "irreducibility_checks": irred_checks,
        "irreducible": irreducible,
        "rho_ratio": ratio,
        "aperiodic": aperiodic,
        "ergodic": irreducible and aperiodic,
    }


def _is_connected(network: CHSENetwork) -> bool:
    """Check if the network graph is connected via BFS from node 0."""
    if network.n_players <= 1:
        return True
    visited = {0}
    queue = [0]
    while queue:
        node = queue.pop(0)
        for nb in network.neighbours(node):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return len(visited) == network.n_players


def _is_approximately_irrational(x: float, max_denom: int = 20) -> bool:
    """
    Return True if x is not close to any rational p/q with q ≤ max_denom.
    Sufficient condition for aperiodicity.
    """
    for q in range(1, max_denom + 1):
        p = round(x * q)
        if abs(x - p / q) < 1e-6:
            return False
    return True
