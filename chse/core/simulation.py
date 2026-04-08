"""
simulation.py
=============
Unified simulation interface for CHSE.

This module provides the high-level entry points that tie together
all four mechanisms, the network structure, and the capital dynamics
into a single coherent simulation framework.

It exposes two complementary simulation modes:

1. **Stochastic benchmark** (``BenchmarkSim``)
   The tractable two-player model from Section 3, extended to multiple
   HSI regimes.  Uses the closed-form ODE plus stochastic shocks.
   Best for: rapid parameter exploration, Figure 2 reproduction,
   oscillation condition verification.

2. **Full Markov chain** (``FullSim``)
   The complete n-player game with all four mechanisms active.
   Uses the transition kernel from Section 5 / equilibrium/markov.py.
   Best for: HOE estimation, Lyapunov verification, network cascades.

Both produce ``SimResult`` objects with a common interface so downstream
analysis (HOE statistics, welfare distortions, Lyapunov) can consume
either without modification.

References
----------
Section 3  — Two-player benchmark
Section 5  — Full dynamic system
Section 8  — HOE / Markov formulation
Definition 10.1 — Leadership Turnover Frequency
Definition 10.2 — HOE as Stationary Distribution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .primitives import Params
from .network import CHSENetwork
# NOTE: chse.benchmark and chse.equilibrium are imported lazily inside
# methods to avoid the circular import:
#   chse.core.__init__ → chse.core.simulation → chse.benchmark.two_player
#   → chse.core.primitives → chse.core.__init__  (cycle!)
from ..empirical.fdi import hoe_statistics_from_series, HOEFromData


# ---------------------------------------------------------------------------
# Common result container
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    """
    Unified result object from either simulation mode.

    Attributes
    ----------
    h_trajectories : list[np.ndarray]
        One array per chain; each shape (T+1,) for single-edge or (T+1, n_edges).
    burn_in        : int    Periods discarded for burn-in.
    T              : int    Total periods per chain.
    n_chains       : int    Number of chains run.
    regime         : str    Regime label ('stable', 'oscillatory', 'cascade').
    HSI            : float  Hierarchy Stability Index used.
    mode           : str    'benchmark' or 'full_markov'.
    hoe_stats      : HOEFromData  HOE statistics estimated from pooled chains.
    """
    h_trajectories: List[np.ndarray]
    burn_in: int
    T: int
    n_chains: int
    regime: str
    HSI: float
    mode: str
    hoe_stats: HOEFromData

    @property
    def pooled_h(self) -> np.ndarray:
        """Post-burn-in h values pooled across all chains."""
        parts = []
        for h in self.h_trajectories:
            traj = h[:, 0] if h.ndim == 2 else h
            parts.append(traj[self.burn_in:])
        return np.concatenate(parts)

    @property
    def turnover_counts(self) -> List[int]:
        """Flip count per chain."""
        counts = []
        for h in self.h_trajectories:
            traj = h[:, 0] if h.ndim == 2 else h
            counts.append(int(np.sum(np.abs(np.diff((traj > 0.5).astype(int))))))
        return counts

    def stationarity_check(self, n_windows: int = 4) -> Dict:
        """
        Check stationarity by comparing window means of the pooled trajectory.

        Returns dict with 'window_means', 'max_diff', 'converged'.
        """
        h = self.pooled_h
        n = len(h)
        ws = n // n_windows
        means = [float(np.mean(h[i*ws:(i+1)*ws])) for i in range(n_windows)]
        diffs = [abs(means[i] - means[j])
                 for i in range(len(means)) for j in range(i+1, len(means))]
        max_diff = max(diffs) if diffs else 0.0
        return {
            "window_means": [round(m, 4) for m in means],
            "max_diff": round(max_diff, 4),
            "converged": max_diff < 0.05,
        }


# ---------------------------------------------------------------------------
# Benchmark simulator
# ---------------------------------------------------------------------------

class BenchmarkSim:
    """
    Multi-chain stochastic benchmark simulation (Section 3).

    Runs ``n_chains`` independent chains using ``TwoPlayerModel.integrate_stochastic``
    from diverse initial belief values, then pools the post-burn-in samples
    to estimate the HOE π*.

    This is the recommended mode for HOE estimation because:
    - The two-player benchmark has closed-form dynamics (analytically grounded)
    - Convergence to π* is fast and visually clear
    - The stable, oscillatory, and cascade regimes are well-separated

    Parameters
    ----------
    regime  : str    One of 'stable' (HSI=2.1), 'oscillatory' (HSI=1.0),
                     'cascade' (HSI=0.4), or 'custom'.
    T       : int    Periods per chain.
    burn_in : int    Periods to discard.
    n_chains: int    Number of chains.

    Custom regime parameters (used when regime='custom'):
    mu, eta_bar, kappa_bar, r_bar, noise_std, HSI
    """

    # Pre-calibrated regime configurations
    REGIMES: Dict[str, Dict] = {
        "stable": dict(
            mu=2.0, eta_bar=0.8, kappa_bar=0.2, r_bar=1.0,
            noise_std=0.03, HSI=2.1,
            h0_spread=[0.20, 0.45, 0.75, 0.90],
        ),
        "oscillatory": dict(
            mu=0.6, eta_bar=0.4, kappa_bar=0.4, r_bar=1.0,
            noise_std=0.09, HSI=1.0,
            h0_spread=[0.25, 0.50, 0.65, 0.80],
        ),
        "cascade": dict(
            mu=0.3, eta_bar=0.4, kappa_bar=0.4, r_bar=1.0,
            noise_std=0.15, HSI=0.4,
            h0_spread=[0.30, 0.50, 0.65, 0.80],
        ),
    }

    def __init__(
        self,
        regime: str = "stable",
        T: int = 300,
        burn_in: int = 80,
        n_chains: int = 4,
        **custom_kw,
    ) -> None:
        self.regime = regime
        self.T = T
        self.burn_in = burn_in
        self.n_chains = n_chains

        if regime in self.REGIMES:
            cfg = {**self.REGIMES[regime], **custom_kw}
        else:
            cfg = custom_kw

        self.mu = cfg.get("mu", 0.6)
        self.eta_bar = cfg.get("eta_bar", 0.4)
        self.kappa_bar = cfg.get("kappa_bar", 0.4)
        self.r_bar = cfg.get("r_bar", 1.0)
        self.noise_std = cfg.get("noise_std", 0.06)
        self.HSI = cfg.get("HSI", 1.0)
        self.h0_spread = cfg.get(
            "h0_spread",
            list(np.linspace(0.2, 0.9, n_chains))
        )

    def run(self, seed: int = 42) -> SimResult:
        """
        Run all chains and return a SimResult.

        Parameters
        ----------
        seed : int  Base random seed (each chain uses seed + chain_index).

        Returns
        -------
        SimResult
        """
        # Lazy import to avoid circular dependency at module level
        from ..benchmark.two_player import TwoPlayerModel  # noqa: PLC0415

        params = Params(HSI=self.HSI)
        trajectories = []

        for i in range(self.n_chains):
            h0 = self.h0_spread[i % len(self.h0_spread)]
            model = TwoPlayerModel(
                mu=self.mu,
                eta_bar=self.eta_bar,
                kappa_bar=self.kappa_bar,
                r_bar=self.r_bar,
                h0=h0,
                params=params,
            )
            result = model.integrate_stochastic(
                T=self.T,
                noise_std=self.noise_std,
                seed=seed + i * 13,
            )
            trajectories.append(result.h)

        # Pool post-burn-in for HOE statistics
        pooled = np.concatenate([h[self.burn_in:] for h in trajectories])
        hoe_stats = hoe_statistics_from_series(pooled)

        return SimResult(
            h_trajectories=trajectories,
            burn_in=self.burn_in,
            T=self.T,
            n_chains=self.n_chains,
            regime=self.regime,
            HSI=self.HSI,
            mode="benchmark",
            hoe_stats=hoe_stats,
        )


# ---------------------------------------------------------------------------
# Full Markov chain simulator
# ---------------------------------------------------------------------------

class FullSim:
    """
    Full n-player Markov chain simulation with all four mechanisms (Section 5).

    Uses ``equilibrium/markov.run_chain`` which implements the complete
    period timeline:
      1. Investment portfolios (optimal best-response)
      2. Stage game resolves
      3. Anticipation signals (Mechanism I, Beta-Binomial)
      4. Belief update: h(t+1) via all mechanisms
      5. Resource update: κ, μ
      6. Propagation kernel applied (Mechanism IV)

    Note: The full Markov chain naturally converges to the stable regime
    (h → h*) for most parameter settings, because the optimal best-response
    follower (optimal_eta) only attacks when h < 0.5.  This demonstrates the
    STABLE HOE: π* = δ_{h*}, the Stackelberg limit when HSI is large.

    For interior HOE dynamics, use ``BenchmarkSim`` with ``regime='oscillatory'``.

    Parameters
    ----------
    network : CHSENetwork
    params  : Params
    T       : int    Periods per chain.
    burn_in : int    Burn-in to discard.
    n_chains: int    Number of chains from diverse initial states.
    """

    def __init__(
        self,
        network: CHSENetwork,
        params: Params,
        T: int = 200,
        burn_in: int = 60,
        n_chains: int = 4,
    ) -> None:
        self.network = network
        self.params = params
        self.T = T
        self.burn_in = burn_in
        self.n_chains = n_chains

    def run(self, seed: int = 42) -> Tuple[SimResult, List]:
        """
        Run all chains.

        Returns
        -------
        sim_result : SimResult
        chain_results : list[ChainResult]   Raw chain outputs for detailed analysis.
        """
        # Lazy import to avoid circular dependency at module level
        from ..equilibrium.markov import run_chain, ChainResult  # noqa: PLC0415

        n_edges = len(self.network.canon_edges)
        n = self.network.n_players

        # Diverse initial states
        rng_init = np.random.default_rng(0)
        h0_list = [rng_init.uniform(0.3, 0.8, n_edges) for _ in range(self.n_chains)]
        kappa0_list = [rng_init.uniform(0.3 * self.params.K_cap,
                                         0.8 * self.params.K_cap, n)
                       for _ in range(self.n_chains)]
        mu0_list = [rng_init.uniform(0.3 * self.params.M_cap,
                                      0.8 * self.params.M_cap, n)
                    for _ in range(self.n_chains)]

        chain_results: List[ChainResult] = []
        trajectories: List[np.ndarray] = []

        for i in range(self.n_chains):
            net = self.network.copy()
            chain = run_chain(
                network=net,
                params=self.params,
                T=self.T,
                h0=h0_list[i],
                kappa0=kappa0_list[i],
                mu0=mu0_list[i],
                seed=seed + i * 7,
            )
            chain_results.append(chain)
            # Use first edge for single-edge summary
            trajectories.append(chain.h_trajectory[:, 0])

        pooled = np.concatenate([h[self.burn_in:] for h in trajectories])
        hoe_stats = hoe_statistics_from_series(pooled)

        hsi_val = self.params.HSI or 1.0
        regime = self.params.regime(K_i=0, V_j=0) if self.params.HSI else "unknown"

        sim_result = SimResult(
            h_trajectories=trajectories,
            burn_in=self.burn_in,
            T=self.T,
            n_chains=self.n_chains,
            regime=regime,
            HSI=float(hsi_val),
            mode="full_markov",
            hoe_stats=hoe_stats,
        )
        return sim_result, chain_results
