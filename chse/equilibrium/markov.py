"""
markov.py
=========
Markov chain simulation engine for CHSE.

The CHSE dynamical system defines a Markov process on the compact state
space S = H × ∏_i [0,K_i] × [0,M_i].

This module implements the transition kernel P(s, ·) by running one full
period of the game:

    Stage 1 — investment portfolios chosen (best-response)
    Stage 2 — stage game resolves, payoffs realised
    Stage 3 — anticipation signals realised, suppression applied
    Stage 4 — belief update: h(t+1) via Mechanisms I–IV
    Stage 5 — resource update: κ(t+1), μ(t+1)
    Stage 6 — propagation kernel applied

The chain is run for many periods from many initial states to estimate
the invariant measure π* (the HOE).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..core.primitives import Params, CapitalStocks
from ..core.network import CHSENetwork, Edge
from ..core.mechanisms import (
    ambiguity_push,
    reframe_success_prob,
    reframe_resistance,
    optimal_eta,
    optimal_kappa_spend,
)
from ..core.anticipation import AnticipatState, draw_anticipation_signal
from ..core.kernel import build_kernel, TrustState


# ---------------------------------------------------------------------------
# State snapshot
# ---------------------------------------------------------------------------

@dataclass
class MarkovState:
    """
    Complete state of the CHSE Markov chain at one time step.

    Attributes
    ----------
    h       : np.ndarray   Belief vector (one entry per canonical edge).
    kappa   : np.ndarray   Credibility capital per player.
    mu      : np.ndarray   Manipulation capital per player.
    t       : int          Period index.
    """
    h: np.ndarray
    kappa: np.ndarray
    mu: np.ndarray
    t: int = 0

    def copy(self) -> "MarkovState":
        return MarkovState(
            h=self.h.copy(),
            kappa=self.kappa.copy(),
            mu=self.mu.copy(),
            t=self.t,
        )

    @property
    def dim(self) -> int:
        return len(self.h) + len(self.kappa) + len(self.mu)

    def to_vector(self) -> np.ndarray:
        return np.concatenate([self.h, self.kappa, self.mu])


# ---------------------------------------------------------------------------
# Transition kernel — one period
# ---------------------------------------------------------------------------

class CHSETransition:
    """
    Implements one period of the CHSE Markov chain.

    Parameters
    ----------
    network : CHSENetwork
    params  : Params
    rng     : np.random.Generator
    """

    def __init__(
        self,
        network: CHSENetwork,
        params: Params,
        rng: np.random.Generator,
    ) -> None:
        self.network = network
        self.params = params
        self.rng = rng
        self.n = network.n_players
        self.edges = network.canon_edges
        self.n_edges = len(self.edges)
        self.edge_idx = {e: i for i, e in enumerate(self.edges)}

    def step(
        self,
        state: MarkovState,
        ant_state: AnticipatState,
        trust_state: TrustState,
    ) -> MarkovState:
        """
        Advance the Markov chain by one period.

        Returns the new MarkovState (does not modify inputs).
        Modifies ant_state and trust_state in place (they carry history).
        """
        p = self.params
        h = state.h.copy()
        kappa = state.kappa.copy()
        mu = state.mu.copy()

        delta_total = np.zeros(self.n_edges)

        # ── Mechanisms II & III per edge ──────────────────────────────
        for e_idx, e in enumerate(self.edges):
            i, j = e
            h_ij = h[e_idx]
            leader = i if h_ij >= 0.5 else j
            follower = j if h_ij >= 0.5 else i

            # Mechanism II — ambiguity push
            # Both players spend a small fraction of mu on ambiguity
            gamma = 0.05 * mu[follower]
            delta_II = ambiguity_push(h_ij, gamma, p)
            mu[follower] = max(0.0, mu[follower] - gamma)

            # Mechanism III — reframe attack by follower
            eta = optimal_eta(h_ij, p) if h_ij < 0.5 else 0.0
            eta = min(eta, mu[follower])
            rho = reframe_resistance(kappa[leader], p)
            P_R = reframe_success_prob(eta, rho, p)
            # Signed: follower attacks leader → h moves toward 0.5
            sign = -1.0 if h_ij >= 0.5 else +1.0
            delta_III = sign * p.alpha_R * P_R

            # Capital depletion
            mu[follower] = max(0.0, mu[follower] - eta)
            kappa[leader] = max(0.0, kappa[leader] - p.delta_kappa * P_R)

            delta_total[e_idx] += delta_II + delta_III

        # ── Mechanism I — anticipation signals ───────────────────────
        for e_idx, e in enumerate(self.edges):
            i, j = e
            h_ij = h[e_idx]
            xi = draw_anticipation_signal(i, j, h_ij, p, self.rng)
            ant_state.update(i, j, xi)
            delta_total[e_idx] += p.alpha_I * xi

        # ── Mechanism IV — propagation ────────────────────────────────
        K = build_kernel(self.network, ant_state, trust_state, p)
        prop = K @ delta_total
        delta_total += prop

        # ── Belief update ─────────────────────────────────────────────
        h_new = np.clip(h + delta_total, 0.0, 1.0)

        # ── Trust update ──────────────────────────────────────────────
        for e1 in self.edges:
            for e2 in self.edges:
                if e1 != e2:
                    trust_state.update(
                        e1, e2,
                        delta_total[self.edge_idx[e1]],
                        delta_total[self.edge_idx[e2]],
                    )

        # ── Resource replenishment ────────────────────────────────────
        kappa_new = np.minimum(p.K_cap, kappa + p.rho_kappa)
        mu_new = np.minimum(p.M_cap, mu + p.rho_mu)

        # ── Update network beliefs for next-period kernel ─────────────
        self.network.set_belief_vector(h_new)

        return MarkovState(
            h=h_new,
            kappa=kappa_new,
            mu=mu_new,
            t=state.t + 1,
        )


# ---------------------------------------------------------------------------
# Chain runner
# ---------------------------------------------------------------------------

@dataclass
class ChainResult:
    """
    Output of a Markov chain simulation run.

    Attributes
    ----------
    states      : list[MarkovState]   All states (length T+1).
    h_trajectory: np.ndarray          Shape (T+1, n_edges).
    kappa_traj  : np.ndarray          Shape (T+1, n_players).
    mu_traj     : np.ndarray          Shape (T+1, n_players).
    params      : Params
    n_edges     : int
    n_players   : int
    """
    states: List[MarkovState]
    h_trajectory: np.ndarray
    kappa_traj: np.ndarray
    mu_traj: np.ndarray
    params: Params
    n_edges: int
    n_players: int

    @property
    def T(self) -> int:
        return len(self.states) - 1

    def turnover_count(self, edge_idx: int = 0) -> int:
        """Number of leadership flips on the given edge."""
        h = self.h_trajectory[:, edge_idx]
        return int(np.sum(np.abs(np.diff((h > 0.5).astype(int)))))

    def turnover_frequency(self, edge_idx: int = 0) -> float:
        """Flips per period on the given edge."""
        return self.turnover_count(edge_idx) / max(self.T, 1)

    def h_variance(self, edge_idx: int = 0) -> float:
        return float(np.var(self.h_trajectory[self.T // 4:, edge_idx]))

    def h_mean(self, edge_idx: int = 0) -> float:
        return float(np.mean(self.h_trajectory[self.T // 4:, edge_idx]))


def run_chain(
    network: CHSENetwork,
    params: Params,
    T: int = 500,
    kappa0: Optional[np.ndarray] = None,
    mu0: Optional[np.ndarray] = None,
    h0: Optional[np.ndarray] = None,
    seed: int = 42,
) -> ChainResult:
    """
    Run the CHSE Markov chain for T periods from a given initial state.

    Parameters
    ----------
    network : CHSENetwork
    params  : Params
    T       : int    Number of periods.
    kappa0  : np.ndarray | None  Initial credibility capitals (n,).
    mu0     : np.ndarray | None  Initial manipulation capitals (n,).
    h0      : np.ndarray | None  Initial beliefs (n_edges,).
    seed    : int    Random seed.

    Returns
    -------
    ChainResult
    """
    n = network.n_players
    n_edges = len(network.canon_edges)
    rng = np.random.default_rng(seed)

    # Initial state
    if h0 is None:
        h0 = network.belief_vector()
    if kappa0 is None:
        kappa0 = np.full(n, params.K_cap / 2.0)
    if mu0 is None:
        mu0 = np.full(n, params.M_cap / 2.0)

    # Sync network beliefs with h0
    net = network.copy()
    net.set_belief_vector(h0)

    state = MarkovState(h=h0.copy(), kappa=kappa0.copy(), mu=mu0.copy())
    ant_state = AnticipatState.initialise(net)
    trust_state = TrustState.initialise(net)

    transition = CHSETransition(net, params, rng)

    h_traj = np.zeros((T + 1, n_edges))
    kappa_traj = np.zeros((T + 1, n))
    mu_traj = np.zeros((T + 1, n))
    states = [state.copy()]

    h_traj[0] = state.h
    kappa_traj[0] = state.kappa
    mu_traj[0] = state.mu

    for t in range(T):
        state = transition.step(state, ant_state, trust_state)
        h_traj[t + 1] = state.h
        kappa_traj[t + 1] = state.kappa
        mu_traj[t + 1] = state.mu
        states.append(state.copy())

    return ChainResult(
        states=states,
        h_trajectory=h_traj,
        kappa_traj=kappa_traj,
        mu_traj=mu_traj,
        params=params,
        n_edges=n_edges,
        n_players=n,
    )
