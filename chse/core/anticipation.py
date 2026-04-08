"""
anticipation.py
===============
Mechanism I — Anticipation as a Public Good (Section 4.1).

Each player i has a latent predictive type θ_i ∈ Θ.  Anticipation
success ξ_ij(t) is a noisy signal of θ_i > θ_j.  The hierarchy belief
is the posterior:

    h_ij(t+1) = P(θ_i > θ_j | ξ_ij(1), ..., ξ_ij(t))

Updated via Beta-Binomial:

    α_i^(t+1) / (α_i^(t+1) + β_i^(t+1))
        = (α_i^(t) + ξ_ij(t)) / (α_i^(t) + β_i^(t) + 1)

Successful anticipation is a public good: it propagates to network
neighbours unless suppressed.

Suppression technology (opacity investment δ ≥ 0):

    σ_ij(δ) = 1 − exp(−λ_σ · δ)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np

from .network import CHSENetwork, Edge, _canon
from .primitives import Params


# ---------------------------------------------------------------------------
# Beta-Binomial belief state per directed pair
# ---------------------------------------------------------------------------

@dataclass
class AnticipateBelief:
    """
    Beta-Binomial state for one directed pair (i leads j).

    alpha : float  Beta distribution α parameter (successes + prior)
    beta  : float  Beta distribution β parameter (failures + prior)
    """
    alpha: float = 1.0   # uniform prior: α=β=1
    beta: float = 1.0

    @property
    def mean(self) -> float:
        """Current posterior mean = α / (α + β)."""
        return self.alpha / (self.alpha + self.beta)

    def update(self, xi: float) -> "AnticipateBelief":
        """
        Return updated belief after observing signal xi ∈ {0, 1}.

        α^(t+1) = α^(t) + ξ
        β^(t+1) = β^(t) + (1 − ξ)
        Posterior mean = (α + ξ) / (α + β + 1)
        """
        return AnticipateBelief(
            alpha=self.alpha + float(xi),
            beta=self.beta + (1.0 - float(xi)),
        )

    def accuracy(self) -> float:
        """
        Time-averaged anticipation accuracy Acc_ij.
        Equals the posterior mean of the Beta distribution.
        """
        return self.mean


# ---------------------------------------------------------------------------
# Per-edge anticipation state tracker
# ---------------------------------------------------------------------------

@dataclass
class AnticipatState:
    """
    Tracks Beta-Binomial belief states for all directed pairs in the network.

    beliefs[(i,j)] — state for directed pair (i leads j), canonical i < j.
    The reverse direction (j leads i) has complementary accuracy 1 - acc(i,j).
    """
    beliefs: Dict[Edge, AnticipateBelief] = field(default_factory=dict)

    @classmethod
    def initialise(cls, network: CHSENetwork,
                   alpha0: float = 1.0,
                   beta0: float = 1.0) -> "AnticipatState":
        """Create a fresh state with uniform Beta(alpha0, beta0) priors."""
        state = cls()
        for e in network.canon_edges:
            state.beliefs[e] = AnticipateBelief(alpha=alpha0, beta=beta0)
        return state

    def accuracy(self, i: int, j: int) -> float:
        """Return Acc_ij — time-averaged accuracy for pair (i leads j)."""
        e = _canon(i, j)
        if e not in self.beliefs:
            return 0.5
        acc = self.beliefs[e].accuracy()
        return acc if e == (i, j) else 1.0 - acc

    def update(self, i: int, j: int, xi: float) -> None:
        """Update the Beta-Binomial state for directed pair (i, j)."""
        e = _canon(i, j)
        if e not in self.beliefs:
            self.beliefs[e] = AnticipateBelief()
        if e == (i, j):
            self.beliefs[e] = self.beliefs[e].update(xi)
        else:
            # xi = 1 means i anticipated j correctly, which is a *failure*
            # for the canonical direction (j leads i)
            self.beliefs[e] = self.beliefs[e].update(1.0 - xi)


# ---------------------------------------------------------------------------
# Anticipation signal generation
# ---------------------------------------------------------------------------

def draw_anticipation_signal(
    i: int,
    j: int,
    h_ij: float,
    params: Params,
    rng: np.random.Generator,
) -> float:
    """
    Draw the anticipation signal ξ_ij ∈ {0, 1}.

    The signal is a noisy indicator of θ_i > θ_j.  We model it as:

        P(ξ_ij = 1) = h_ij · (1 − noise) + (1 − h_ij) · noise

    where noise = 0.1 (small fixed signal noise).  This means:
      - When h_ij = 1 (i clearly leads), ξ = 1 with prob 0.9.
      - When h_ij = 0.5 (pure ambiguity), ξ = 1 with prob 0.5.

    Parameters
    ----------
    i, j   : int    Player indices.
    h_ij   : float  Current hierarchy belief P(i leads j).
    params : Params Model parameters.
    rng    : np.random.Generator  RNG for reproducibility.

    Returns
    -------
    float  0.0 or 1.0.
    """
    noise = 0.1
    p_success = h_ij * (1.0 - noise) + (1.0 - h_ij) * noise
    return float(rng.random() < p_success)


# ---------------------------------------------------------------------------
# Mechanism I belief update
# ---------------------------------------------------------------------------

def mechanism_I_update(
    network: CHSENetwork,
    ant_state: AnticipatState,
    params: Params,
    rng: np.random.Generator,
    suppress: Optional[Dict[Edge, float]] = None,
) -> Dict[Edge, float]:
    """
    Compute Δ^I h for all edges in the network (Mechanism I, Section 4.1).

    Steps:
    1. For each edge (i,j), draw anticipation signal ξ_ij.
    2. Apply suppression σ_ij(δ) — signal is hidden with prob σ.
    3. Update Beta-Binomial belief state.
    4. Propagate the public signal to network neighbours.

    Returns a dict {edge: delta_h} for all edges (direct + network spillover).

    Parameters
    ----------
    network   : CHSENetwork
    ant_state : AnticipatState  Modified in place.
    params    : Params
    rng       : np.random.Generator
    suppress  : dict {edge: delta} — opacity investment per edge.
    """
    suppress = suppress or {}
    delta_h: Dict[Edge, float] = {e: 0.0 for e in network.canon_edges}

    for e in network.canon_edges:
        i, j = e
        h_ij = network.belief(i, j)

        # Draw signal
        xi = draw_anticipation_signal(i, j, h_ij, params, rng)

        # Suppression
        delta_op = suppress.get(e, 0.0)
        sigma = 1.0 - np.exp(-params.lambda_sigma * delta_op)
        if rng.random() < sigma:
            xi = 0.0   # signal suppressed — treated as no success

        # Update Beta-Binomial
        ant_state.update(i, j, xi)

        # Direct belief update on this edge
        # Δ^I h_ij = alpha_I * xi  (success raises h_ij)
        delta_h[e] += params.alpha_I * xi

        # Network spillover: propagate to all other edges involving i or j
        for k in network.neighbours(i):
            if k == j:
                continue
            e_ik = _canon(i, k)
            phi = network.distance_decay(j, k, decay_rate=1.0)
            contrib = params.alpha_I * xi * (1.0 - sigma) * phi
            # Positive for i's other edges (i gains credibility)
            if e_ik == (i, k):
                delta_h[e_ik] = delta_h.get(e_ik, 0.0) + contrib
            else:
                delta_h[e_ik] = delta_h.get(e_ik, 0.0) - contrib

        for k in network.neighbours(j):
            if k == i:
                continue
            e_jk = _canon(j, k)
            phi = network.distance_decay(i, k, decay_rate=1.0)
            contrib = params.beta_I * xi * (1.0 - sigma) * phi
            # Negative for j's other edges (j loses predictability credibility)
            if e_jk == (j, k):
                delta_h[e_jk] = delta_h.get(e_jk, 0.0) - contrib
            else:
                delta_h[e_jk] = delta_h.get(e_jk, 0.0) + contrib

    return delta_h


# ---------------------------------------------------------------------------
# Suppression technology
# ---------------------------------------------------------------------------

def suppression_probability(delta: float, params: Params) -> float:
    """
    σ_ij(δ) = 1 − exp(−λ_σ · δ)

    Parameters
    ----------
    delta  : float  Opacity investment δ ≥ 0.
    params : Params Model parameters.

    Returns
    -------
    float  Suppression probability ∈ [0, 1).
    """
    if delta < 0:
        raise ValueError(f"delta must be >= 0, got {delta}")
    return float(1.0 - np.exp(-params.lambda_sigma * delta))
