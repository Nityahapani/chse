"""
mechanisms.py
=============
Implements the mechanism-level belief update functions for the two-player
benchmark (Phase 1).  Mechanisms I and IV (Bayesian anticipation and
contagious propagation) require a network and are deferred to Phase 2.

Each function returns a delta_h — the additive change to the hierarchy
belief h_12 caused by that mechanism in one period.

References
----------
Section 4.2 (Mechanism II) and Section 4.3 (Mechanism III) of the paper.
"""

from __future__ import annotations

import numpy as np
from .primitives import Params, CapitalStocks


# ---------------------------------------------------------------------------
# Mechanism II — Role Ambiguity as Strategic Instrument
# ---------------------------------------------------------------------------

def ambiguity_push(
    h: float,
    gamma: float,
    params: Params,
) -> float:
    """
    Compute Δ^II h_ij from Mechanism II (Section 4.2).

    Player i spends γ units of manipulation capital to push the belief
    toward 0.5 (role ambiguity):

        Δ^II h_ij = −μ_II · γ · (h_ij − 1/2)

    This is mean-reverting: if h > 0.5 (i leads), the push is negative
    (toward ambiguity).  If h < 0.5, the push is positive.

    Parameters
    ----------
    h     : float  Current hierarchy belief h_12 ∈ [0, 1].
    gamma : float  Manipulation capital spent on ambiguity (γ ≥ 0).
    params: Params Model parameters.

    Returns
    -------
    float  The belief delta Δ^II h.
    """
    if gamma < 0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")
    return -params.mu_II * gamma * (h - 0.5)


# ---------------------------------------------------------------------------
# Mechanism III — Retroactive Reframing
# ---------------------------------------------------------------------------

def reframe_resistance(c: float, params: Params) -> float:
    """
    Compute reframe-resistance ρ_i from credibility capital spend c.

        ρ_i(τ) = 1 − exp(−λ_κ · c)

    Parameters
    ----------
    c      : float  Credibility capital spent at commitment time (c ≥ 0).
    params : Params Model parameters.

    Returns
    -------
    float  Reframe-resistance ρ ∈ [0, 1).
    """
    if c < 0:
        raise ValueError(f"c must be >= 0, got {c}")
    return 1.0 - np.exp(-params.lambda_kappa * c)


def reframe_success_prob(eta: float, rho: float, params: Params) -> float:
    """
    Compute the probability of a successful retroactive reframe (Section 4.3).

        P_R(η, ρ) = (1 − exp(−λ_R · η)) · (1 − ρ)

    Parameters
    ----------
    eta    : float  Manipulation capital spent by follower j on reframing (η ≥ 0).
    rho    : float  Leader i's reframe-resistance ρ ∈ [0, 1].
    params : Params Model parameters.

    Returns
    -------
    float  Success probability P_R ∈ [0, 1].
    """
    if eta < 0:
        raise ValueError(f"eta must be >= 0, got {eta}")
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    attack_power = 1.0 - np.exp(-params.lambda_R * eta)
    return attack_power * (1.0 - rho)


def reframing_investment(
    h: float,
    eta: float,
    rho: float,
    params: Params,
) -> float:
    """
    Compute Δ^III h_ij from Mechanism III (Section 4.3).

    Upon a successful reframe, the belief drops by α_R:

        Δ^III h_ij = −α_R · P_R(η, ρ)

    Note: the capital depletion (Δκ = −δ_κ on success) is handled
    separately in dynamics.py.

    Parameters
    ----------
    h      : float  Current hierarchy belief h_12.
    eta    : float  Follower's reframing spend η ≥ 0.
    rho    : float  Leader's reframe-resistance ρ ∈ [0, 1].
    params : Params Model parameters.

    Returns
    -------
    float  The belief delta Δ^III h (always ≤ 0 when h > 0.5).
    """
    P_R = reframe_success_prob(eta, rho, params)
    return -params.alpha_R * P_R


def commitment_resistance(
    h: float,
    kappa: float,
    r: float,
    params: Params,
) -> float:
    """
    Compute the resistance term in the two-player deterministic ODE.

    In the benchmark the continuous analogue of reframing resistance is:

        resistance_term = κ(t) · r(t)

    where r(t) is the rate of reframing attack faced by the leader.
    This term enters the ODE as +κ·r, counteracting η.

    Parameters
    ----------
    h      : float  Current hierarchy belief.
    kappa  : float  Leader's credibility capital κ.
    r      : float  Reframing attack rate r ≥ 0.
    params : Params Model parameters.

    Returns
    -------
    float  The resistance contribution to ḣ (positive = stabilising for leader).
    """
    return kappa * r


# ---------------------------------------------------------------------------
# Optimal best-response functions (Section 3 / Bottleneck 2)
# ---------------------------------------------------------------------------

def optimal_eta(h: float, params: Params) -> float:
    """
    Follower j's optimal reframing investment η*(h).

    From the log-optimal best-response (paper Section on Bottleneck 2):

        η*(h) = (1/λ_R) · ln(α_R · λ_R · (1/2 − h) / c_μ)  if h < 1/2
              = 0                                              if h ≥ 1/2

    Parameters
    ----------
    h      : float  Current hierarchy belief h_12.
    params : Params Model parameters (needs lambda_R, alpha_R, c_mu).

    Returns
    -------
    float  Optimal reframing spend η* ≥ 0.
    """
    if h >= 0.5:
        return 0.0
    interior = (params.alpha_R * params.lambda_R * (0.5 - h)) / params.c_mu
    if interior <= 0:
        return 0.0
    return max(0.0, (1.0 / params.lambda_R) * np.log(interior))


def optimal_kappa_spend(h: float, params: Params) -> float:
    """
    Leader i's optimal credibility capital investment κ*(h).

    From the log-optimal best-response (paper Section on Bottleneck 2):

        κ*(h) = (1/λ_κ) · ln(α_R · λ_κ · (h − 1/2) / c_κ)  if h > 1/2
              = 0                                              if h ≤ 1/2

    Parameters
    ----------
    h      : float  Current hierarchy belief h_12.
    params : Params Model parameters (needs lambda_kappa, alpha_R, c_kappa).

    Returns
    -------
    float  Optimal credibility spend κ* ≥ 0.
    """
    if h <= 0.5:
        return 0.0
    interior = (params.alpha_R * params.lambda_kappa * (h - 0.5)) / params.c_kappa
    if interior <= 0:
        return 0.0
    return max(0.0, (1.0 / params.lambda_kappa) * np.log(interior))
