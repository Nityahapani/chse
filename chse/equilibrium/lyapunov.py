"""
lyapunov.py
===========
Lyapunov stability analysis for the HOE (Theorem 8.2).

The Lyapunov function (Definition 8.2):

    V(s) = d(s, Ω*)² + Σ_i [θ_κ(κ_i − κ_i*(s))² + θ_μ(μ_i − μ_i*(s))²]

where:
    Ω* = supp(π*) is the HOE orbit support
    s*(s) = argmin_{s*∈Ω*} ‖s − s*‖ is the nearest point on the orbit
    θ_κ, θ_μ > 0 are weights

Theorem 8.2 (Lyapunov Stability):
    If Γ < (1−δ)/(1+δ), then π* is Lyapunov stable.
    With positive replenishment rates and sufficiently large efficiency
    parameters, π* is asymptotically stable.

This module:
  1. Estimates Ω* from a chain's stationary distribution
  2. Evaluates V(s) along a trajectory
  3. Verifies ΔV < 0 outside Ω* (stability criterion)
  4. Computes the stability condition Γ < (1−δ)/(1+δ)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from ..core.primitives import Params
from .markov import ChainResult, MarkovState


# ---------------------------------------------------------------------------
# Orbit support estimation
# ---------------------------------------------------------------------------

def estimate_orbit_support(
    chain: ChainResult,
    burn_in: int = 100,
    n_support_points: int = 50,
) -> np.ndarray:
    """
    Estimate Ω* = supp(π*) from the post-burn-in trajectory.

    Uses k-means-style subsampling: select n_support_points representative
    states from the stationary portion of the chain.

    Parameters
    ----------
    chain            : ChainResult
    burn_in          : int    Periods to discard.
    n_support_points : int    Number of points to represent Ω*.

    Returns
    -------
    np.ndarray  Shape (n_support_points, state_dim).
                Each row is a state vector [h..., kappa..., mu...].
    """
    states_post = chain.states[burn_in:]
    n_post = len(states_post)

    # Subsample evenly
    indices = np.linspace(0, n_post - 1, n_support_points, dtype=int)
    support = np.array([states_post[i].to_vector() for i in indices])
    return support


def nearest_on_orbit(
    s_vec: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    """
    Return the nearest point on the orbit support to s.

    Parameters
    ----------
    s_vec   : np.ndarray  Current state as vector.
    support : np.ndarray  Shape (n_support, dim) — orbit support points.

    Returns
    -------
    np.ndarray  The nearest support point.
    """
    dists = np.linalg.norm(support - s_vec, axis=1)
    return support[np.argmin(dists)]


# ---------------------------------------------------------------------------
# Lyapunov function
# ---------------------------------------------------------------------------

def lyapunov_V(
    state: MarkovState,
    support: np.ndarray,
    params: Params,
    theta_kappa: float = 1.0,
    theta_mu: float = 1.0,
) -> float:
    """
    Evaluate the Lyapunov function V(s) at a given state.

        V(s) = ‖s − s*(s)‖² + Σ_i [θ_κ(κ_i−κ_i*)² + θ_μ(μ_i−μ_i*)²]

    Parameters
    ----------
    state       : MarkovState  Current state.
    support     : np.ndarray   Orbit support (n_support, dim).
    params      : Params
    theta_kappa : float        Weight on κ deviations.
    theta_mu    : float        Weight on μ deviations.

    Returns
    -------
    float  V(s) ≥ 0.
    """
    s_vec = state.to_vector()
    s_star = nearest_on_orbit(s_vec, support)

    # Distance to orbit
    dist_sq = float(np.sum((s_vec - s_star) ** 2))

    # Capital deviations
    n_h = len(state.h)
    n_k = len(state.kappa)
    kappa_star = s_star[n_h: n_h + n_k]
    mu_star = s_star[n_h + n_k:]

    kappa_dev = float(np.sum(theta_kappa * (state.kappa - kappa_star) ** 2))
    mu_dev = float(np.sum(theta_mu * (state.mu - mu_star) ** 2))

    return dist_sq + kappa_dev + mu_dev


# ---------------------------------------------------------------------------
# Stability verification
# ---------------------------------------------------------------------------

@dataclass
class LyapunovResult:
    """
    Output of the Lyapunov stability check.

    Attributes
    ----------
    V_trajectory    : np.ndarray  V(s(t)) along the trajectory.
    delta_V         : np.ndarray  ΔV(t) = V(s(t+1)) − V(s(t)).
    frac_decreasing : float       Fraction of steps with ΔV < 0.
    mean_delta_V    : float       Mean ΔV (negative → stable tendency).
    stability_cond  : float       Γ value used.
    stability_bound : float       (1−δ)/(1+δ) — the stability threshold.
    lyapunov_stable : bool        Γ < (1−δ)/(1+δ).
    """
    V_trajectory: np.ndarray
    delta_V: np.ndarray
    frac_decreasing: float
    mean_delta_V: float
    stability_cond: float
    stability_bound: float
    lyapunov_stable: bool

    def summary(self) -> str:
        return (
            f"Lyapunov Stability Analysis\n"
            f"  Γ (propagation factor)  : {self.stability_cond:.4f}\n"
            f"  (1-δ)/(1+δ) bound       : {self.stability_bound:.4f}\n"
            f"  Lyapunov stable (Γ<bound): {self.lyapunov_stable}\n"
            f"  Fraction ΔV < 0         : {self.frac_decreasing:.4f}\n"
            f"  Mean ΔV                 : {self.mean_delta_V:.6f}\n"
        )


def verify_lyapunov(
    chain: ChainResult,
    params: Params,
    burn_in: int = 100,
    n_support_points: int = 50,
    theta_kappa: float = 1.0,
    theta_mu: float = 1.0,
    Gamma: float = 0.4,
) -> LyapunovResult:
    """
    Verify Lyapunov stability along a simulated trajectory.

    Steps:
    1. Estimate Ω* from post-burn-in states.
    2. Evaluate V(s(t)) for each post-burn-in state.
    3. Compute ΔV(t) = V(t+1) − V(t).
    4. Check whether mean ΔV < 0 (stability).
    5. Check formal condition Γ < (1−δ)/(1+δ).

    Parameters
    ----------
    chain            : ChainResult
    params           : Params
    burn_in          : int    Burn-in to discard.
    n_support_points : int    Points to represent Ω*.
    theta_kappa      : float  Capital weight.
    theta_mu         : float  Capital weight.
    Gamma            : float  Propagation factor to test.

    Returns
    -------
    LyapunovResult
    """
    support = estimate_orbit_support(chain, burn_in, n_support_points)
    post_states = chain.states[burn_in:]

    V_vals = np.array([
        lyapunov_V(s, support, params, theta_kappa, theta_mu)
        for s in post_states
    ])

    delta_V = np.diff(V_vals)
    frac_dec = float(np.mean(delta_V < 0))
    mean_dV = float(np.mean(delta_V))

    # Formal stability condition: Γ < (1-δ)/(1+δ)
    delta = params.discount
    bound = (1.0 - delta) / (1.0 + delta)
    lyap_stable = Gamma < bound

    return LyapunovResult(
        V_trajectory=V_vals,
        delta_V=delta_V,
        frac_decreasing=frac_dec,
        mean_delta_V=mean_dV,
        stability_cond=Gamma,
        stability_bound=bound,
        lyapunov_stable=lyap_stable,
    )
