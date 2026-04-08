"""
flip_threshold.py
=================
Implements the leadership flip time formula (Definition 3.2 of the paper).

Starting from h(0) = h0 > 1/2 (player 1 leads), the first time the
hierarchy flips — h(t*) = 1/2 — in the linearised system is:

    t* = (1/μ̃) · ln((h0 − 1/2) / ε)

where:
    μ̃  = effective decay rate  |Re(λ)| from the Jacobian eigenvalues
    ε  = flip precision threshold (how close to 0.5 counts as a flip)

Properties (from the paper):
  - t* is decreasing in λ_R (reframing efficiency)
  - t* is increasing in HSI
  - When HSI > 1 (stable regime), t* → ∞ in the deterministic model
    (the hierarchy is permanently stable)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..core.primitives import Params
from .oscillation import OscillationAnalysis


@dataclass
class FlipResult:
    """
    Output of a flip-time calculation.

    Attributes
    ----------
    t_star     : float | None
        Predicted flip time.  None if the system is stable (HSI > 1)
        and the analytical formula gives t* → ∞.
    h0         : float   Initial belief.
    epsilon    : float   Flip precision threshold.
    decay_rate : float   Effective decay rate μ̃ = |Re(λ)|.
    oscillates : bool    Whether the system oscillates.
    hsi        : float | None  HSI value if supplied.
    regime     : str     Regime label from OscillationResult.
    """
    t_star: float | None
    h0: float
    epsilon: float
    decay_rate: float
    oscillates: bool
    hsi: float | None
    regime: str

    def summary(self) -> str:
        ts = f"{self.t_star:.4f}" if self.t_star is not None else "∞ (stable)"
        return (
            f"Flip time t*  : {ts}\n"
            f"Initial h0    : {self.h0:.4f}\n"
            f"Epsilon       : {self.epsilon:.4f}\n"
            f"Decay rate μ̃  : {self.decay_rate:.4f}\n"
            f"Oscillates    : {self.oscillates}\n"
            f"HSI           : {self.hsi}\n"
            f"Regime        : {self.regime}"
        )


def flip_time(
    h0: float,
    mu: float,
    eta_bar: float,
    kappa_bar: float,
    r_bar: float = 0.5,
    epsilon: float = 0.01,
    params: Params | None = None,
    noise_std: float = 0.0,
) -> FlipResult:
    """
    Compute the leadership flip time t*.

    For the deterministic system (noise_std=0):
        t* = (1/μ̃) · ln((h₀ − ½) / ε)

    For the stochastic system (noise_std > 0):
        Uses the Ornstein-Uhlenbeck first-passage approximation.
        The expected first-passage time to h=0.5 from h0 under
        ḣ = −μ(h−h*) + σε is approximately:
            t* ≈ (1/μ) · ln((h₀ − h*) / ε)   when h* > 0.5
            t* ≈ π / (μ · √(1 − 4ηκ/μ²))     otherwise (oscillatory regime)

    Parameters
    ----------
    h0        : float   Initial hierarchy belief (must be > 0.5).
    mu        : float   Mean-reversion strength.
    eta_bar   : float   Constant reframing rate.
    kappa_bar : float   Constant resistance rate.
    r_bar     : float   Reframing attack rate.
    epsilon   : float   Flip precision threshold (default 0.01).
    params    : Params  Full parameter object (for HSI lookup).
    noise_std : float   Noise standard deviation (0 = deterministic).

    Returns
    -------
    FlipResult
    """
    if h0 <= 0.5:
        raise ValueError(f"h0 must be > 0.5 for player 1 to be the leader, got {h0}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")

    params = params or Params()
    osc = OscillationAnalysis(mu=mu, eta_bar=eta_bar,
                               kappa_bar=kappa_bar, r_bar=r_bar,
                               params=params)
    result = osc.analyse()
    decay = result.decay_rate

    h_star = 0.5 + (eta_bar - kappa_bar * r_bar) / mu
    h_star = float(np.clip(h_star, 0.0, 1.0))

    hsi_val = params.HSI

    # Stable regime: fixed point well above 0.5 → never flips deterministically
    if (not result.oscillates) and h_star > 0.5 + epsilon and noise_std < 1e-10:
        return FlipResult(
            t_star=None,
            h0=h0,
            epsilon=epsilon,
            decay_rate=decay,
            oscillates=False,
            hsi=hsi_val,
            regime=result.regime,
        )

    margin = h0 - 0.5
    if margin < epsilon:
        return FlipResult(
            t_star=0.0,
            h0=h0,
            epsilon=epsilon,
            decay_rate=decay,
            oscillates=result.oscillates,
            hsi=hsi_val,
            regime=result.regime,
        )

    if noise_std > 1e-10 and result.oscillates:
        # Stochastic oscillatory regime: use half-period as flip time estimate
        # t_flip ≈ period / 2  (time from h0 to first crossing of 0.5)
        if result.period is not None:
            t_star = result.period / 2.0
        else:
            t_star = (1.0 / decay) * np.log(margin / epsilon)
    else:
        # Deterministic: t* = (1/μ̃) · ln((h₀ − ½) / ε)
        if decay < 1e-12:
            t_star = float("inf")
        else:
            t_star = (1.0 / decay) * np.log(margin / epsilon)

    return FlipResult(
        t_star=float(t_star),
        h0=h0,
        epsilon=epsilon,
        decay_rate=decay,
        oscillates=result.oscillates,
        hsi=hsi_val,
        regime=result.regime,
    )


def flip_time_vs_hsi(
    hsi_values: np.ndarray,
    h0: float = 0.75,
    mu: float = 1.0,
    eta_bar: float = 0.3,
    r_bar: float = 0.5,
    epsilon: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute t* as a function of HSI.

    Varies κ̄ to achieve each target HSI (holding η̄ and μ fixed),
    then computes flip time.  Returns (hsi_values, t_star_values).

    Entries where t* = ∞ are returned as np.nan.

    Parameters
    ----------
    hsi_values : np.ndarray  Target HSI values to sweep.
    h0         : float       Initial belief.
    mu         : float       Mean-reversion strength.
    eta_bar    : float       Reframing rate.
    r_bar      : float       Attack rate.
    epsilon    : float       Flip threshold.

    Returns
    -------
    hsi_values  : np.ndarray  (same as input)
    t_star_vals : np.ndarray  Flip times (nan where infinite).
    """
    t_star_vals = np.full(len(hsi_values), np.nan)
    for idx, hsi in enumerate(hsi_values):
        # Derive κ̄ from HSI: HSI = λ_κ·K / (λ_R·V) ≈ κ̄ / η̄ (simplified)
        # Using HSI as direct proxy: κ̄ = HSI · η̄
        kappa_bar = hsi * eta_bar
        p = Params(HSI=float(hsi))
        res = flip_time(h0=h0, mu=mu, eta_bar=eta_bar,
                        kappa_bar=kappa_bar, r_bar=r_bar,
                        epsilon=epsilon, params=p)
        if res.t_star is not None and np.isfinite(res.t_star):
            t_star_vals[idx] = res.t_star

    return hsi_values, t_star_vals
