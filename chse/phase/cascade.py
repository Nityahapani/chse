"""
cascade.py
==========
Cascade analysis as spectral threshold processes (Section 7).

Implements:
  - Proposition 7.1: Spectral Cascade Condition
  - Cascade size distribution
  - Hierarchy Persistence Paradox (Bottleneck 8)
  - Cascade probability as a function of ρ(K)

The key result:

    ρ(K) < 1  →  no infinite cascade
               →  E[cascade size] ≤ α_R / (1 − ρ(K))

    ρ(K) ≥ 1  →  cascade possible

Hierarchy Persistence Paradox:
    E[cascade | collapse] is INCREASING in HSI.
    High-HSI leaders accumulate more anticipation successes, which inflates
    K weights, making the network more susceptible to cascade if they fall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from ..core.primitives import Params
from ..core.network import CHSENetwork
from ..core.kernel import (
    spectral_radius,
    expected_cascade_size,
    edge_fragility,
)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class CascadeResult:
    """
    Output of a cascade analysis at a single (ρ(K), HSI) point.

    Attributes
    ----------
    rho_K             : float  Spectral radius of propagation kernel.
    HSI               : float  Hierarchy Stability Index.
    cascade_possible  : bool   True if ρ(K) ≥ 1.
    cascade_prob      : float  P(cascade) — percolation-like probability.
    expected_size     : float  E[cascade size] (inf if ρ(K) ≥ 1).
    cascade_threshold : float  1 − 1/HSI  (Theorem 6.2).
    above_threshold   : bool   ρ(K) ≥ cascade_threshold.
    """
    rho_K: float
    HSI: float
    cascade_possible: bool
    cascade_prob: float
    expected_size: float
    cascade_threshold: float
    above_threshold: bool

    def summary(self) -> str:
        size_str = f"{self.expected_size:.4f}" if np.isfinite(self.expected_size) else "inf"
        return (
            f"ρ(K)            : {self.rho_K:.4f}\n"
            f"HSI             : {self.HSI:.4f}\n"
            f"Cascade possible: {self.cascade_possible}\n"
            f"Cascade prob    : {self.cascade_prob:.4f}\n"
            f"E[cascade size] : {size_str}\n"
            f"Threshold 1-1/H : {self.cascade_threshold:.4f}\n"
            f"Above threshold : {self.above_threshold}"
        )


# ---------------------------------------------------------------------------
# Cascade analysis class
# ---------------------------------------------------------------------------

class CascadeAnalysis:
    """
    Cascade analysis for the CHSE model.

    Parameters
    ----------
    params : Params  Model parameters (needs alpha_R, HSI).
    """

    def __init__(self, params: Params) -> None:
        self.params = params

    def cascade_probability(self, rho_K: float) -> float:
        """
        Cascade probability as a function of ρ(K).

        Uses a percolation-like sigmoid transition at ρ(K) = 1:

            P(cascade) ≈ 1 / (1 + exp(−k · (ρ(K) − 1)))

        with steepness k = 20 (sharp transition at the critical point).

        Parameters
        ----------
        rho_K : float  Spectral radius of K.

        Returns
        -------
        float  P(cascade) ∈ [0, 1].
        """
        k = 20.0
        return float(1.0 / (1.0 + np.exp(-k * (rho_K - 1.0))))

    def analyse(self, rho_K: float) -> CascadeResult:
        """
        Full cascade analysis for a given ρ(K).

        Parameters
        ----------
        rho_K : float  Spectral radius of propagation kernel K.

        Returns
        -------
        CascadeResult
        """
        alpha_R = self.params.alpha_R
        HSI = self.params.HSI if self.params.HSI is not None else 1.0

        cascade_possible = rho_K >= 1.0
        cascade_prob = self.cascade_probability(rho_K)
        exp_size = expected_cascade_size(rho_K, alpha_R)

        threshold = max(0.0, 1.0 - 1.0 / HSI) if HSI > 0 else 1.0
        above = rho_K >= threshold

        return CascadeResult(
            rho_K=rho_K,
            HSI=float(HSI),
            cascade_possible=cascade_possible,
            cascade_prob=cascade_prob,
            expected_size=exp_size,
            cascade_threshold=threshold,
            above_threshold=above,
        )

    def scan_rho_K(
        self,
        rho_K_vals: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute cascade probability and expected size over a range of ρ(K).

        Returns
        -------
        rho_K_vals : np.ndarray
        probs      : np.ndarray  P(cascade) at each ρ(K).
        sizes      : np.ndarray  E[cascade size] at each ρ(K) (nan where inf).
        """
        if rho_K_vals is None:
            rho_K_vals = np.linspace(0.0, 1.5, 300)

        probs = np.array([self.cascade_probability(r) for r in rho_K_vals])
        alpha_R = self.params.alpha_R
        sizes = np.array([
            expected_cascade_size(r, alpha_R) if r < 1.0 else np.nan
            for r in rho_K_vals
        ])

        return rho_K_vals, probs, sizes

    # ------------------------------------------------------------------
    # Hierarchy Persistence Paradox
    # ------------------------------------------------------------------

    def persistence_paradox_scan(
        self,
        hsi_vals: np.ndarray | None = None,
        acc_scaling: float = 0.3,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Demonstrate the Hierarchy Persistence Paradox (Bottleneck 8):

            ∂E[cascade | collapse] / ∂HSI > 0

        High-HSI leaders accumulate more anticipation accuracy (Acc_ij),
        which inflates ρ(K), making cascade larger upon their eventual fall.

        We model:
            Acc_ij(HSI)  = acc_scaling · HSI / (1 + HSI)  ∈ (0, acc_scaling)
            ρ(K)(HSI)    = Acc_ij · trust_avg · phi_avg
            E[cascade]   = α_R / (1 − ρ(K))  when ρ(K) < 1

        Parameters
        ----------
        hsi_vals    : np.ndarray  HSI values to sweep.
        acc_scaling : float       Max accuracy at HSI → ∞.

        Returns
        -------
        hsi_vals     : np.ndarray
        rho_K_vals   : np.ndarray  ρ(K) at each HSI.
        cascade_sizes: np.ndarray  E[cascade | collapse] at each HSI.
        """
        if hsi_vals is None:
            hsi_vals = np.linspace(0.3, 3.0, 100)

        alpha_R = self.params.alpha_R
        trust_avg = 0.5   # average cross-edge trust
        phi_avg = 0.5     # average distance decay

        rho_K_vals = np.zeros(len(hsi_vals))
        cascade_sizes = np.zeros(len(hsi_vals))

        for idx, hsi in enumerate(hsi_vals):
            # Acc_ij increases with HSI (stronger leaders predict better)
            acc = acc_scaling * hsi / (1.0 + hsi)
            rho_K = acc * trust_avg * phi_avg

            rho_K_vals[idx] = rho_K
            if rho_K < 1.0:
                cascade_sizes[idx] = expected_cascade_size(rho_K, alpha_R)
            else:
                cascade_sizes[idx] = np.nan

        return hsi_vals, rho_K_vals, cascade_sizes

    def cascade_size_distribution(
        self,
        rho_K: float,
        n_max: int = 20,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Negative-binomial cascade size distribution (Figure 4 right panel).

        Models the number of additional edge flips triggered by a single
        reframe success, via a branching process with offspring mean ρ(K).

        P(cascade size = k) ∝ ρ(K)^k · (1 − ρ(K))  (geometric approximation)

        Parameters
        ----------
        rho_K : float  Spectral radius (must be < 1 for finite cascade).
        n_max : int    Maximum cascade size to compute.

        Returns
        -------
        sizes : np.ndarray  Cascade sizes 0..n_max.
        probs : np.ndarray  Probabilities.
        """
        if rho_K >= 1.0:
            sizes = np.arange(n_max + 1)
            probs = np.ones(n_max + 1) / (n_max + 1)  # uniform when unbounded
            return sizes, probs

        p = 1.0 - rho_K
        sizes = np.arange(n_max + 1)
        probs = p * (rho_K ** sizes)
        probs = probs / probs.sum()   # normalise to n_max
        return sizes, probs
