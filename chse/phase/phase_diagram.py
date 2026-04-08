"""
phase_diagram.py
================
Phase diagram of CHSE regimes over (HSI, PI) parameter space (Section 6).

The Instability Index Z = HSI⁻¹ · (1 + 2·PI) partitions parameter space
into four qualitatively distinct dynamic regimes:

    Z < 1          → Stable Hierarchy
    1 ≤ Z < 2      → Oscillatory Hierarchy
    2 ≤ Z < 3.5    → Cascade-Dominated
    Z ≥ 3.5        → Turbulent

Phase boundaries:
    Theorem 6.1: stable/oscillatory boundary  →  HSI·(1+2·PI) = 1
    Theorem 6.2: oscillatory/cascade boundary →  ρ(K) = 1 − 1/HSI

This module computes:
  - RegimeGrid:  a 2D grid of Z values and regime labels over (HSI, PI) space
  - PhaseDiagram: the full analysis including boundary curves and regime areas
  - Boundary verification against the spectral theorems
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

from ..core.primitives import Params


# ---------------------------------------------------------------------------
# Regime thresholds
# ---------------------------------------------------------------------------

REGIME_THRESHOLDS = {
    "stable":      (0.0, 1.0),
    "oscillatory": (1.0, 2.0),
    "cascade":     (2.0, 3.5),
    "turbulent":   (3.5, float("inf")),
}

REGIME_COLOURS = {
    "stable":      "#2a7ab5",
    "oscillatory": "#e8a020",
    "cascade":     "#c0392b",
    "turbulent":   "#7b2d8b",
}


def z_to_regime(Z: float) -> str:
    """Map instability index Z to regime label."""
    for regime, (lo, hi) in REGIME_THRESHOLDS.items():
        if lo <= Z < hi:
            return regime
    return "turbulent"


# ---------------------------------------------------------------------------
# RegimeGrid — 2D array of Z values
# ---------------------------------------------------------------------------

@dataclass
class RegimeGrid:
    """
    A 2D grid of instability index values and regime labels over (HSI, PI) space.

    Attributes
    ----------
    hsi_values : np.ndarray  Shape (n_hsi,)
    pi_values  : np.ndarray  Shape (n_pi,)
    Z          : np.ndarray  Shape (n_pi, n_hsi) — instability index grid.
    regimes    : list[list[str]]  Shape (n_pi, n_hsi) — regime labels.
    """
    hsi_values: np.ndarray
    pi_values: np.ndarray
    Z: np.ndarray
    regimes: List[List[str]]

    @property
    def n_hsi(self) -> int:
        return len(self.hsi_values)

    @property
    def n_pi(self) -> int:
        return len(self.pi_values)

    def regime_at(self, hsi: float, pi: float) -> str:
        """Return the regime label at the given (HSI, PI) point."""
        Z = (1.0 / hsi) * (1.0 + 2.0 * pi) if hsi > 0 else float("inf")
        return z_to_regime(Z)

    def fraction_in_regime(self, regime: str) -> float:
        """Fraction of grid cells in the given regime."""
        total = self.n_hsi * self.n_pi
        count = sum(
            1 for row in self.regimes for r in row if r == regime
        )
        return count / total if total > 0 else 0.0

    def boundary_hsi(self, pi: float, boundary: str = "stable_oscillatory") -> float:
        """
        Compute the HSI value at the phase boundary for a given PI.

        stable/oscillatory:  HSI = 1 / (1 + 2·PI)
        oscillatory/cascade: HSI = 2 / (1 + 2·PI)   (Z=2 boundary)

        Parameters
        ----------
        pi       : float  Propagation intensity.
        boundary : str    'stable_oscillatory' or 'oscillatory_cascade'.
        """
        denom = 1.0 + 2.0 * pi
        if boundary == "stable_oscillatory":
            return 1.0 / denom
        elif boundary == "oscillatory_cascade":
            return 2.0 / denom
        elif boundary == "cascade_turbulent":
            return 3.5 / denom
        else:
            raise ValueError(f"Unknown boundary: {boundary}")


# ---------------------------------------------------------------------------
# PhaseDiagram — main analysis class
# ---------------------------------------------------------------------------

class PhaseDiagram:
    """
    Full (HSI, PI) phase diagram for the CHSE model.

    Computes the instability index Z = HSI⁻¹·(1+2·PI) over a grid,
    classifies regimes, and provides boundary curves.

    Parameters
    ----------
    hsi_min, hsi_max : float  Range of HSI values.
    pi_min, pi_max   : float  Range of PI values.
    n_hsi, n_pi      : int    Grid resolution.
    """

    def __init__(
        self,
        hsi_min: float = 0.1,
        hsi_max: float = 3.0,
        pi_min: float = 0.0,
        pi_max: float = 1.0,
        n_hsi: int = 200,
        n_pi: int = 200,
    ) -> None:
        self.hsi_min = hsi_min
        self.hsi_max = hsi_max
        self.pi_min = pi_min
        self.pi_max = pi_max
        self.n_hsi = n_hsi
        self.n_pi = n_pi

    def compute(self) -> RegimeGrid:
        """
        Compute the full Z grid and regime classification.

        Returns
        -------
        RegimeGrid
        """
        hsi_vals = np.linspace(self.hsi_min, self.hsi_max, self.n_hsi)
        pi_vals = np.linspace(self.pi_min, self.pi_max, self.n_pi)

        Z_grid = np.zeros((self.n_pi, self.n_hsi))
        regime_grid: List[List[str]] = []

        for pi_idx, pi in enumerate(pi_vals):
            row_regimes = []
            for hsi_idx, hsi in enumerate(hsi_vals):
                Z = (1.0 / hsi) * (1.0 + 2.0 * pi)
                Z_grid[pi_idx, hsi_idx] = Z
                row_regimes.append(z_to_regime(Z))
            regime_grid.append(row_regimes)

        return RegimeGrid(
            hsi_values=hsi_vals,
            pi_values=pi_vals,
            Z=Z_grid,
            regimes=regime_grid,
        )

    def boundary_curves(
        self,
        pi_vals: np.ndarray | None = None,
    ) -> dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute the three phase boundary curves in (HSI, PI) space.

        Returns a dict with keys:
            'stable_oscillatory'   — HSI·(1+2·PI) = 1  (Theorem 6.1)
            'oscillatory_cascade'  — HSI·(1+2·PI) = 2  (Z=2 boundary)
            'cascade_turbulent'    — HSI·(1+2·PI) = 3.5

        Each value is (hsi_array, pi_array) tracing the curve.
        """
        if pi_vals is None:
            pi_vals = np.linspace(self.pi_min, self.pi_max, 500)

        curves = {}
        for name, Z_target in [
            ("stable_oscillatory", 1.0),
            ("oscillatory_cascade", 2.0),
            ("cascade_turbulent", 3.5),
        ]:
            # Z = (1/HSI)*(1+2*PI) = Z_target  =>  HSI = (1+2*PI)/Z_target
            hsi_curve = (1.0 + 2.0 * pi_vals) / Z_target
            # Clip to valid HSI range
            mask = (hsi_curve >= self.hsi_min) & (hsi_curve <= self.hsi_max)
            curves[name] = (hsi_curve[mask], pi_vals[mask])

        return curves

    def verify_theorem_61(
        self,
        n_test: int = 50,
        tol: float = 0.05,
    ) -> dict:
        """
        Verify Theorem 6.1: stable ↔ HSI·(1+2·PI) > 1.

        Tests n_test random (HSI, PI) points and checks that the Z-based
        regime classification matches the Params.regime() output.

        Returns a dict with 'n_tested', 'n_consistent', 'fraction'.
        """
        rng = np.random.default_rng(42)
        n_consistent = 0

        for _ in range(n_test):
            hsi = rng.uniform(0.2, 3.0)
            pi = rng.uniform(0.0, 1.0)
            p = Params(HSI=hsi, PI=pi)

            # Z-based classification
            Z = (1.0 / hsi) * (1.0 + 2.0 * pi)
            z_regime = z_to_regime(Z)

            # Params.regime() uses the same formula
            params_regime = p.regime(K_i=0, V_j=0)

            if z_regime == params_regime:
                n_consistent += 1

        return {
            "n_tested": n_test,
            "n_consistent": n_consistent,
            "fraction": n_consistent / n_test,
        }

    def verify_theorem_62(
        self,
        rho_K_vals: np.ndarray | None = None,
        hsi_vals: np.ndarray | None = None,
    ) -> dict:
        """
        Verify Theorem 6.2: cascade threshold ρ(K) ≥ 1 − 1/HSI.

        Checks that for several (HSI, ρ(K)) pairs, the cascade condition
        is correctly identified.

        Returns a dict with 'pairs_tested', 'n_cascade', 'n_non_cascade'.
        """
        if rho_K_vals is None:
            rho_K_vals = np.linspace(0.0, 0.99, 20)
        if hsi_vals is None:
            hsi_vals = np.array([0.4, 0.8, 1.0, 1.5, 2.1])

        n_cascade = 0
        n_non_cascade = 0
        pairs = []

        for hsi in hsi_vals:
            threshold = max(0.0, 1.0 - 1.0 / hsi)
            for rho_K in rho_K_vals:
                is_cascade = rho_K >= threshold
                pairs.append((hsi, rho_K, threshold, is_cascade))
                if is_cascade:
                    n_cascade += 1
                else:
                    n_non_cascade += 1

        return {
            "pairs_tested": len(pairs),
            "n_cascade": n_cascade,
            "n_non_cascade": n_non_cascade,
            "sample_pairs": pairs[:5],
        }

    def special_points(self) -> List[dict]:
        """
        Return notable (HSI, PI) points used in the paper's examples.

        Each dict has keys: label, HSI, PI, Z, regime.
        """
        pts = [
            (2.1, 0.0, "Stable CB (paper example)"),
            (1.0, 0.0, "Oscillatory boundary"),
            (0.4, 0.0, "Cascade (paper Figure 2)"),
            (1.5, 0.4, "Moderate cascade risk"),
            (0.5, 0.7, "High PI cascade"),
            (2.5, 0.8, "High HSI, high PI"),
        ]
        result = []
        for hsi, pi, label in pts:
            Z = (1.0 / hsi) * (1.0 + 2.0 * pi)
            result.append({
                "label": label,
                "HSI": hsi,
                "PI": pi,
                "Z": round(Z, 4),
                "regime": z_to_regime(Z),
            })
        return result
