"""
jacobian.py
===========
Full system Jacobian for the n-player CHSE game (Section 6 / Appendix B).

The state vector is s = (h_ij for all edges, κ_i for all i, μ_i for all i).
Its dimension is 2|E| + 2n.

The Jacobian J has the block structure:

    J = J_belief + K^T

where J_belief is block-diagonal with 2×2 blocks (one per edge, each
shaped like the two-player J_2 below), and K^T is the transpose of the
propagation kernel matrix.

Two-player Jacobian block (J_2):

    J_2 = −μ + ∂η*/∂h − ∂κ*/∂h · r̄

Oscillation condition from J_2 eigenvalues:

    disc(J_2) = (∂η*/∂h)² − 4 · μ · ∂κ*/∂h · r̄

  disc > 0  →  real eigenvalues  →  stable node
  disc < 0  →  complex eigenvalues  →  oscillatory (Hopf bifurcation)

Phase boundary theorems:
    Theorem 6.1: Stable ↔ ρ(J) < 1  ↔  HSI·(1+2·PI) > 1
    Theorem 6.2: Cascade ↔ ρ(K) ≥ 1 − 1/HSI
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from ..core.primitives import Params
from ..core.network import CHSENetwork
from ..core.kernel import spectral_radius as kernel_spectral_radius


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class JacobianResult:
    """
    Output of a Jacobian spectral analysis.

    Attributes
    ----------
    rho_J        : float   Spectral radius of J = J_belief + K^T.
    rho_J_belief : float   Spectral radius of J_belief alone.
    rho_K        : float   Spectral radius of propagation kernel K.
    eigenvalues  : ndarray Complex eigenvalues of J (full system).
    disc_2player : float   Two-player discriminant μ² − 4η̄κ̄.
    regime       : str     Regime label.
    Z            : float   Instability index HSI⁻¹·(1+2·PI).
    HSI          : float
    PI           : float
    n_edges      : int     |E|.
    """
    rho_J: float
    rho_J_belief: float
    rho_K: float
    eigenvalues: np.ndarray
    disc_2player: float
    regime: str
    Z: float
    HSI: float
    PI: float
    n_edges: int

    def summary(self) -> str:
        lines = [
            f"HSI            : {self.HSI:.4f}",
            f"PI             : {self.PI:.4f}",
            f"Z (instability): {self.Z:.4f}",
            f"Regime         : {self.regime}",
            f"ρ(J)           : {self.rho_J:.4f}",
            f"ρ(J_belief)    : {self.rho_J_belief:.4f}",
            f"ρ(K)           : {self.rho_K:.4f}",
            f"disc (2-player): {self.disc_2player:.4f}",
            f"|E|            : {self.n_edges}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Jacobian builder
# ---------------------------------------------------------------------------

class SystemJacobian:
    """
    Constructs and analyses the full CHSE system Jacobian.

    For Phase 2 we work with the belief-only subsystem (dimension |E|),
    with the resource dynamics approximated by their steady-state values.
    The full 2|E|+2n system is noted but the resource block is diagonal
    and does not affect the regime classification.

    Parameters
    ----------
    network : CHSENetwork
    params  : Params
    mu      : float   Mean-reversion strength μ (composite of μ_II).
    eta_bar : float   Steady-state reframing rate η̄.
    kappa_bar: float  Steady-state resistance rate κ̄.
    r_bar   : float   Attack rate r̄.
    K       : ndarray | None  Propagation kernel.  If None, K = 0 (no network).
    """

    def __init__(
        self,
        network: CHSENetwork,
        params: Params,
        mu: float,
        eta_bar: float,
        kappa_bar: float,
        r_bar: float = 1.0,
        K: Optional[np.ndarray] = None,
    ) -> None:
        self.network = network
        self.params = params
        self.mu = mu
        self.eta_bar = eta_bar
        self.kappa_bar = kappa_bar
        self.r_bar = r_bar
        self.K = K if K is not None else np.zeros(
            (len(network.canon_edges), len(network.canon_edges))
        )

    # ------------------------------------------------------------------
    # Two-player Jacobian block J_2
    # ------------------------------------------------------------------

    def _J2_eigenvalues(self) -> tuple[float, float]:
        """
        Eigenvalues of the linearised two-player block.

        Characteristic equation: λ² + μλ + η̄κ̄ = 0
        Roots: λ = (−μ ± √(μ² − 4η̄κ̄)) / 2
        """
        mu = self.mu
        eta = self.eta_bar
        kappa = self.kappa_bar
        disc = mu**2 - 4.0 * eta * kappa
        sqrt_disc = np.sqrt(complex(disc))
        lam1 = (-mu + sqrt_disc) / 2.0
        lam2 = (-mu - sqrt_disc) / 2.0
        return complex(lam1), complex(lam2)

    def _J2_spectral_radius(self) -> float:
        lam1, lam2 = self._J2_eigenvalues()
        return float(max(abs(lam1), abs(lam2)))

    # ------------------------------------------------------------------
    # Full belief-subsystem Jacobian
    # ------------------------------------------------------------------

    def _build_J_belief(self) -> np.ndarray:
        """
        Block-diagonal Jacobian J_belief (|E| × |E|).

        Each diagonal entry is the spectral radius of the corresponding
        two-player J_2 block, scaled by the stability indicator.

        For a single edge this reduces to the scalar −μ (the dominant
        real part of the J_2 eigenvalues).  For the network we use the
        full block structure: each edge gets its own J_2, and off-diagonal
        entries are zero (propagation coupling enters via K^T separately).
        """
        n_e = len(self.network.canon_edges)
        J_b = np.zeros((n_e, n_e))

        # Real part of J_2 dominant eigenvalue for each edge
        # (all edges share the same μ, η̄, κ̄ in Phase 2)
        lam1, lam2 = self._J2_eigenvalues()
        # Use the eigenvalue with largest real part (least stable)
        dominant_real = max(lam1.real, lam2.real)

        for idx in range(n_e):
            J_b[idx, idx] = dominant_real

        return J_b

    def build(self) -> np.ndarray:
        """
        Build the full system Jacobian J = J_belief + K^T.

        Returns
        -------
        np.ndarray  Shape (|E|, |E|).
        """
        J_belief = self._build_J_belief()
        return J_belief + self.K.T

    # ------------------------------------------------------------------
    # Spectral analysis and regime classification
    # ------------------------------------------------------------------

    def analyse(self) -> JacobianResult:
        """
        Full spectral analysis of the system Jacobian.

        Returns
        -------
        JacobianResult
        """
        J = self.build()
        J_belief = self._build_J_belief()

        eigenvalues = np.linalg.eigvals(J)
        rho_J = float(np.max(np.abs(eigenvalues)))
        rho_J_belief = self._J2_spectral_radius()
        rho_K = kernel_spectral_radius(self.K)

        disc = self.mu**2 - 4.0 * self.eta_bar * self.kappa_bar

        # Composite indices
        HSI = self.params.hsi(self.kappa_bar, self.eta_bar)
        PI = self.params.pi()
        Z = self.params.instability_index(self.kappa_bar, self.eta_bar)

        # Regime from Z (Definition 6.1)
        regime = self.params.regime(self.kappa_bar, self.eta_bar)

        # Cascade override: if ρ(K) ≥ 1 − 1/HSI, cascade-dominated
        if HSI > 0 and rho_K >= 1.0 - (1.0 / HSI):
            if regime == "stable":
                regime = "cascade"

        return JacobianResult(
            rho_J=rho_J,
            rho_J_belief=rho_J_belief,
            rho_K=rho_K,
            eigenvalues=eigenvalues,
            disc_2player=float(disc),
            regime=regime,
            Z=float(Z),
            HSI=float(HSI),
            PI=float(PI),
            n_edges=len(self.network.canon_edges),
        )

    # ------------------------------------------------------------------
    # Phase boundary checks (Theorems 6.1 and 6.2)
    # ------------------------------------------------------------------

    def is_stable(self) -> bool:
        """Theorem 6.1: HSI·(1+2·PI) > 1  ↔  stable hierarchy."""
        return self.params.instability_index(self.kappa_bar, self.eta_bar) < 1.0

    def cascade_threshold_rho_K(self) -> float:
        """
        Theorem 6.2: cascade threshold for ρ(K).

        Cascade-dominated when ρ(K) ≥ 1 − 1/HSI.
        Returns the threshold value 1 − 1/HSI.
        """
        HSI = self.params.hsi(self.kappa_bar, self.eta_bar)
        if HSI == float("inf"):
            return 1.0
        return max(0.0, 1.0 - 1.0 / HSI)

    def hopf_bifurcation_condition(self) -> bool:
        """
        True if the system is at or past the Hopf bifurcation boundary.

        Oscillatory iff: (∂η*/∂h)² < 4·μ·(∂κ*/∂h·r̄)
        In the simplified form: μ² < 4·η̄·κ̄
        """
        return self.mu**2 < 4.0 * self.eta_bar * self.kappa_bar
