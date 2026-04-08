"""
oscillation.py
==============
Implements the oscillation condition and stability analysis for the
two-player benchmark (Section 3.2 of the paper).

The oscillation condition (Definition 3.1):

    The system oscillates iff  μ² < 4 · η̄ · κ̄

Equivalently, the eigenvalues of the linearised Jacobian are complex
(negative real part, nonzero imaginary part), producing sustained
oscillations around h*.

The characteristic equation of the linearised system is:

    λ² + μλ + η̄κ̄ = 0

Roots:
    λ = (−μ ± √(μ² − 4η̄κ̄)) / 2

  μ² > 4η̄κ̄  →  real, negative roots  →  stable node  →  no oscillation
  μ² = 4η̄κ̄  →  repeated real root    →  critically damped
  μ² < 4η̄κ̄  →  complex roots         →  oscillatory (stable spiral)

Connection to HSI/PI (Appendix B):
  The stability boundary μ²  = 4η̄κ̄ translates to
  HSI · (1 + 2·PI) = 1 at the network level.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..core.primitives import Params


@dataclass
class OscillationResult:
    """
    Full oscillation analysis output.

    Attributes
    ----------
    oscillates   : bool    True if μ² < 4η̄κ̄.
    discriminant : float   μ² − 4η̄κ̄ (negative → oscillates).
    eigenvalues  : tuple   Complex pair (λ₁, λ₂) of the Jacobian.
    period       : float | None
        Oscillation period 2π / |Im(λ)|.  None if non-oscillatory.
    decay_rate   : float
        |Re(λ)| — rate at which oscillations damp toward h*.
        In the stable-node case this is the smaller |λ|.
    mu           : float   Mean-reversion strength used.
    eta_bar      : float   Reframing rate used.
    kappa_bar    : float   Resistance rate used.
    h_star       : float   Fixed-point belief.
    regime       : str     'stable_node' | 'oscillatory' | 'critically_damped'
    """
    oscillates: bool
    discriminant: float
    eigenvalues: tuple
    period: float | None
    decay_rate: float
    mu: float
    eta_bar: float
    kappa_bar: float
    h_star: float
    regime: str

    def summary(self) -> str:
        lines = [
            f"Regime           : {self.regime}",
            f"Oscillates       : {self.oscillates}",
            f"Discriminant     : {self.discriminant:.6f}  (μ² − 4η̄κ̄)",
            f"Eigenvalues      : {self.eigenvalues[0]:.4f},  {self.eigenvalues[1]:.4f}",
            f"Fixed point h*   : {self.h_star:.4f}",
        ]
        if self.oscillates and self.period is not None:
            lines.append(f"Period           : {self.period:.4f}")
        lines.append(f"Decay rate       : {self.decay_rate:.4f}")
        return "\n".join(lines)


class OscillationAnalysis:
    """
    Oscillation and stability analysis for the two-player benchmark.

    Parameters
    ----------
    mu        : float  Mean-reversion strength μ > 0.
    eta_bar   : float  Constant reframing rate η̄ ≥ 0.
    kappa_bar : float  Constant resistance rate κ̄ ≥ 0.
    r_bar     : float  Reframing attack rate r̄ ≥ 0.
    params    : Params Full parameter object (optional).
    """

    def __init__(
        self,
        mu: float,
        eta_bar: float,
        kappa_bar: float,
        r_bar: float = 0.5,
        params: Params | None = None,
    ) -> None:
        if mu <= 0:
            raise ValueError(f"mu must be > 0, got {mu}")
        self.mu = mu
        self.eta_bar = eta_bar
        self.kappa_bar = kappa_bar
        self.r_bar = r_bar
        self.params = params or Params()

    def analyse(self) -> OscillationResult:
        """
        Compute the full oscillation analysis.

        Returns
        -------
        OscillationResult
        """
        mu = self.mu
        eta = self.eta_bar
        kappa = self.kappa_bar

        # Characteristic equation: λ² + μλ + η̄κ̄ = 0
        discriminant = mu**2 - 4 * eta * kappa

        # Eigenvalues
        sqrt_disc = np.sqrt(complex(discriminant))
        lam1 = (-mu + sqrt_disc) / 2.0
        lam2 = (-mu - sqrt_disc) / 2.0

        # Classify
        tol = 1e-10
        if abs(discriminant) < tol:
            regime = "critically_damped"
            oscillates = False
            period = None
            decay_rate = float(np.abs(lam1.real))
        elif discriminant < 0:
            regime = "oscillatory"
            oscillates = True
            period = 2 * np.pi / abs(lam1.imag)
            decay_rate = float(abs(lam1.real))
        else:
            regime = "stable_node"
            oscillates = False
            period = None
            decay_rate = float(min(abs(lam1.real), abs(lam2.real)))

        h_star = 0.5 + (eta - kappa * self.r_bar) / mu
        h_star = float(np.clip(h_star, 0.0, 1.0))

        return OscillationResult(
            oscillates=oscillates,
            discriminant=float(discriminant),
            eigenvalues=(complex(lam1), complex(lam2)),
            period=period,
            decay_rate=decay_rate,
            mu=mu,
            eta_bar=eta,
            kappa_bar=kappa,
            h_star=h_star,
            regime=regime,
        )

    def condition_holds(self) -> bool:
        """
        Return True if the oscillation condition μ² < 4η̄κ̄ holds.
        Convenience wrapper around analyse().
        """
        return self.analyse().oscillates

    # ------------------------------------------------------------------
    # Phase portrait data
    # ------------------------------------------------------------------

    def phase_portrait(
        self,
        h_range: tuple[float, float] = (0.0, 1.0),
        n_points: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute ḣ = f(h) across a range of h values.

        Useful for phase portrait plots — plots ḣ vs h and marks h* and h=0.5.

        Returns
        -------
        h_vals : np.ndarray  h values
        dh_vals: np.ndarray  corresponding ḣ = f(h) values
        """
        h_vals = np.linspace(h_range[0], h_range[1], n_points)
        dh_vals = (-self.mu * (h_vals - 0.5)
                   + self.eta_bar
                   - self.kappa_bar * self.r_bar)
        return h_vals, dh_vals

    # ------------------------------------------------------------------
    # Scan over parameter space
    # ------------------------------------------------------------------

    @staticmethod
    def stability_scan(
        mu_values: np.ndarray,
        eta_values: np.ndarray,
        kappa_bar: float = 0.3,
        r_bar: float = 0.5,
    ) -> np.ndarray:
        """
        Compute the oscillation condition over a grid of (μ, η̄) values.

        Returns a boolean array of shape (len(mu_values), len(eta_values))
        where True means the system oscillates.

        Useful for generating phase portraits in (μ, η̄) space.
        """
        oscillates = np.zeros((len(mu_values), len(eta_values)), dtype=bool)
        for i, mu in enumerate(mu_values):
            for j, eta in enumerate(eta_values):
                disc = mu**2 - 4 * eta * kappa_bar
                oscillates[i, j] = disc < 0
        return oscillates
