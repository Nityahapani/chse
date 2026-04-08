"""
primitives.py
=============
Core dataclasses for the CHSE framework.

Implements the primitives from Section 2 of the paper:
  - Params        : all 15 raw model parameters, plus the two composite indices HSI and PI
  - CapitalStocks : credibility capital κ and manipulation capital μ per player
  - CHSEState     : the full state (h, κ, μ) at a single point in time

In the two-player benchmark (Phase 1) the network has N = {1, 2}, E = {(1,2)},
and we write h for h_12.  The coherence constraint h_12 + h_21 = 1 means we
only need to track one number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class Params:
    """
    Full parameter set for CHSE.

    Raw parameters (Appendix A of the paper)
    -----------------------------------------
    lambda_kappa : float
        Resistance efficiency — converts credibility capital spend into
        reframe-resistance ρ.  HSI numerator factor.
    lambda_R : float
        Reframing efficiency — converts narrative spend η into attack
        success probability P_R.  HSI denominator factor.
    lambda_sigma : float
        Suppression efficiency — converts opacity investment δ into
        signal suppression σ(δ) = 1 − exp(−λ_σ δ).
    K_cap : float
        Hard cap on credibility capital κ_i.
    M_cap : float
        Hard cap on manipulation capital μ_i.
    rho_kappa : float
        Per-period replenishment rate for credibility capital.
    rho_mu : float
        Per-period replenishment rate for manipulation capital.
    mu_II : float
        Ambiguity mean-reversion strength (Mechanism II).
    zeta_II : float
        Network spillover rate for role ambiguity.
    alpha_I : float
        Anticipation impact on h (Mechanism I).
    beta_I : float
        Predictability penalty (Mechanism I).
    alpha_R : float
        Direct belief drop on successful reframe (Mechanism III).
    beta_R : float
        Network spillover factor for successful reframe.
    delta_kappa : float
        Credibility capital lost per successful reframe.
    discount : float
        Discount factor δ ∈ (0, 1).

    Composite indices (Definition 2.2 and 2.3)
    -------------------------------------------
    These are derived from the raw parameters and a given state, but we
    accept them as direct inputs when using the two-player benchmark in
    closed form.  They are the two numbers that carry all predictive content.

    HSI : float | None
        Hierarchy Stability Index = λ_κ · K_i / (λ_R · V_j).
        Provide directly to override the per-state calculation.
    PI : float | None
        Propagation Intensity = Γ · E[φ(d, G)].
        Provide directly; requires a network for full calculation.

    Cost parameters (used in best-response derivations)
    ----------------------------------------------------
    c_mu : float
        Marginal cost of reframing investment η.
    c_kappa : float
        Marginal cost of credibility investment c.
    """

    # --- raw efficiency parameters ---
    lambda_kappa: float = 1.0
    lambda_R: float = 1.0
    lambda_sigma: float = 1.0

    # --- capital caps and replenishment ---
    K_cap: float = 10.0
    M_cap: float = 10.0
    rho_kappa: float = 0.31   # incommensurate with rho_mu → aperiodic chain (Prop. on Ergodicity)
    rho_mu: float = 0.5

    # --- mechanism-specific parameters ---
    mu_II: float = 1.0        # ambiguity mean-reversion (Mech II)
    zeta_II: float = 0.3      # ambiguity network spillover (Mech II)
    alpha_I: float = 0.2      # anticipation h-impact (Mech I)
    beta_I: float = 0.1       # predictability penalty (Mech I)
    alpha_R: float = 0.3      # reframe direct belief drop (Mech III)
    beta_R: float = 0.1       # reframe network spillover (Mech III)
    delta_kappa: float = 0.5  # credibility loss per reframe (Mech III)

    # --- discount ---
    discount: float = 0.95

    # --- composite indices (optional override) ---
    HSI: float | None = None
    PI: float | None = None

    # --- cost parameters for best-response functions ---
    c_mu: float = 0.5
    c_kappa: float = 0.5

    def hsi(self, K_i: float, V_j: float) -> float:
        """
        Compute the Hierarchy Stability Index for a given leader i and
        follower j.

        HSI_ij = λ_κ · K_i / (λ_R · V_j)

        If a fixed HSI was set at construction, return that instead.
        """
        if self.HSI is not None:
            return self.HSI
        if V_j == 0:
            return float("inf")
        return (self.lambda_kappa * K_i) / (self.lambda_R * V_j)

    def pi(self, Gamma: float = 0.0, expected_phi: float = 0.0) -> float:
        """
        Compute Propagation Intensity.

        PI = Γ · E[φ(d, G)]

        In the two-player benchmark there is no network, so Gamma = 0
        and PI = 0 by default.  Supply Gamma and expected_phi for
        multi-player networks.
        """
        if self.PI is not None:
            return self.PI
        return Gamma * expected_phi

    def instability_index(self, K_i: float, V_j: float,
                          Gamma: float = 0.0,
                          expected_phi: float = 0.0) -> float:
        """
        Instability Index Z = HSI⁻¹ · (1 + 2·PI).

        Regime thresholds (Definition 6.1):
          Z < 1          → Stable Hierarchy
          1 ≤ Z < 2      → Oscillatory Hierarchy
          2 ≤ Z < 3.5    → Cascade-Dominated
          Z ≥ 3.5        → Turbulent / high-sensitivity
        """
        hsi_val = self.hsi(K_i, V_j)
        pi_val = self.pi(Gamma, expected_phi)
        if hsi_val == float("inf"):
            return 0.0
        return (1.0 / hsi_val) * (1.0 + 2.0 * pi_val)

    def regime(self, K_i: float, V_j: float,
               Gamma: float = 0.0,
               expected_phi: float = 0.0) -> str:
        """Return the qualitative regime label for these parameters."""
        Z = self.instability_index(K_i, V_j, Gamma, expected_phi)
        if Z < 1:
            return "stable"
        elif Z < 2:
            return "oscillatory"
        elif Z < 3.5:
            return "cascade"
        else:
            return "turbulent"


# ---------------------------------------------------------------------------
# Capital stocks
# ---------------------------------------------------------------------------

@dataclass
class CapitalStocks:
    """
    Per-player resource stocks.

    Attributes
    ----------
    kappa : float
        Credibility capital κ_i ∈ [0, K_cap].
        Funds commitment resistance (Mechanism III) and reframe-resistant
        announcements.
    mu : float
        Manipulation capital μ_i ∈ [0, M_cap].
        Funds role ambiguity (Mech II), anticipation suppression (Mech I),
        and reframing attacks (Mech III).
    """
    kappa: float = 5.0
    mu: float = 5.0

    def deplete_kappa(self, amount: float, params: Params) -> "CapitalStocks":
        """Return a new CapitalStocks with κ reduced by amount (floored at 0)."""
        new_kappa = max(0.0, self.kappa - amount)
        return CapitalStocks(kappa=new_kappa, mu=self.mu)

    def deplete_mu(self, amount: float, params: Params) -> "CapitalStocks":
        """Return a new CapitalStocks with μ reduced by amount (floored at 0)."""
        new_mu = max(0.0, self.mu - amount)
        return CapitalStocks(kappa=self.kappa, mu=new_mu)

    def replenish(self, params: Params,
                  kappa_depleted: float = 0.0,
                  mu_depleted: float = 0.0) -> "CapitalStocks":
        """
        Apply one period of replenishment (Section 5.1).

        κ(t+1) = min(K_cap,  κ(t) + ρ_κ − depleted_κ)
        μ(t+1) = min(M_cap,  μ(t) + ρ_μ − depleted_μ)
        """
        new_kappa = min(params.K_cap,
                        self.kappa + params.rho_kappa - kappa_depleted)
        new_mu = min(params.M_cap,
                     self.mu + params.rho_mu - mu_depleted)
        return CapitalStocks(
            kappa=max(0.0, new_kappa),
            mu=max(0.0, new_mu),
        )


# ---------------------------------------------------------------------------
# Full CHSE state (two-player version)
# ---------------------------------------------------------------------------

@dataclass
class CHSEState:
    """
    Complete state for the two-player CHSE game at a single time step.

    In the two-player benchmark:
      h      : float — h_12(t), the probability player 1 leads player 2.
                       Coherence: h_21 = 1 - h.
      stocks : Dict[int, CapitalStocks] — keyed by player index {1, 2}.
      t      : int   — period index.

    The state space S = H × [0,K_1] × [0,M_1] × [0,K_2] × [0,M_2]
    is compact, ensuring existence of the HOE invariant measure.
    """
    h: float
    stocks: Dict[int, CapitalStocks] = field(default_factory=lambda: {
        1: CapitalStocks(),
        2: CapitalStocks(),
    })
    t: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.h <= 1.0:
            raise ValueError(f"h must be in [0, 1], got {self.h}")

    @property
    def h_21(self) -> float:
        """Coherence: h_21 = 1 − h_12."""
        return 1.0 - self.h

    def leader(self) -> int:
        """Return the current leader (player with h > 0.5)."""
        return 1 if self.h >= 0.5 else 2

    def stage_payoff(self, u_L: float, u_F: float, player: int) -> float:
        """
        Stage game payoff U_i(h) = h_ij · u_i^L + (1 − h_ij) · u_i^F.

        Parameters
        ----------
        u_L : float  Leadership payoff u_i^L
        u_F : float  Followership payoff u_i^F  (u_L > u_F required)
        player : int  1 or 2
        """
        if player == 1:
            return self.h * u_L + (1 - self.h) * u_F
        else:
            return self.h_21 * u_L + (1 - self.h_21) * u_F

    def project(self) -> "CHSEState":
        """Project h back into [0, 1] (in case of floating point drift)."""
        return CHSEState(
            h=float(np.clip(self.h, 0.0, 1.0)),
            stocks=self.stocks,
            t=self.t,
        )
