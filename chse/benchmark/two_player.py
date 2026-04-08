"""
two_player.py
=============
The two-player benchmark model (Section 3 of the paper).

In the deterministic limit (suppressing stochastic anticipation and network
propagation), the hierarchy belief evolves as:

    ḣ(t) = −μ · (h(t) − 1/2) + η(t) − κ(t) · r(t)

where:
    μ   : mean-reversion strength (composite of μ_II and ambiguity)
    η   : follower j's reframing investment rate
    κ·r : leader i's commitment capital deployed as resistance

With constant η̄, κ̄, r̄ the system has a fixed point at:

    h* = 1/2 + (η̄ − κ̄·r̄) / (2μ)

The class TwoPlayerModel supports:
  - Analytical fixed-point computation
  - Numerical ODE integration (scipy solve_ivp)
  - Optimal best-response trajectories (η*, κ* from mechanisms.py)
  - Phase-portrait generation data
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal
import numpy as np
from scipy.integrate import solve_ivp

from ..core.primitives import Params
from ..core.mechanisms import optimal_eta, optimal_kappa_spend


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TwoPlayerResult:
    """
    Output of a two-player benchmark simulation.

    Attributes
    ----------
    t      : np.ndarray  Time points.
    h      : np.ndarray  Hierarchy belief h(t) at each time point.
    eta    : np.ndarray  Follower's reframing investment η(t).
    kappa  : np.ndarray  Leader's resistance investment κ*(t).
    h_star : float       Fixed-point belief h*.
    params : Params      Parameters used.
    mode   : str         'constant' or 'best_response'.
    """
    t: np.ndarray
    h: np.ndarray
    eta: np.ndarray
    kappa: np.ndarray
    h_star: float
    params: Params
    mode: str

    @property
    def final_h(self) -> float:
        return float(self.h[-1])

    @property
    def leadership_periods(self) -> np.ndarray:
        """Boolean array: True when player 1 leads (h > 0.5)."""
        return self.h > 0.5

    @property
    def turnover_count(self) -> int:
        """Number of leadership flips (h crosses 0.5)."""
        crossings = np.diff((self.h > 0.5).astype(int))
        return int(np.sum(np.abs(crossings)))

    @property
    def turnover_frequency(self) -> float:
        """Flips per unit time."""
        if len(self.t) < 2:
            return 0.0
        duration = self.t[-1] - self.t[0]
        return self.turnover_count / duration if duration > 0 else 0.0


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class TwoPlayerModel:
    """
    Two-player CHSE benchmark.

    Parameters
    ----------
    mu : float
        Mean-reversion strength μ > 0.  Composite of μ_II and the ambiguity
        strength.  Controls how fast h reverts toward 0.5 absent other forces.
    eta_bar : float
        Constant reframing rate η̄ (follower's investment, when mode='constant').
    kappa_bar : float
        Constant resistance rate κ̄ (leader's capital, when mode='constant').
    r_bar : float
        Constant reframing-attack rate r̄ ≥ 0.
    h0 : float
        Initial hierarchy belief h(0) ∈ [0, 1].  Default 0.75 (player 1 leads).
    params : Params
        Full parameter object (needed for best-response mode).

    Examples
    --------
    Stable regime (HSI = 2.1):

        >>> from chse.benchmark import TwoPlayerModel
        >>> from chse.core import Params
        >>> p = Params(HSI=2.1)
        >>> model = TwoPlayerModel(mu=1.0, eta_bar=0.3, kappa_bar=0.8,
        ...                        r_bar=0.5, h0=0.75, params=p)
        >>> result = model.integrate(T=80)
        >>> result.turnover_count
        ...  # small number, h stays near h*
    """

    def __init__(
        self,
        mu: float,
        eta_bar: float,
        kappa_bar: float,
        r_bar: float,
        h0: float = 0.75,
        params: Params | None = None,
    ) -> None:
        if mu <= 0:
            raise ValueError(f"mu must be > 0, got {mu}")
        if not 0.0 <= h0 <= 1.0:
            raise ValueError(f"h0 must be in [0, 1], got {h0}")

        self.mu = mu
        self.eta_bar = eta_bar
        self.kappa_bar = kappa_bar
        self.r_bar = r_bar
        self.h0 = h0
        self.params = params or Params()

    # ------------------------------------------------------------------
    # Fixed-point analysis
    # ------------------------------------------------------------------

    def fixed_point(self) -> float:
        """
        Analytical fixed point of the constant-parameter system.

        Setting ḣ = 0 in the ODE:
            −μ(h* − ½) + η̄ − κ̄·r̄ = 0
            h* = ½ + (η̄ − κ̄·r̄) / μ

        Note: the paper's Section 3.1 expresses the fixed point using a
        linearised two-dimensional system which introduces a factor of 2μ in
        the denominator.  This implementation uses the direct ODE fixed point
        consistent with the scalar form ḣ = −μ(h−½) + η̄ − κ̄r̄.

        Clipped to [0, 1].
        """
        h_star = 0.5 + (self.eta_bar - self.kappa_bar * self.r_bar) / self.mu
        return float(np.clip(h_star, 0.0, 1.0))

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------

    def _rhs_constant(self, t: float, y: list[float]) -> list[float]:
        """
        RHS for constant η̄, κ̄, r̄:

            ḣ = −μ(h − 1/2) + η̄ − κ̄·r̄
        """
        h = float(np.clip(y[0], 0.0, 1.0))
        dh = -self.mu * (h - 0.5) + self.eta_bar - self.kappa_bar * self.r_bar
        return [dh]

    def _rhs_best_response(self, t: float, y: list[float]) -> list[float]:
        """
        RHS using optimal best-response functions η*(h) and κ*(h)
        (Bottleneck 2 derivations in the paper).

            ḣ = −μ(h − 1/2) + η*(h) − κ*(h)·r̄
        """
        h = float(np.clip(y[0], 0.0, 1.0))
        eta_star = optimal_eta(h, self.params)
        kappa_star = optimal_kappa_spend(h, self.params)
        dh = -self.mu * (h - 0.5) + eta_star - kappa_star * self.r_bar
        return [dh]

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def integrate(
        self,
        T: float = 80.0,
        n_points: int = 800,
        mode: Literal["constant", "best_response"] = "constant",
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ) -> TwoPlayerResult:
        """
        Numerically integrate the two-player ODE from t=0 to t=T.

        Parameters
        ----------
        T        : float  Total simulation time.
        n_points : int    Number of output time points.
        mode     : str    'constant' uses η̄, κ̄, r̄ directly.
                          'best_response' uses η*(h), κ*(h) from the paper.
        rtol, atol : float  ODE solver tolerances.

        Returns
        -------
        TwoPlayerResult
        """
        t_eval = np.linspace(0.0, T, n_points)
        rhs = (self._rhs_constant if mode == "constant"
               else self._rhs_best_response)

        sol = solve_ivp(
            fun=rhs,
            t_span=(0.0, T),
            y0=[self.h0],
            t_eval=t_eval,
            method="RK45",
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")

        h_traj = np.clip(sol.y[0], 0.0, 1.0)

        # Record the investment trajectories alongside h
        if mode == "constant":
            eta_traj = np.full_like(h_traj, self.eta_bar)
            kappa_traj = np.full_like(h_traj, self.kappa_bar)
        else:
            eta_traj = np.array([optimal_eta(h, self.params) for h in h_traj])
            kappa_traj = np.array([optimal_kappa_spend(h, self.params) for h in h_traj])

        return TwoPlayerResult(
            t=sol.t,
            h=h_traj,
            eta=eta_traj,
            kappa=kappa_traj,
            h_star=self.fixed_point(),
            params=self.params,
            mode=mode,
        )

    # ------------------------------------------------------------------
    # Convenience: run all three HSI regimes (reproduces Figure 2)
    # ------------------------------------------------------------------

    def integrate_stochastic(
        self,
        T: int = 80,
        noise_std: float = 0.05,
        seed: int = 42,
    ) -> TwoPlayerResult:
        """
        Discrete-time stochastic simulation of the two-player benchmark.

        Implements the stochastic analogue of the paper's belief update
        (Section 5.1) for a single edge, using the simplified dynamics:

            h(t+1) = Proj[0,1]{ h(t) − μ(h(t)−½) + η̄ − κ̄·r̄ + ε(t) }

        where ε(t) ~ N(0, noise_std²) represents the stochastic component
        from Mechanism I (anticipation noise) and other shocks.

        Parameters
        ----------
        T         : int    Number of discrete periods.
        noise_std : float  Standard deviation of the per-period noise.
        seed      : int    Random seed.

        Returns
        -------
        TwoPlayerResult
        """
        rng = np.random.default_rng(seed)
        t_arr = np.arange(T + 1, dtype=float)
        h_arr = np.zeros(T + 1)
        h_arr[0] = self.h0

        drift = self.eta_bar - self.kappa_bar * self.r_bar

        for t in range(T):
            h = h_arr[t]
            dh = -self.mu * (h - 0.5) + drift + rng.normal(0.0, noise_std)
            h_arr[t + 1] = float(np.clip(h + dh, 0.0, 1.0))

        eta_arr = np.full(T + 1, self.eta_bar)
        kappa_arr = np.full(T + 1, self.kappa_bar)

        return TwoPlayerResult(
            t=t_arr,
            h=h_arr,
            eta=eta_arr,
            kappa=kappa_arr,
            h_star=self.fixed_point(),
            params=self.params,
            mode="stochastic",
        )

    @classmethod
    def figure2_regimes(
        cls,
        T: int = 80,
    ) -> dict[str, TwoPlayerResult]:
        """
        Run the three canonical HSI regimes reproducing Figure 2 of the paper.

        Returns a dict with keys 'stable', 'oscillatory', 'cascade', each
        a TwoPlayerResult from a discrete stochastic simulation.

        Regime calibration
        ------------------
        The three panels differ in:
          - μ  : mean-reversion strength (higher → faster convergence to h*)
          - η̄,κ̄: balance determines h* (h* = ½ + (η̄ − κ̄r̄)/μ)
          - noise_std: stochastic shock amplitude

        Stable (HSI≈2.1):
            μ=2.0, η̄=0.8, κ̄=0.2, r̄=1.0 → h*≈0.65
            disc=μ²−4η̄κ̄=3.36>0 (stable node); small noise → 0 flips.

        Oscillatory (HSI≈1.0):
            μ=0.6, η̄=0.4, κ̄=0.4, r̄=1.0 → h*=0.5
            disc=0.36−0.64<0 (complex eigenvalues); moderate noise → periodic flips.

        Cascade (HSI≈0.4):
            μ=0.3, η̄=0.4, κ̄=0.4, r̄=1.0 → h*=0.5
            disc<0; large noise → frequent, large-amplitude flips.
        """
        configs = {
            "stable": dict(
                mu=2.0, eta_bar=0.8, kappa_bar=0.2, r_bar=1.0,
                params=Params(HSI=2.1),
                noise_std=0.03,
                h0=0.75,
            ),
            "oscillatory": dict(
                mu=0.6, eta_bar=0.4, kappa_bar=0.4, r_bar=1.0,
                params=Params(HSI=1.0),
                noise_std=0.10,   # wider noise → wider amplitude
                h0=0.50,          # start at h* so noise drives both directions
            ),
            "cascade": dict(
                mu=0.3, eta_bar=0.4, kappa_bar=0.4, r_bar=1.0,
                params=Params(HSI=0.4),
                noise_std=0.15,
                h0=0.75,
            ),
        }
        results = {}
        for label, cfg in configs.items():
            noise_std = cfg.pop("noise_std")
            h0 = cfg.pop("h0")
            model = cls(h0=h0, **cfg)
            results[label] = model.integrate_stochastic(T=T, noise_std=noise_std)
        return results
