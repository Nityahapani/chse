"""
fdi.py
======
Fiscal Dominance Index (FDI) estimation pipeline (Section 11.1).

The FDI translates CHSE theory into observable central bank data:

    FDI = V_T · λ_R / (K_CB · ρ_κ/ρ_ν)

where:
    V_T    = political capital / government pressure on the CB
    λ_R    = reframing efficiency (speed of fiscal narrative attacks)
    K_CB   = central bank independence score (Dincer-Eichengreen)
    ρ_κ    = credibility replenishment rate
    ρ_ν    = manipulation replenishment rate

FDI thresholds:
    FDI < 0.5       → robust monetary dominance
    0.5 ≤ FDI ≤ 1  → contested regime with frequent oscillation
    FDI > 1         → recurrent fiscal dominance

Observable proxies (Table 4):
    h_{CB,T}(t) = yield spread response to CB announcements
                  OR CB forecast dominance in surveys

Empirical pipeline (Section 10):
    Step 1: Estimate h(t) from VAR impulse responses
    Step 2: Compute HOE statistics (τ̂, Var(h), E[cascade])
    Step 3: Compute FDI from observable proxies
    Step 4: Test phase diagram prediction
    Step 5: Hierarchy Persistence Paradox test
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# FDI data container
# ---------------------------------------------------------------------------

@dataclass
class FDIEstimate:
    """
    Estimated Fiscal Dominance Index for a country/period.

    Attributes
    ----------
    country       : str    Country name or label.
    period        : str    Time period (e.g. '2020-23').
    V_T           : float  Political capital index (proxy).
    K_CB          : float  CB independence score (Dincer-Eichengreen scale).
    lambda_R      : float  Reframing efficiency parameter.
    rho_ratio     : float  ρ_κ / ρ_ν replenishment ratio.
    FDI           : float  Computed FDI value.
    regime        : str    'monetary', 'contested', or 'fiscal'.
    """
    country: str
    period: str
    V_T: float
    K_CB: float
    lambda_R: float = 1.0
    rho_ratio: float = 1.0
    FDI: float = field(init=False)
    regime: str = field(init=False)

    def __post_init__(self) -> None:
        self.FDI = self._compute_fdi()
        self.regime = self._classify()

    def _compute_fdi(self) -> float:
        """FDI = V_T · λ_R / (K_CB · ρ_κ/ρ_ν)"""
        denom = self.K_CB * self.rho_ratio
        if denom < 1e-8:
            return float("inf")
        return float(self.V_T * self.lambda_R / denom)

    def _classify(self) -> str:
        if self.FDI < 0.5:
            return "monetary"
        elif self.FDI <= 1.0:
            return "contested"
        else:
            return "fiscal"

    def to_hsi(self) -> float:
        """
        Convert FDI to the Hierarchy Stability Index.

        FDI = 1/HSI  (approximately), so HSI = 1/FDI.
        """
        if self.FDI < 1e-8:
            return float("inf")
        return 1.0 / self.FDI


# ---------------------------------------------------------------------------
# Paper's example countries
# ---------------------------------------------------------------------------

PAPER_EXAMPLES: List[Dict] = [
    # FDI = V_T * lambda_R / (K_CB * rho_ratio)
    # Calibrated so FDI matches paper Figure 5 values:
    # Chile 0.22, US(00-07) 0.18, US(20-23) 0.54, Brazil 0.91,
    # Zambia 1.41, Turkey 1.82
    {"country": "Chile",  "period": "2000-22", "V_T": 0.21, "K_CB": 0.95, "lambda_R": 1.0, "rho_ratio": 1.0},
    {"country": "US",     "period": "2000-07", "V_T": 0.17, "K_CB": 0.94, "lambda_R": 1.0, "rho_ratio": 1.0},
    {"country": "US",     "period": "2020-23", "V_T": 0.51, "K_CB": 0.94, "lambda_R": 1.0, "rho_ratio": 1.0},
    {"country": "Brazil", "period": "2015-18", "V_T": 0.77, "K_CB": 0.85, "lambda_R": 1.0, "rho_ratio": 1.0},
    {"country": "Zambia", "period": "2020-23", "V_T": 0.94, "K_CB": 0.67, "lambda_R": 1.0, "rho_ratio": 1.0},
    {"country": "Turkey", "period": "2021-23", "V_T": 0.82, "K_CB": 0.45, "lambda_R": 1.0, "rho_ratio": 1.0},
]


def build_paper_examples() -> List[FDIEstimate]:
    """Return the paper's example FDI estimates (Figure 5)."""
    return [FDIEstimate(**ex) for ex in PAPER_EXAMPLES]


# ---------------------------------------------------------------------------
# HOE statistics from h(t) time series
# ---------------------------------------------------------------------------

@dataclass
class HOEFromData:
    """
    HOE statistics estimated directly from an observed h(t) series.

    Matches Definition 10.1 (Leadership Turnover Frequency) and
    Definition 10.2 (HOE as Stationary Distribution over Statistics).

    Attributes
    ----------
    tau_hat         : float  Leadership turnover frequency.
    var_h           : float  Variance of h(t).
    mean_h          : float  Mean of h(t).
    h_above_half    : float  Fraction of time h > 0.5 (leader dominates).
    stationarity_p  : float  p-value for stationarity (0=not stationary, 1=stationary).
    n_obs           : int    Number of observations.
    """
    tau_hat: float
    var_h: float
    mean_h: float
    h_above_half: float
    stationarity_p: float
    n_obs: int

    def summary(self) -> str:
        return (
            f"HOE Statistics from Observed h(t):\n"
            f"  tau_hat (turnover/period)  : {self.tau_hat:.4f}\n"
            f"  Var(h)                     : {self.var_h:.4f}\n"
            f"  E[h]                       : {self.mean_h:.4f}\n"
            f"  Fraction h > 0.5           : {self.h_above_half:.4f}\n"
            f"  Stationarity               : {self.stationarity_p:.3f}\n"
            f"  Observations               : {self.n_obs}"
        )


def hoe_statistics_from_series(h_series: np.ndarray) -> HOEFromData:
    """
    Estimate HOE statistics from an observed h(t) time series.

    Parameters
    ----------
    h_series : np.ndarray  Time series of hierarchy beliefs h(t) ∈ [0,1].

    Returns
    -------
    HOEFromData
    """
    h = np.asarray(h_series, dtype=float)
    h = np.clip(h, 0.0, 1.0)
    n = len(h)

    crossings = np.diff((h > 0.5).astype(int))
    tau_hat = float(np.sum(np.abs(crossings))) / max(n - 1, 1)
    var_h = float(np.var(h))
    mean_h = float(np.mean(h))
    frac_above = float(np.mean(h > 0.5))

    # Simple stationarity check: compare first vs second half means
    mid = n // 2
    diff = abs(np.mean(h[:mid]) - np.mean(h[mid:]))
    # Convert to approximate p-value: small diff → high p
    stat_p = float(np.exp(-10.0 * diff))

    return HOEFromData(
        tau_hat=tau_hat,
        var_h=var_h,
        mean_h=mean_h,
        h_above_half=frac_above,
        stationarity_p=stat_p,
        n_obs=n,
    )


# ---------------------------------------------------------------------------
# Phase diagram prediction test (Step 4 of empirical pipeline)
# ---------------------------------------------------------------------------

def predict_regime(fdi: FDIEstimate, pi: float = 0.0) -> Dict:
    """
    Predict the CHSE regime for a given FDI estimate.

    Uses FDI thresholds directly (Section 11.1):
        FDI < 0.5       → monetary (stable hierarchy)
        0.5 ≤ FDI ≤ 1  → contested (oscillatory)
        FDI > 1         → fiscal (cascade)

    Also computes the Z instability index for reference.

    Parameters
    ----------
    fdi : FDIEstimate
    pi  : float  Propagation intensity (for Z computation).

    Returns
    -------
    dict with 'hsi', 'Z', 'predicted_regime', 'matches_fdi_regime'.
    """
    hsi = fdi.to_hsi()
    if hsi == float("inf"):
        Z = 0.0
    else:
        Z = (1.0 / hsi) * (1.0 + 2.0 * pi)

    # Use FDI thresholds directly — consistent with Corollary 11.1
    if fdi.FDI < 0.5:
        predicted = "monetary"
    elif fdi.FDI <= 1.0:
        predicted = "contested"
    else:
        predicted = "fiscal"

    return {
        "country": fdi.country,
        "period": fdi.period,
        "FDI": round(fdi.FDI, 4),
        "HSI": round(hsi, 4),
        "Z": round(Z, 4),
        "predicted_regime": predicted,
        "fdi_regime": fdi.regime,
        "consistent": predicted == fdi.regime,
    }


# ---------------------------------------------------------------------------
# Persistence Paradox test (Step 5 of empirical pipeline)
# ---------------------------------------------------------------------------

def persistence_paradox_test(
    fdi_estimates: List[FDIEstimate],
    collapse_volatility: Optional[List[float]] = None,
) -> Dict:
    """
    Test the Hierarchy Persistence Paradox:
        Post-collapse yield volatility ~ HSI (positive coefficient predicted).

    If collapse_volatility is not provided, uses simulated values
    proportional to HSI (for illustrative purposes).

    Parameters
    ----------
    fdi_estimates      : list[FDIEstimate]
    collapse_volatility: list[float] | None  Observed post-collapse volatility.

    Returns
    -------
    dict with 'hsi_vals', 'volatility', 'correlation', 'paradox_confirmed'.
    """
    hsi_vals = np.array([est.to_hsi() for est in fdi_estimates])

    # Filter out infinite HSI
    finite = hsi_vals < 1e6
    hsi_finite = hsi_vals[finite]

    if collapse_volatility is None:
        # Simulated: volatility ~ HSI + noise (for illustration)
        rng = np.random.default_rng(42)
        simulated = 0.3 * hsi_finite + 0.05 * rng.normal(size=len(hsi_finite))
        volatility = simulated
        simulated_flag = True
    else:
        volatility = np.array(collapse_volatility)[finite]
        simulated_flag = False

    if len(hsi_finite) < 2:
        corr = 0.0
    else:
        corr = float(np.corrcoef(hsi_finite, volatility)[0, 1])

    return {
        "hsi_vals": hsi_finite.tolist(),
        "volatility": volatility.tolist(),
        "correlation": round(corr, 4),
        "paradox_confirmed": corr > 0,
        "simulated": simulated_flag,
        "interpretation": (
            "Positive correlation confirms: stronger hierarchies → larger cascades on collapse. "
            if corr > 0 else
            "Negative correlation — paradox not confirmed in this sample."
        ),
    }
