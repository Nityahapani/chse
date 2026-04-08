"""
distortions.py
==============
Welfare analysis — Three Welfare Distortions (Section on Bottleneck 6).

In any interior HOE π*, relative to the social optimum, there are three
distortions:

1. Over-investment in reframing (Mechanism III)
   Each follower ignores the negative externality of reframing on
   adjacent edges (network spillover −β_R · P_R · φ).
   Excess ∝ β_R · Γ / (1 − Γ)

2. Over-investment in commitment resistance
   Each leader ignores the positive externality of reframe-resistant
   commitments to third parties (coordination benefit).
   Excess ∝ Σ_k φ(d(i,k)) · u_k^coordination

3. Under-investment in hierarchy clarity
   h_ij = 1 (or 0) is a public good — reduces ambiguity for all
   actors interacting with i or j.  Because players only internalise
   their own payoff from clarity, hierarchy is more ambiguous than
   socially optimal.
   Under-investment ∝ ζ_II · |E_i| (network-wide ambiguity spillover)

Policy implications:
   - Public commitment requirements (central bank mandates, board
     resolutions) correct distortion 3 (under-investment in clarity).
   - Prohibitions on reframing (legal estoppel, institutional precedent)
     correct distortion 1 (over-investment in reframing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from ..core.primitives import Params
from ..core.network import CHSENetwork
from ..core.mechanisms import reframe_success_prob, reframe_resistance


# ---------------------------------------------------------------------------
# Welfare function
# ---------------------------------------------------------------------------

def total_welfare(
    network: CHSENetwork,
    u_L: float,
    u_F: float,
    investment_costs: Optional[np.ndarray] = None,
) -> float:
    """
    Compute total welfare W = Σ_{i,j} [h_ij·u_i^L + (1−h_ij)·u_j^F] − costs.

    Parameters
    ----------
    network          : CHSENetwork
    u_L              : float  Leadership payoff.
    u_F              : float  Followership payoff.
    investment_costs : np.ndarray | None  Per-edge investment costs.

    Returns
    -------
    float  Total welfare.
    """
    W = 0.0
    for idx, e in enumerate(network.canon_edges):
        i, j = e
        h_ij = network.h[e]
        # i's payoff
        W += h_ij * u_L + (1.0 - h_ij) * u_F
        # j's payoff
        W += (1.0 - h_ij) * u_L + h_ij * u_F

    if investment_costs is not None:
        W -= float(np.sum(investment_costs))

    return W


# ---------------------------------------------------------------------------
# Distortion 1 — over-investment in reframing
# ---------------------------------------------------------------------------

def reframing_distortion(
    network: CHSENetwork,
    params: Params,
    eta_eq: float,
    Gamma: float,
) -> float:
    """
    Distortion 1: over-investment in reframing.

    Excess reframing investment relative to social optimum:

        Excess ∝ β_R · Γ / (1 − Γ)

    The equilibrium reframing rate η_eq is larger than the socially
    optimal η_SO by this amount because each follower ignores the
    negative network spillover of their attack on adjacent edges.

    Parameters
    ----------
    network : CHSENetwork
    params  : Params
    eta_eq  : float  Equilibrium reframing investment.
    Gamma   : float  Spectral propagation factor.

    Returns
    -------
    float  Excess reframing investment (positive → over-investment).
    """
    if Gamma >= 1.0:
        return float("inf")
    excess_factor = params.beta_R * Gamma / (1.0 - Gamma)
    return float(eta_eq * excess_factor)


def social_optimal_eta(
    network: CHSENetwork,
    params: Params,
    eta_eq: float,
    Gamma: float,
) -> float:
    """
    Compute the socially optimal reframing rate η_SO.

    η_SO = η_eq / (1 + β_R · Γ / (1 − Γ))

    The planner internalises the network spillover, leading to lower
    reframing than the decentralised equilibrium.
    """
    if Gamma >= 1.0:
        return 0.0
    excess_factor = params.beta_R * Gamma / (1.0 - Gamma)
    return float(eta_eq / (1.0 + excess_factor))


# ---------------------------------------------------------------------------
# Distortion 2 — over-investment in commitment resistance
# ---------------------------------------------------------------------------

def resistance_distortion(
    network: CHSENetwork,
    params: Params,
    kappa_eq: float,
    u_coordination: float = 0.5,
) -> float:
    """
    Distortion 2: over-investment in commitment resistance.

    Each leader ignores the positive externality of reframe-resistant
    commitments — greater legibility reduces coordination costs for all
    network neighbours.

    Excess ∝ Σ_k φ(d(i,k), G) · u_k^coordination

    Parameters
    ----------
    network         : CHSENetwork
    params          : Params
    kappa_eq        : float  Equilibrium credibility investment.
    u_coordination  : float  Per-neighbour coordination benefit.

    Returns
    -------
    float  Excess commitment resistance investment.
    """
    # Average spillover to network neighbours
    total_spillover = 0.0
    count = 0
    for i in range(network.n_players):
        for k in range(network.n_players):
            if i != k:
                phi = network.distance_decay(i, k, decay_rate=1.0)
                total_spillover += phi * u_coordination
                count += 1

    avg_spillover = total_spillover / max(count, 1)
    return float(kappa_eq * avg_spillover)


def social_optimal_kappa(
    network: CHSENetwork,
    params: Params,
    kappa_eq: float,
    u_coordination: float = 0.5,
) -> float:
    """
    Social optimum for credibility investment (accounting for legibility externality).

    κ_SO = κ_eq + coordination_benefit / c_kappa

    The planner invests more in credibility than the decentralised
    equilibrium because they capture the coordination externality.
    """
    dist = resistance_distortion(network, params, kappa_eq, u_coordination)
    adjustment = dist / max(params.c_kappa, 1e-6)
    return float(kappa_eq + adjustment)


# ---------------------------------------------------------------------------
# Distortion 3 — under-investment in hierarchy clarity
# ---------------------------------------------------------------------------

def clarity_distortion(
    network: CHSENetwork,
    params: Params,
) -> Dict[str, float]:
    """
    Distortion 3: under-investment in hierarchy clarity.

    Hierarchy clarity (h_ij close to 0 or 1) is a public good.
    Under-investment ∝ ζ_II · |E_i| (ambiguity spillover × degree).

    Returns a dict with per-player under-investment magnitudes.

    Parameters
    ----------
    network : CHSENetwork
    params  : Params

    Returns
    -------
    dict  {player_index: under_investment_magnitude}
    """
    result = {}
    for i in range(network.n_players):
        degree_i = len(network.neighbours(i))
        under_investment = params.zeta_II * degree_i
        result[i] = float(under_investment)
    return result


def total_clarity_gap(
    network: CHSENetwork,
    params: Params,
) -> float:
    """
    Total hierarchy clarity gap across the network.

    Measures how far the current belief state is from full clarity
    (h ∈ {0, 1} for all edges), weighted by the spillover rate.

    Total gap = Σ_{edges} ζ_II · (1 − |2h_ij − 1|) · degree_avg
    """
    degree_avg = np.mean([len(network.neighbours(i))
                          for i in range(network.n_players)])
    gap = 0.0
    for e in network.canon_edges:
        h = network.h[e]
        ambiguity = 1.0 - abs(2.0 * h - 1.0)
        gap += params.zeta_II * ambiguity * degree_avg
    return float(gap)


# ---------------------------------------------------------------------------
# Composite distortion report
# ---------------------------------------------------------------------------

@dataclass
class WelfareDistortions:
    """
    Summary of all three welfare distortions at the HOE.

    Attributes
    ----------
    reframing_excess    : float   Excess η investment (Distortion 1).
    eta_SO              : float   Socially optimal η.
    eta_eq              : float   Equilibrium η.
    resistance_excess   : float   Excess κ investment (Distortion 2).
    clarity_gap         : float   Total hierarchy clarity gap (Distortion 3).
    per_player_clarity  : dict    Per-player clarity under-investment.
    total_welfare_eq    : float   Total welfare at equilibrium.
    total_welfare_SO    : float   Total welfare at social optimum (estimated).
    welfare_loss        : float   Welfare loss from distortions.
    policy_implications : list    Suggested policy interventions.
    """
    reframing_excess: float
    eta_SO: float
    eta_eq: float
    resistance_excess: float
    clarity_gap: float
    per_player_clarity: dict
    total_welfare_eq: float
    total_welfare_SO: float
    welfare_loss: float
    policy_implications: List[str]

    def summary(self) -> str:
        lines = [
            "=== Welfare Distortions at HOE ===",
            "",
            "Distortion 1 — Over-investment in reframing:",
            f"  Equilibrium η        : {self.eta_eq:.4f}",
            f"  Social optimum η_SO  : {self.eta_SO:.4f}",
            f"  Excess investment    : {self.reframing_excess:.4f}",
            "",
            "Distortion 2 — Over-investment in commitment resistance:",
            f"  Excess investment    : {self.resistance_excess:.4f}",
            "",
            "Distortion 3 — Under-investment in hierarchy clarity:",
            f"  Total clarity gap    : {self.clarity_gap:.4f}",
            "  Per-player gap       : " +
            ", ".join(f"player {k}: {v:.3f}" for k, v in self.per_player_clarity.items()),
            "",
            f"Total welfare (equilibrium)    : {self.total_welfare_eq:.4f}",
            f"Total welfare (social optimum) : {self.total_welfare_SO:.4f}",
            f"Welfare loss                   : {self.welfare_loss:.4f}",
            "",
            "Policy implications:",
        ] + [f"  • {p}" for p in self.policy_implications]
        return "\n".join(lines)


def compute_welfare_distortions(
    network: CHSENetwork,
    params: Params,
    eta_eq: float,
    kappa_eq: float,
    u_L: float = 10.0,
    u_F: float = 2.0,
    Gamma: float = 0.4,
    u_coordination: float = 0.5,
) -> WelfareDistortions:
    """
    Compute all three welfare distortions and the total welfare loss.

    Parameters
    ----------
    network         : CHSENetwork
    params          : Params
    eta_eq          : float  Equilibrium reframing investment.
    kappa_eq        : float  Equilibrium credibility investment.
    u_L             : float  Leadership payoff.
    u_F             : float  Followership payoff.
    Gamma           : float  Spectral propagation factor.
    u_coordination  : float  Per-neighbour coordination benefit.

    Returns
    -------
    WelfareDistortions
    """
    # Distortion 1
    excess_ref = reframing_distortion(network, params, eta_eq, Gamma)
    eta_so = social_optimal_eta(network, params, eta_eq, Gamma)

    # Distortion 2
    excess_res = resistance_distortion(network, params, kappa_eq, u_coordination)
    kappa_so = social_optimal_kappa(network, params, kappa_eq, u_coordination)

    # Distortion 3
    per_player = clarity_distortion(network, params)
    c_gap = total_clarity_gap(network, params)

    # Welfare at equilibrium
    W_eq = total_welfare(
        network, u_L, u_F,
        investment_costs=np.array([eta_eq + kappa_eq] * len(network.canon_edges))
    )

    # Welfare at social optimum (corrected investments, pushed toward clarity)
    net_so = network.copy()
    for e in net_so.canon_edges:
        h = net_so.h[e]
        h_so = 0.9 if h >= 0.5 else 0.1
        net_so.h[e] = h_so

    W_so = total_welfare(
        net_so, u_L, u_F,
        investment_costs=np.array([eta_so + kappa_so] * len(net_so.canon_edges))
    )

    # Welfare loss = sum of distortion costs (monetised)
    # Distortion 1: excess reframing cost = excess_ref * c_mu per edge
    d1_cost = excess_ref * params.c_mu * len(network.canon_edges)
    # Distortion 2: excess resistance cost = excess_res * c_kappa per edge
    d2_cost = excess_res * params.c_kappa * len(network.canon_edges)
    # Distortion 3: clarity gap cost = c_gap (already weighted by zeta_II)
    d3_cost = c_gap
    welfare_loss = d1_cost + d2_cost + d3_cost

    policy = [
        "Public commitment requirements (central bank mandates, board resolutions) "
        "correct Distortion 3 by making hierarchy clarity a required output.",
        "Prohibitions on reframing (legal estoppel, institutional precedent) "
        "correct Distortion 1 by internalising the network externality of attacks.",
        "Legibility subsidies (transparent announcements, published mandates) "
        "correct Distortion 2 by capturing the coordination benefit of resistance.",
    ]

    return WelfareDistortions(
        reframing_excess=excess_ref,
        eta_SO=eta_so,
        eta_eq=eta_eq,
        resistance_excess=excess_res,
        clarity_gap=c_gap,
        per_player_clarity=per_player,
        total_welfare_eq=W_eq,
        total_welfare_SO=W_so,
        welfare_loss=welfare_loss,
        policy_implications=policy,
    )
