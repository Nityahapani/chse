"""
kernel.py
=========
Mechanism IV — Contagious Propagation via the Endogenous Kernel (Section 4.4).

The propagation kernel K is endogenous: it depends on trust accumulated
through past anticipation accuracy and network centrality.

    K_{ij→kl}(t) = Acc_ij(t) · Trust_{ij→kl}(t) · φ(d({i,j},{k,l}), G)
                   ─────────────────────────────────────────────────────────
                                    Σ_{(p,q)} [·]

where:
    Acc_ij(t)          — time-averaged anticipation accuracy on edge (i,j)
    Trust_{ij→kl}(t)   — cross-edge trust level, updated each period
    φ(d, G)            — distance-decay function

The total belief update from propagation:

    Δ^IV h_kl(t) = Σ_{(i,j)≠(k,l)} K_{ij→kl} · (Δ^I + Δ^II + Δ^III) h_ij

Key result (Proposition 7.1):
    ρ(K) < 1  →  no infinite cascade; expected cascade size bounded
    ρ(K) ≥ 1  →  cascade possible

Strategic cascade seeding (Proposition 7.2):
    Players can manipulate ρ(K) by investing in accuracy on edges
    adjacent to opponents' fragile relationships.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .network import CHSENetwork, Edge, _canon
from .anticipation import AnticipatState
from .primitives import Params


# ---------------------------------------------------------------------------
# Trust state
# ---------------------------------------------------------------------------

@dataclass
class TrustState:
    """
    Cross-edge trust levels Trust_{ij→kl}(t).

    trust[(e1, e2)] — trust from edge e1 toward edge e2.
    Updated each period based on whether e1's belief updates proved
    useful for predicting e2's belief movements.
    """
    trust: Dict[Tuple[Edge, Edge], float] = field(default_factory=dict)
    decay: float = 0.9   # geometric decay per period

    @classmethod
    def initialise(cls, network: CHSENetwork,
                   initial_trust: float = 0.5,
                   decay: float = 0.9) -> "TrustState":
        """Initialise uniform trust between all edge pairs."""
        state = cls(decay=decay)
        edges = network.canon_edges
        for e1 in edges:
            for e2 in edges:
                if e1 != e2:
                    state.trust[(e1, e2)] = initial_trust
        return state

    def get(self, e1: Edge, e2: Edge) -> float:
        return self.trust.get((e1, e2), 0.5)

    def update(self, e1: Edge, e2: Edge, delta1: float, delta2: float) -> None:
        """
        Update trust from e1 to e2.

        Trust increases when both edges moved in the same direction
        (positive correlation), decreases otherwise.  Bounded in [0, 1].

        Args:
            delta1: belief change on e1 this period
            delta2: belief change on e2 this period
        """
        current = self.get(e1, e2)
        # Correlation signal: +1 if same direction, -1 if opposite
        if abs(delta1) > 1e-8 and abs(delta2) > 1e-8:
            signal = 1.0 if (delta1 * delta2 > 0) else -1.0
        else:
            signal = 0.0
        new_trust = self.decay * current + (1.0 - self.decay) * (0.5 + 0.5 * signal)
        self.trust[(e1, e2)] = float(np.clip(new_trust, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Kernel construction
# ---------------------------------------------------------------------------

def build_kernel(
    network: CHSENetwork,
    ant_state: AnticipatState,
    trust_state: TrustState,
    params: Params,
    decay_rate: float = 1.0,
) -> np.ndarray:
    """
    Build the propagation kernel matrix K.

    K is an |E| × |E| matrix where K[kl, ij] is the weight with which
    a belief shift on edge (i,j) propagates to edge (k,l).

    K_{ij→kl} = Acc_ij · Trust_{ij→kl} · φ(d({i,j},{k,l}), G) / Z

    where Z is the row-normalisation constant and d({i,j},{k,l}) is the
    minimum graph distance between endpoints of the two edges.

    Parameters
    ----------
    network    : CHSENetwork
    ant_state  : AnticipatState  Provides Acc_ij values.
    trust_state: TrustState      Provides Trust_{ij→kl} values.
    params     : Params
    decay_rate : float           Rate parameter for distance decay φ.

    Returns
    -------
    np.ndarray  Shape (|E|, |E|).  K[kl_idx, ij_idx] = K_{ij→kl}.
    """
    edges = network.canon_edges
    n_edges = len(edges)
    edge_idx = {e: idx for idx, e in enumerate(edges)}

    K = np.zeros((n_edges, n_edges))

    for ij_idx, e_ij in enumerate(edges):
        i, j = e_ij
        acc_ij = ant_state.accuracy(i, j)

        for kl_idx, e_kl in enumerate(edges):
            if e_ij == e_kl:
                continue   # diagonal: no self-propagation
            k, l = e_kl

            trust = trust_state.get(e_ij, e_kl)

            # Edge distance: min over endpoint pairs
            d = min(
                network.shortest_path_length(i, k),
                network.shortest_path_length(i, l),
                network.shortest_path_length(j, k),
                network.shortest_path_length(j, l),
            )
            phi = np.exp(-decay_rate * d)

            K[kl_idx, ij_idx] = acc_ij * trust * phi

    # Row normalisation (each row sums to ≤ Γ, the propagation bound)
    Gamma = params.PI / max(network.expected_distance_decay(decay_rate), 1e-8) \
        if params.PI is not None else 0.5

    row_sums = K.sum(axis=1, keepdims=True)
    # Avoid division by zero; scale so max row sum ≤ Gamma
    max_row = row_sums.max()
    if max_row > 1e-10:
        K = K * (Gamma / max_row)

    return K


def spectral_radius(K: np.ndarray) -> float:
    """
    Compute the spectral radius ρ(K) = max |eigenvalue|.

    Parameters
    ----------
    K : np.ndarray  Square matrix.

    Returns
    -------
    float  Spectral radius.
    """
    if K.size == 0:
        return 0.0
    eigenvalues = np.linalg.eigvals(K)
    return float(np.max(np.abs(eigenvalues)))


def expected_cascade_size(rho_K: float, alpha_R: float) -> float:
    """
    Expected cascade size bound (Proposition 7.1):

        E[cascade size] ≤ α_R / (1 − ρ(K))   when ρ(K) < 1

    Returns inf when ρ(K) ≥ 1 (unbounded cascade possible).

    Parameters
    ----------
    rho_K   : float  Spectral radius of K.
    alpha_R : float  Direct belief drop per successful reframe.

    Returns
    -------
    float  Expected cascade size bound.
    """
    if rho_K >= 1.0:
        return float("inf")
    return alpha_R / (1.0 - rho_K)


# ---------------------------------------------------------------------------
# Propagation update (Mechanism IV)
# ---------------------------------------------------------------------------

def mechanism_IV_update(
    K: np.ndarray,
    direct_deltas: Dict[Edge, float],
    network: CHSENetwork,
) -> Dict[Edge, float]:
    """
    Apply the propagation kernel to compute Δ^IV h for all edges.

        Δ^IV h_kl = Σ_{(i,j)≠(k,l)} K_{ij→kl} · (Δ^I + Δ^II + Δ^III) h_ij

    Parameters
    ----------
    K             : np.ndarray  Propagation kernel (|E| × |E|).
    direct_deltas : dict        {edge: delta_h} from Mechanisms I+II+III.
    network       : CHSENetwork

    Returns
    -------
    dict  {edge: delta_h_IV} — propagation contribution per edge.
    """
    edges = network.canon_edges
    edge_idx = {e: idx for idx, e in enumerate(edges)}

    # Assemble direct delta vector
    delta_vec = np.array([direct_deltas.get(e, 0.0) for e in edges])

    # Propagated contribution: K @ delta_vec
    prop_vec = K @ delta_vec

    return {e: float(prop_vec[edge_idx[e]]) for e in edges}


# ---------------------------------------------------------------------------
# Fragility
# ---------------------------------------------------------------------------

def edge_fragility(network: CHSENetwork) -> Dict[Edge, float]:
    """
    Compute fragility_j(h) for each edge (Section on Bottleneck 5).

    fragility(e) = 1 − |2·h_ij − 1|

    Fragility is 1 when h = 0.5 (maximum ambiguity) and 0 when h ∈ {0,1}.
    High-fragility edges are close to a leadership flip.

    Returns
    -------
    dict  {edge: fragility} in [0, 1].
    """
    result = {}
    for e in network.canon_edges:
        h = network.h[e]
        result[e] = float(1.0 - abs(2.0 * h - 1.0))
    return result


def optimal_cascade_seed(
    K: np.ndarray,
    network: CHSENetwork,
    attacker: int,
) -> Dict[Edge, float]:
    """
    Compute the optimal accuracy allocation for cascade seeding
    (Proposition on Optimal Cascade Seeding, Bottleneck 5).

    Player i (attacker) maximises ∂ρ(K)/∂Acc_ij · fragility_j(h)
    over all edges adjacent to i.

    The sensitivity ∂ρ(K)/∂Acc_ij equals the (ij, ij) element of the
    left-right eigenvector outer product at the dominant eigenvalue.

    Returns a dict {edge: weight} — relative accuracy allocation,
    normalised to sum to 1.

    Parameters
    ----------
    K        : np.ndarray  Current propagation kernel.
    network  : CHSENetwork
    attacker : int         Player index whose cascade strategy we compute.

    Returns
    -------
    dict  {edge: weight} for edges incident to attacker.
    """
    edges = network.canon_edges
    edge_idx = {e: idx for idx, e in enumerate(edges)}
    fragility = edge_fragility(network)

    # Dominant eigenvector sensitivity
    if K.size == 0 or spectral_radius(K) < 1e-10:
        # Trivial: uniform allocation
        adj = [e for e in edges if attacker in e]
        w = 1.0 / len(adj) if adj else 0.0
        return {e: w for e in adj}

    eigenvalues = np.linalg.eigvals(K)
    dom_idx = int(np.argmax(np.abs(eigenvalues)))
    # Right eigenvector
    _, vr = np.linalg.eig(K)
    r_vec = np.abs(vr[:, dom_idx])
    # Left eigenvector
    _, vl = np.linalg.eig(K.T)
    l_vec = np.abs(vl[:, dom_idx])

    # Sensitivity: outer product diagonal
    sensitivity = l_vec * r_vec

    # Weight = sensitivity * fragility for edges incident to attacker
    adj_edges = [e for e in edges if attacker in e]
    weights = {}
    for e in adj_edges:
        idx = edge_idx[e]
        opponent = e[1] if e[0] == attacker else e[0]
        frag = fragility.get(e, 0.0)
        weights[e] = float(sensitivity[idx] * frag)

    total = sum(weights.values())
    if total > 1e-10:
        weights = {e: w / total for e, w in weights.items()}
    else:
        w = 1.0 / len(adj_edges) if adj_edges else 0.0
        weights = {e: w for e in adj_edges}

    return weights
