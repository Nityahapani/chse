"""
network.py
==========
Network structure and belief state for the n-player CHSE game.

The network G = (N, E) is an undirected graph.  For each undirected
edge {i,j} we track TWO directed beliefs:
    h_ij  — probability that i leads j
    h_ji  — probability that j leads i
Coherence: h_ij + h_ji = 1 at all times.

We store only h_ij for the canonical direction (i < j) and derive h_ji
on demand.  The belief matrix is a dict keyed by (i, j) with i < j.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterator, List, Set, Tuple
import numpy as np


Edge = Tuple[int, int]   # always (i, j) with i < j


def _canon(i: int, j: int) -> Edge:
    """Return the canonical (smaller, larger) edge tuple."""
    return (i, j) if i < j else (j, i)


@dataclass
class CHSENetwork:
    """
    The CHSE network: nodes, edges, and the hierarchy belief matrix.

    Parameters
    ----------
    n_players : int
        Number of players.
    edges : list of (int, int)
        Undirected edges.  Order within each pair does not matter.
    initial_h : dict | None
        Initial belief h_ij for each canonical edge (i, j) with i < j.
        Defaults to 0.75 for all edges (player with smaller index leads).
    """

    n_players: int
    edges: List[Edge]
    initial_h: Dict[Edge, float] | None = None

    # Canonical edge set (i < j)
    _canon_edges: List[Edge] = field(init=False, repr=False)
    # Belief matrix h[(i,j)] = P(i leads j), with i < j
    h: Dict[Edge, float] = field(init=False)

    def __post_init__(self) -> None:
        seen: Set[FrozenSet[int]] = set()
        self._canon_edges = []
        for (a, b) in self.edges:
            if a == b:
                raise ValueError(f"Self-loop not allowed: ({a}, {b})")
            key = frozenset({a, b})
            if key in seen:
                raise ValueError(f"Duplicate edge: {{{a}, {b}}}")
            seen.add(key)
            self._canon_edges.append(_canon(a, b))

        # Initialise beliefs
        self.h = {}
        for e in self._canon_edges:
            if self.initial_h and e in self.initial_h:
                v = self.initial_h[e]
            elif self.initial_h:
                # try reversed key
                rev = (e[1], e[0])
                v = 1.0 - self.initial_h[rev] if rev in self.initial_h else 0.75
            else:
                v = 0.75
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"h must be in [0,1], got {v} for edge {e}")
            self.h[e] = float(v)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def canon_edges(self) -> List[Edge]:
        return self._canon_edges

    def belief(self, i: int, j: int) -> float:
        """
        Return h_ij — probability that i leads j.

        Handles both orderings via the coherence constraint.
        """
        e = _canon(i, j)
        if e == (i, j):
            return self.h[e]
        else:
            return 1.0 - self.h[e]

    def set_belief(self, i: int, j: int, value: float) -> None:
        """Set h_ij, updating the canonical entry and enforcing [0,1]."""
        value = float(np.clip(value, 0.0, 1.0))
        e = _canon(i, j)
        if e == (i, j):
            self.h[e] = value
        else:
            self.h[e] = 1.0 - value

    def neighbours(self, i: int) -> List[int]:
        """Return all j such that {i,j} is an edge."""
        result = []
        for (a, b) in self._canon_edges:
            if a == i:
                result.append(b)
            elif b == i:
                result.append(a)
        return result

    def leader_on_edge(self, i: int, j: int) -> int:
        """Return the player currently assigned leadership on edge {i,j}."""
        return i if self.belief(i, j) >= 0.5 else j

    def belief_vector(self) -> np.ndarray:
        """
        Return beliefs as a flat numpy array in canonical edge order.
        Shape: (|E|,)
        """
        return np.array([self.h[e] for e in self._canon_edges])

    def set_belief_vector(self, v: np.ndarray) -> None:
        """Set beliefs from a flat array in canonical edge order."""
        v = np.clip(v, 0.0, 1.0)
        for e, val in zip(self._canon_edges, v):
            self.h[e] = float(val)

    def copy(self) -> "CHSENetwork":
        """Return a deep copy of the network with the same beliefs."""
        new = CHSENetwork.__new__(CHSENetwork)
        new.n_players = self.n_players
        new.edges = list(self.edges)
        new.initial_h = None
        new._canon_edges = list(self._canon_edges)
        new.h = dict(self.h)
        return new

    # ------------------------------------------------------------------
    # Graph metrics
    # ------------------------------------------------------------------

    def shortest_path_length(self, i: int, j: int) -> int:
        """
        BFS shortest path length between nodes i and j (in hops).
        Returns n_players (infinity proxy) if disconnected.
        """
        if i == j:
            return 0
        visited = {i}
        queue = [(i, 0)]
        while queue:
            node, dist = queue.pop(0)
            for nb in self.neighbours(node):
                if nb == j:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        return self.n_players  # disconnected

    def distance_decay(self, i: int, j: int,
                       decay_rate: float = 1.0) -> float:
        """
        φ(d(i,j), G) = exp(−decay_rate · d(i,j)).

        Used in Mechanism I network spillover and PI computation.
        """
        d = self.shortest_path_length(i, j)
        return float(np.exp(-decay_rate * d))

    def expected_distance_decay(self, decay_rate: float = 1.0) -> float:
        """
        E[φ(d, G)] — average distance decay over all ordered node pairs
        (i, j) with i ≠ j.  Used in the PI formula.
        """
        n = self.n_players
        if n <= 1:
            return 0.0
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    total += self.distance_decay(i, j, decay_rate)
                    count += 1
        return total / count if count > 0 else 0.0

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def complete(cls, n: int, initial_h: float = 0.75) -> "CHSENetwork":
        """Complete graph K_n with uniform initial beliefs."""
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        init = {(i, j): initial_h for (i, j) in edges}
        return cls(n_players=n, edges=edges, initial_h=init)

    @classmethod
    def path(cls, n: int, initial_h: float = 0.75) -> "CHSENetwork":
        """Path graph 1-2-...-n with uniform initial beliefs."""
        edges = [(i, i + 1) for i in range(n - 1)]
        init = {(i, j): initial_h for (i, j) in edges}
        return cls(n_players=n, edges=edges, initial_h=init)

    @classmethod
    def star(cls, n: int, initial_h: float = 0.75) -> "CHSENetwork":
        """Star graph: node 0 connected to all others."""
        edges = [(0, i) for i in range(1, n)]
        init = {(i, j): initial_h for (i, j) in edges}
        return cls(n_players=n, edges=edges, initial_h=init)

    @classmethod
    def two_player(cls, h0: float = 0.75) -> "CHSENetwork":
        """The canonical two-player single-edge benchmark network."""
        return cls(
            n_players=2,
            edges=[(0, 1)],
            initial_h={(0, 1): h0},
        )

    def __repr__(self) -> str:
        return (f"CHSENetwork(n={self.n_players}, "
                f"|E|={len(self._canon_edges)}, "
                f"h={self.h})")
