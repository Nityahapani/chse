# CHSE Documentation

**Contested Hierarchy with Social Embedding** — a formal theory of endogenous authority, strategic ambiguity, and contagious hierarchy in dynamic games.

---

## Contents

| Document | What it covers |
|----------|---------------|
| [Quickstart](quickstart.md) | Install, verify, five examples in five minutes |
| [Theory](theory.md) | Complete formal theory — all definitions, theorems, proofs |
| [API Reference](api.md) | Every public class and function, with signatures and examples |
| [Parameters](parameters.md) | All 15 parameters with defaults and economic interpretation |
| [How-to Guides](guides.md) | Concrete recipes for the most common tasks |
| [Design Notes](design.md) | Why things are built the way they are |

---

## The model in thirty seconds

Players are connected in a network G. For each edge {i,j}, the **hierarchy belief** h_ij(t) ∈ [0,1] is the network-wide probability that i leads j. It evolves through four coupled mechanisms:

| Mechanism | Name | Drives |
|-----------|------|--------|
| I | Anticipation as a public good | Bayesian belief updating; signal spillover across network |
| II | Role ambiguity as a strategic instrument | Mean-reversion toward h=0.5 |
| III | Retroactive reframing | Followers attack past commitments; leaders resist |
| IV | Contagious propagation | Belief shifts spread across edges via endogenous kernel K |

Two composite indices carry all predictive content:

```
HSI = λ_κ · K_i / (λ_R · V_j)     — resistance / attack capacity
PI  = Γ · E[φ(d, G)]              — network cascade potential
Z   = HSI⁻¹ · (1 + 2·PI)         — instability index
```

| Z | Regime |
|---|--------|
| Z < 1 | Stable — h(t) converges to fixed point |
| 1 ≤ Z < 2 | Oscillatory — leadership alternates |
| 2 ≤ Z < 3.5 | Cascade-Dominated — network-wide collapses |
| Z ≥ 3.5 | Turbulent — sensitive dependence |

The solution concept is the **Hierarchy Orbit Equilibrium (HOE)**: an invariant measure of the induced Markov process, generalising both Nash (h*=0.5) and Stackelberg (h*=1) as degenerate special cases.

---

## Package map

```
chse/
├── core/           Primitives, mechanisms I–IV, network, simulation API
├── benchmark/      Two-player ODE, oscillation condition, flip time
├── phase/          Phase diagram, Jacobian, cascade analysis
├── equilibrium/    Markov chain engine, HOE estimation, Lyapunov stability
├── welfare/        Three distortions, social optimum, Hierarchy Persistence Paradox
└── empirical/      Fiscal Dominance Index, Figure 5, empirical tests
```

---

## Key results

**Theorem 6.1** — stable iff HSI·(1+2·PI) > 1  
**Theorem 6.2** — cascade iff ρ(K) ≥ 1 − 1/HSI  
**Theorem 8.1** — HOE exists (Krylov-Bogolyubov on compact S)  
**Theorem 8.2** — HOE is Lyapunov stable iff Γ < (1−δ)/(1+δ)  
**Proposition 7.1** — E[cascade size] ≤ α_R/(1−ρ(K)) when ρ(K) < 1  
**Hierarchy Persistence Paradox** — ∂E[cascade|collapse]/∂HSI > 0  
**Corollary 11.1** — fiscal dominance iff FDI = V_T·λ_R/(K_CB·ρ_κ/ρ_ν) > 1  

---

## Running the tests

```bash
pytest tests/ -v                     # 111 tests, all pass
python3 -m unittest discover tests/  # without pytest
```

Python 3.10, 3.11, 3.12 — all supported.
