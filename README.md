# CHSE — Contested Hierarchy with Social Embedding

A complete Python implementation of the formal theory from:

> *Contested Hierarchy with Social Embedding: A Formal Theory of Endogenous
> Authority, Strategic Ambiguity, and Contagious Hierarchy in Dynamic Games*
> — Working Paper v2.0, April 2025.

[![CI](https://github.com/Nityahapani/chse/actions/workflows/ci.yml/badge.svg)](https://github.com/Nityahapani/chse/actions)

---

## Quickstart

```bash
git clone https://github.com/Nityahapani/chse.git
cd chse
pip install -e ".[dev]"
jupyter notebook notebooks/01_two_player.ipynb
```

Python 3.10+. Dependencies: `numpy`, `scipy`, `matplotlib`.

---

## Repository layout

```
chse/
├── chse/
│   ├── core/
│   │   ├── primitives.py      Params, CapitalStocks, CHSEState
│   │   ├── mechanisms.py      Mechanisms II & III, best-response functions
│   │   ├── network.py         CHSENetwork — graph, beliefs, distance decay
│   │   ├── anticipation.py    Mechanism I — Bayesian Beta-Binomial updating
│   │   ├── kernel.py          Mechanism IV — endogenous propagation kernel K
│   │   └── simulation.py      BenchmarkSim / FullSim — unified simulation API
│   ├── benchmark/
│   │   ├── two_player.py      Two-player ODE + stochastic integration
│   │   ├── oscillation.py     Oscillation condition, Jacobian eigenvalues
│   │   └── flip_threshold.py  Leadership flip time t*
│   ├── phase/
│   │   ├── phase_diagram.py   (HSI, PI) phase diagram, boundary theorems
│   │   ├── jacobian.py        System Jacobian J = J_belief + K^T
│   │   └── cascade.py         Cascade analysis, percolation threshold
│   ├── equilibrium/
│   │   ├── markov.py          Full 4-mechanism Markov chain engine
│   │   ├── hoe.py             HOE estimation, ergodicity, stationarity
│   │   └── lyapunov.py        Lyapunov stability verification
│   ├── welfare/
│   │   ├── distortions.py     Three welfare distortions, social optimum
│   │   └── paradox.py         Hierarchy Persistence Paradox (calibrated)
│   └── empirical/
│       └── fdi.py             FDI pipeline, central bank case, paradox test
├── notebooks/
│   ├── 01_two_player.ipynb    Phase 1 — benchmark, oscillation, flip time
│   ├── 02_phase_diagram.ipynb Phase 2 — network, kernel, phase diagram
│   └── 03_hoe_welfare_empirical.ipynb  Phase 3 — HOE, welfare, empirics
├── tests/
│   ├── test_benchmark.py      56 tests — Phases 1 & 2
│   └── test_phase3.py         55 tests — Phase 3
└── .github/workflows/ci.yml   CI: Python 3.10 / 3.11 / 3.12
```

---

## The model in one paragraph

Players are connected in a network G. For each edge {i,j}, h_ij(t) ∈ [0,1]
is the probability that i leads j (the *hierarchy belief*). Two composite
indices carry all predictive content:

```
HSI = λ_κ · K_i / (λ_R · V_j)       # resistance / attack capacity
PI  = Γ · E[φ(d, G)]                 # network cascade potential
Z   = HSI⁻¹ · (1 + 2·PI)            # instability index
```

| Z | Regime | Theorem |
|---|--------|---------|
| Z < 1 | Stable Hierarchy | 6.1 |
| 1 ≤ Z < 2 | Oscillatory | 6.1 |
| 2 ≤ Z < 3.5 | Cascade-Dominated | 6.2 |
| Z ≥ 3.5 | Turbulent | 6.1 |

The solution concept is the **Hierarchy Orbit Equilibrium (HOE)** — an
invariant probability measure π* of the induced Markov process, not a
fixed-point strategy profile.

---

## Quick examples

```python
# ── Phase 1: two-player benchmark ────────────────────────────────────────
from chse.benchmark import TwoPlayerModel

regimes = TwoPlayerModel.figure2_regimes(T=80)
for name, r in regimes.items():
    print(f"{name}: {r.turnover_count} flips, h∈[{r.h.min():.2f},{r.h.max():.2f}]")
# stable:      0 flips, h∈[0.58,1.00]
# oscillatory: 31 flips, h∈[0.31,0.74]
# cascade:     26 flips, h∈[0.17,0.92]

# ── Phase 2: phase diagram ───────────────────────────────────────────────
from chse.phase.phase_diagram import PhaseDiagram

grid = PhaseDiagram(n_hsi=200, n_pi=200).compute()
print(grid.regime_at(hsi=2.1, pi=0.0))   # 'stable'
print(grid.regime_at(hsi=0.4, pi=0.0))   # 'cascade'

# Verify Theorem 6.1: 200/200 consistent
v = PhaseDiagram().verify_theorem_61(n_test=200)
print(v['fraction'])   # 1.0

# ── Phase 3: HOE estimation ──────────────────────────────────────────────
from chse.core.simulation import BenchmarkSim

# Stable HOE: all chains converge to h*≈0.80
sim = BenchmarkSim(regime='stable', T=300, burn_in=80, n_chains=4)
result = sim.run(seed=42)
print(result.hoe_stats.summary())
print(result.stationarity_check())

# Oscillatory HOE: non-degenerate π* centred at h=0.5
sim_osc = BenchmarkSim(regime='oscillatory', T=300, burn_in=80, n_chains=4)
r_osc = sim_osc.run(seed=42)
print(f"E[h]={r_osc.hoe_stats.mean_h:.4f}  Var(h)={r_osc.hoe_stats.var_h:.4f}")

# ── Phase 3: welfare and paradox ─────────────────────────────────────────
from chse.welfare.distortions import compute_welfare_distortions
from chse.core.network import CHSENetwork
from chse.core.primitives import Params

net = CHSENetwork.complete(3, initial_h=0.65)
wd = compute_welfare_distortions(net, Params(), eta_eq=0.5, kappa_eq=0.6)
print(f"Welfare loss: {wd.welfare_loss:.4f}")

from chse.welfare.paradox import calibrated_paradox_scan
pr = calibrated_paradox_scan()
print(f"∂E[cascade]/∂HSI > 0: {pr.derivative_sign}")   # True

# ── Phase 3: empirical pipeline ──────────────────────────────────────────
from chse.empirical.fdi import build_paper_examples, predict_regime

for ex in build_paper_examples():
    pred = predict_regime(ex)
    print(f"{ex.country} {ex.period}: FDI={ex.FDI:.2f} → {pred['predicted_regime']}")
# Chile 2000-22:  FDI=0.22 → monetary
# Turkey 2021-23: FDI=1.82 → fiscal
```

---

## Ergodicity (default parameters)

The default parameters are set so ergodicity conditions are satisfied:

| Condition | Parameter | Value | Status |
|-----------|-----------|-------|--------|
| Irreducibility | all λ, ρ > 0 | ✓ | True |
| Aperiodicity | ρ_κ/ρ_μ irrational | 0.31/0.30 ≈ 1.033 | True |

```python
from chse.equilibrium.hoe import check_ergodicity_conditions
from chse.core.network import CHSENetwork
from chse.core.primitives import Params

result = check_ergodicity_conditions(CHSENetwork.two_player(), Params())
print(result['ergodic'])   # True
```

---

## Running tests

```bash
# With pytest installed
pytest tests/ -v

# Without pytest (pure stdlib)
python3 -m unittest discover tests/
```

111 tests, all passing across Python 3.10 / 3.11 / 3.12.

---

## Roadmap

| Phase | Content | Status |
|-------|---------|--------|
| 1 | Two-player benchmark, oscillation condition, flip time | ✅ |
| 2 | Network dynamics, propagation kernel K, phase diagram | ✅ |
| 3 | HOE estimation, welfare distortions, empirical pipeline | ✅ |

---

## Citation

```
Contested Hierarchy with Social Embedding (CHSE) — Working Paper v2.0, April 2025.
```

## License

MIT
