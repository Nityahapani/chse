# CHSE — Contested Hierarchy with Social Embedding

**Who leads and who follows is not a structural primitive. It is a contested social fact.**

Most game theory fixes the Stackelberg hierarchy before the game begins. CHSE makes it endogenous. Players simultaneously play *within* the current hierarchy and *fight over* it — spending credibility capital to resist reframing, manipulation capital to attack leadership, and propagating belief shifts across networks. The solution concept is not a fixed-point strategy profile but an invariant measure over dynamic states: the **Hierarchy Orbit Equilibrium**.

---

## What this is

A fully implemented formal theory in Python, covering:

- **Four coupled mechanisms** driving hierarchy belief dynamics
- **Closed-form two-player benchmark** with oscillation condition and flip time formula
- **Full (HSI, PI) phase diagram** partitioning parameter space into four qualitative regimes
- **Monte Carlo HOE estimation** from multiple chains with stationarity and ergodicity verification
- **Three welfare distortions** with monetised welfare loss and policy implications
- **Hierarchy Persistence Paradox** — the counterintuitive result that stronger hierarchies produce larger cascades when they fall
- **Empirical pipeline** mapping the theory to central bank data with a Fiscal Dominance Index

---

## Install

```bash
git clone https://github.com/Nityahapani/chse.git
cd chse
pip install -e ".[dev]"
```

Python 3.10+. No exotic dependencies — just `numpy`, `scipy`, `matplotlib`.

---

## The two numbers that predict everything

**HSI** (Hierarchy Stability Index) = λ_κ · K_i / (λ_R · V_j)
— ratio of the leader's resistance capacity to the follower's attack capacity.

**PI** (Propagation Intensity) = Γ · E[φ(d, G)]
— how strongly a belief shift on one edge cascades across the network.

**Instability Index** Z = HSI⁻¹ · (1 + 2·PI):

| Z | Regime | Dynamics |
|---|--------|----------|
| Z < 1 | Stable | h(t) converges to fixed point |
| 1 ≤ Z < 2 | Oscillatory | Leadership alternates periodically |
| 2 ≤ Z < 3.5 | Cascade-Dominated | Network-wide belief collapses |
| Z ≥ 3.5 | Turbulent | Sensitive dependence on initial conditions |

```python
from chse.core.primitives import Params

p = Params(HSI=1.5, PI=0.3)
print(p.regime(K_i=0, V_j=0))    # 'oscillatory'
print(p.instability_index(0, 0)) # 1.067
```

---

## Phase diagram

```python
from chse.phase.phase_diagram import PhaseDiagram

pd = PhaseDiagram(n_hsi=300, n_pi=300)
grid = pd.compute()

# Theorem 6.1: stable ↔ HSI·(1+2·PI) > 1
v = pd.verify_theorem_61(n_test=500)
print(v['fraction'])   # 1.0000 — verified on 500 random points

# Boundary curves for plotting
curves = pd.boundary_curves()
# curves['stable_oscillatory']  → HSI·(1+2·PI) = 1
# curves['oscillatory_cascade'] → HSI·(1+2·PI) = 2
```

---

## Four mechanisms

```python
from chse.core.mechanisms import ambiguity_push, reframe_success_prob
from chse.core.anticipation import AnticipateBelief
from chse.core.primitives import Params

p = Params()

# Mechanism I — Bayesian anticipation (Beta-Binomial)
belief = AnticipateBelief(alpha=1.0, beta=1.0)
for xi in [1, 1, 0, 1, 1]:
    belief = belief.update(xi)
print(f"Posterior accuracy: {belief.accuracy():.4f}")   # 0.7143

# Mechanism II — role ambiguity as strategic instrument
delta = ambiguity_push(h=0.8, gamma=1.0, params=p)     # negative: pushes h toward 0.5

# Mechanism III — retroactive reframing
P_R = reframe_success_prob(eta=2.0, rho=0.3, params=p) # attack success probability

# Mechanism IV — endogenous propagation kernel K
from chse.core.network import CHSENetwork
from chse.core.anticipation import AnticipatState
from chse.core.kernel import TrustState, build_kernel, spectral_radius

net = CHSENetwork.complete(4, initial_h=0.65)
K = build_kernel(net, AnticipatState.initialise(net),
                 TrustState.initialise(net), p)
print(f"ρ(K) = {spectral_radius(K):.4f}")   # cascade potential
```

---

## The two-player benchmark

```python
from chse.benchmark import TwoPlayerModel, OscillationAnalysis, flip_time

# Figure 2 — three HSI regimes
regimes = TwoPlayerModel.figure2_regimes(T=80)
for name, r in regimes.items():
    print(f"{name:12s}  flips={r.turnover_count:3d}  "
          f"h∈[{r.h.min():.2f},{r.h.max():.2f}]")
# stable        flips=  0  h∈[0.58,1.00]
# oscillatory   flips= 31  h∈[0.31,0.74]
# cascade       flips= 26  h∈[0.17,0.92]

# Oscillation condition: μ² < 4η̄κ̄  ↔  complex Jacobian eigenvalues
result = OscillationAnalysis(mu=0.6, eta_bar=0.4, kappa_bar=0.4).analyse()
print(result.summary())
# Regime: oscillatory  disc=-0.2800  period=23.75  decay=0.300

# Leadership flip time: t* = (1/μ̃)·ln((h₀−½)/ε)
ft = flip_time(h0=0.75, mu=0.6, eta_bar=0.4, kappa_bar=0.4, r_bar=1.0)
print(f"t* = {ft.t_star:.4f}")   # 10.7296
```

---

## HOE estimation

```python
from chse.core.simulation import BenchmarkSim

# Stable HOE: π* = δ_{h*}
# Four chains from h₀ ∈ {0.20, 0.45, 0.75, 0.90} — all converge to h*≈0.80
sim = BenchmarkSim(regime='stable', T=300, burn_in=80, n_chains=4)
result = sim.run(seed=42)
print(result.hoe_stats.summary())
# tau_hat:  0.0000   (zero flips — stable hierarchy)
# E[h]:     0.7993   (concentrated near h*)
# Var(h):   0.0149
# Stationarity: 0.995  Converged: YES

# Oscillatory HOE: non-degenerate interior distribution
sim_osc = BenchmarkSim(regime='oscillatory', T=300, burn_in=80, n_chains=4)
r_osc = sim_osc.run(seed=42)
print(f"E[h]={r_osc.hoe_stats.mean_h:.4f}  tau_hat={r_osc.hoe_stats.tau_hat:.4f}")
# E[h]=0.5003  tau_hat=0.3805

# Check convergence across windows
print(result.stationarity_check())
# {'window_means': [0.799, 0.800, 0.799, 0.798], 'max_diff': 0.0018, 'converged': True}
```

---

## Ergodicity

The chain is ergodic by default — unique π*, convergence from any starting state:

```python
from chse.equilibrium.hoe import check_ergodicity_conditions
from chse.core.network import CHSENetwork
from chse.core.primitives import Params

result = check_ergodicity_conditions(CHSENetwork.two_player(), Params())
print(result['ergodic'])      # True
print(result['irreducible'])  # True — all λ_R, λ_κ, λ_σ, ρ_κ, ρ_μ > 0
print(result['aperiodic'])    # True — ρ_κ/ρ_μ = 0.31/0.30 ≈ 1.033 (irrational)
```

---

## Welfare distortions

Three distortions push the equilibrium away from the social optimum:

```python
from chse.welfare.distortions import compute_welfare_distortions
from chse.core.network import CHSENetwork
from chse.core.primitives import Params

net = CHSENetwork.complete(3, initial_h=0.65)
wd = compute_welfare_distortions(net, Params(), eta_eq=0.5, kappa_eq=0.6)
print(wd.summary())
```

```
Distortion 1 — Over-investment in reframing:
  Excess: 0.0500  followers ignore network spillover of their attacks
  Fix:    legal estoppel, institutional precedent

Distortion 2 — Over-investment in commitment resistance:
  Excess: 0.1104  leaders ignore legibility externality to third parties
  Fix:    legibility subsidies, transparent announcements

Distortion 3 — Under-investment in hierarchy clarity:
  Gap:    1.2600  clarity is a public good, under-provided in equilibrium
  Fix:    public commitment requirements, board resolutions

Welfare loss: 1.4043
```

---

## The Hierarchy Persistence Paradox

```python
from chse.welfare.paradox import calibrated_paradox_scan
import numpy as np

result = calibrated_paradox_scan(hsi_vals=np.linspace(0.3, 3.5, 300))
print(f"∂E[cascade|collapse]/∂HSI > 0: {result.derivative_sign}")
# True

print(f"Cascade range: [{result.cascade_sizes.min():.3f}, {result.cascade_sizes.max():.3f}]")
# [0.652, 0.738]
```

**The counterintuitive result**: stronger, more credible leaders build more accurate anticipation records → inflates propagation weights in K → network is *more* susceptible to cascade if they eventually fall.

High-credibility central banks produce *larger* market disruptions when independence is lost. Dominant firms produce *larger* supply-chain cascades when authority collapses. Hegemonic creditors produce *larger* debt restructuring crises than marginal ones.

---

## Empirical pipeline

```python
from chse.empirical.fdi import build_paper_examples, predict_regime

for ex in build_paper_examples():
    pred = predict_regime(ex)
    print(f"{ex.country:<8} {ex.period}  FDI={ex.FDI:.2f}  → {pred['predicted_regime']}")
```

```
Chile    2000-22  FDI=0.22  → monetary
US       2000-07  FDI=0.18  → monetary
US       2020-23  FDI=0.54  → contested
Brazil   2015-18  FDI=0.91  → contested
Zambia   2020-23  FDI=1.40  → fiscal
Turkey   2021-23  FDI=1.82  → fiscal
```

6/6 consistent with the phase diagram prediction.

---

## Structure

```
chse/
├── core/
│   ├── primitives.py      Params, CapitalStocks, CHSEState
│   ├── mechanisms.py      Mechanisms II & III, optimal best-response
│   ├── network.py         CHSENetwork — graph, beliefs, distance decay
│   ├── anticipation.py    Mechanism I — Bayesian Beta-Binomial
│   ├── kernel.py          Mechanism IV — endogenous propagation kernel K
│   └── simulation.py      BenchmarkSim / FullSim — unified simulation API
├── benchmark/
│   ├── two_player.py      ODE + stochastic integration, Figure 2
│   ├── oscillation.py     Oscillation condition, Jacobian eigenvalues
│   └── flip_threshold.py  Leadership flip time t*
├── phase/
│   ├── phase_diagram.py   (HSI, PI) grid, boundary curves, theorem verification
│   ├── jacobian.py        System Jacobian J = J_belief + K^T
│   └── cascade.py         Spectral cascade condition, size distribution
├── equilibrium/
│   ├── markov.py          Full 4-mechanism Markov chain engine
│   ├── hoe.py             HOE estimation, ergodicity, stationarity
│   └── lyapunov.py        Lyapunov stability V(s), ΔV verification
├── welfare/
│   ├── distortions.py     Three distortions, social optimum, welfare loss
│   └── paradox.py         Hierarchy Persistence Paradox
└── empirical/
    └── fdi.py             FDI formula, Figure 5, regime prediction

notebooks/
├── 01_two_player.ipynb            Phase 1 — benchmark
├── 02_phase_diagram.ipynb         Phase 2 — network and phase diagram
└── 03_hoe_welfare_empirical.ipynb Phase 3 — HOE, welfare, empirics

tests/
├── test_benchmark.py   56 tests
└── test_phase3.py      55 tests
```

---

## Tests

```bash
pytest tests/ -v                     # with pytest
python3 -m unittest discover tests/  # without pytest
```

111 tests. All pass on Python 3.10, 3.11, 3.12.

---

## License

MIT
