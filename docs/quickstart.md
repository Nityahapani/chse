# Quickstart

Get from zero to meaningful results in under ten minutes.

---

## Install

```bash
git clone https://github.com/Nityahapani/chse.git
cd chse
pip install -e ".[dev]"
```

Python 3.10 or later. Dependencies are `numpy`, `scipy`, and `matplotlib` — nothing else.

---

## Verify the install

```bash
python3 -c "import chse; print('ok')"
pytest tests/ -q
```

You should see `111 passed`. If you don't have pytest: `python3 -m unittest discover tests/`.

---

## Five things you can do in five minutes

### 1. Get a regime label from two numbers

```python
from chse.core.primitives import Params

# HSI = resistance / attack capacity
# PI  = network cascade potential
p = Params(HSI=1.5, PI=0.3)
print(p.regime(K_i=0, V_j=0))       # 'oscillatory'
print(p.instability_index(0, 0))     # 1.067
```

### 2. Reproduce Figure 2 — three qualitative regimes

```python
from chse.benchmark import TwoPlayerModel

regimes = TwoPlayerModel.figure2_regimes(T=80)
for name, r in regimes.items():
    print(f"{name:12s}  flips={r.turnover_count:3d}  "
          f"h ∈ [{r.h.min():.2f}, {r.h.max():.2f}]")
```

```
stable        flips=  0  h ∈ [0.58, 1.00]
oscillatory   flips= 31  h ∈ [0.31, 0.74]
cascade       flips= 26  h ∈ [0.17, 0.92]
```

### 3. Plot the phase diagram

```python
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
from chse.phase.phase_diagram import PhaseDiagram, REGIME_COLOURS

pd = PhaseDiagram(n_hsi=200, n_pi=200)
grid = pd.compute()
curves = pd.boundary_curves()

regime_to_num = {'stable': 0, 'oscillatory': 1, 'cascade': 2, 'turbulent': 3}
num_grid = np.array([[regime_to_num[r] for r in row] for row in grid.regimes])
colours = [REGIME_COLOURS[k] for k in ['stable', 'oscillatory', 'cascade', 'turbulent']]
cmap = mcolors.ListedColormap(colours)

fig, ax = plt.subplots(figsize=(9, 6))
ax.pcolormesh(grid.hsi_values, grid.pi_values, num_grid,
              cmap=cmap, vmin=-0.5, vmax=3.5, shading='auto')
for name, (hsi_c, pi_c) in curves.items():
    ax.plot(hsi_c, pi_c, color='white', lw=1.5)
ax.set_xlabel('HSI'); ax.set_ylabel('PI')
ax.set_title('CHSE Phase Diagram')
patches = [mpatches.Patch(color=c, label=l) for c, l in
           zip(colours, ['Stable', 'Oscillatory', 'Cascade', 'Turbulent'])]
ax.legend(handles=patches, loc='lower right')
plt.tight_layout(); plt.show()
```

### 4. Estimate the HOE from simulated data

```python
from chse.core.simulation import BenchmarkSim

# Stable regime: all chains converge to h*≈0.80
sim = BenchmarkSim(regime='stable', T=300, burn_in=80, n_chains=4)
result = sim.run(seed=42)
print(result.hoe_stats.summary())
print(result.stationarity_check())
```

### 5. Run the empirical pipeline

```python
from chse.empirical.fdi import build_paper_examples, predict_regime

for ex in build_paper_examples():
    pred = predict_regime(ex)
    match = 'OK' if pred['consistent'] else '!!'
    print(f"{match}  {ex.country:<8} {ex.period}  "
          f"FDI={ex.FDI:.2f}  → {pred['predicted_regime']}")
```

---

## Notebooks

Three Jupyter notebooks walk through everything in order:

| Notebook | Content |
|----------|---------|
| `01_two_player.ipynb` | Two-player ODE, oscillation condition, flip time, Figure 2 |
| `02_phase_diagram.ipynb` | Network structure, kernel K, phase diagram, cascade analysis |
| `03_hoe_welfare_empirical.ipynb` | HOE estimation, ergodicity, Lyapunov, welfare, FDI pipeline |

```bash
jupyter notebook notebooks/
```

---

## Package structure at a glance

```
chse.core.primitives   →  Params, CapitalStocks, CHSEState
chse.core.network      →  CHSENetwork (graph + beliefs)
chse.core.mechanisms   →  ambiguity_push, reframe_success_prob, optimal_eta, ...
chse.core.anticipation →  AnticipateBelief, AnticipatState, suppression_probability
chse.core.kernel       →  build_kernel, spectral_radius, edge_fragility, TrustState
chse.core.simulation   →  BenchmarkSim, FullSim, SimResult

chse.benchmark         →  TwoPlayerModel, OscillationAnalysis, flip_time
chse.phase             →  PhaseDiagram, SystemJacobian, CascadeAnalysis
chse.equilibrium       →  run_chain, HOEEstimator, verify_lyapunov, check_ergodicity_conditions
chse.welfare           →  compute_welfare_distortions, calibrated_paradox_scan
chse.empirical         →  FDIEstimate, build_paper_examples, predict_regime
```

---

## Common patterns

**Override HSI directly** (skip per-state calculation):
```python
p = Params(HSI=2.1)   # all methods return 2.1 for HSI
```

**Build any network topology:**
```python
from chse.core.network import CHSENetwork
net = CHSENetwork.complete(n=4, initial_h=0.7)  # K_4
net = CHSENetwork.star(n=5, initial_h=0.65)     # star
net = CHSENetwork.path(n=6, initial_h=0.6)      # path
# or custom:
net = CHSENetwork(n_players=3, edges=[(0,1),(1,2)],
                  initial_h={(0,1): 0.8, (1,2): 0.6})
```

**Check a belief from either direction** (coherence is automatic):
```python
print(net.belief(0, 1))   # h_01
print(net.belief(1, 0))   # 1 − h_01
```

**Run the full Markov engine on any network:**
```python
from chse.equilibrium.markov import run_chain
chain = run_chain(net, Params(), T=200, seed=42)
print(chain.h_trajectory.shape)   # (201, n_edges)
```
