# How-to Guides

Concrete recipes for common tasks. Each section is self-contained.

---

## How to explore the phase diagram

The instability index Z = HSI⁻¹ · (1 + 2·PI) determines everything about the qualitative dynamics. The phase diagram maps this over the full (HSI, PI) space.

```python
from chse.phase.phase_diagram import PhaseDiagram, z_to_regime
import numpy as np

# Compute the full grid (300×300 takes ~0.1s)
pd = PhaseDiagram(hsi_min=0.1, hsi_max=3.0, pi_min=0.0, pi_max=1.0,
                  n_hsi=300, n_pi=300)
grid = pd.compute()

# Look up any point
print(grid.regime_at(hsi=2.1, pi=0.0))   # 'stable'
print(grid.regime_at(hsi=0.4, pi=0.5))   # 'turbulent'

# Fraction of the (HSI, PI) grid in each regime
for regime in ['stable', 'oscillatory', 'cascade', 'turbulent']:
    print(f"{regime}: {grid.fraction_in_regime(regime):.3f}")

# Get the three phase boundary curves (for plotting)
curves = pd.boundary_curves()
# 'stable_oscillatory'  → HSI·(1+2·PI) = 1.0
# 'oscillatory_cascade' → HSI·(1+2·PI) = 2.0
# 'cascade_turbulent'   → HSI·(1+2·PI) = 3.5

# Verify Theorem 6.1 on 500 random points
v = pd.verify_theorem_61(n_test=500)
print(f"Theorem 6.1 consistent: {v['n_consistent']}/{v['n_tested']}")  # 500/500

# Map any Z value directly to a regime
print(z_to_regime(0.8))    # 'stable'
print(z_to_regime(1.5))    # 'oscillatory'
print(z_to_regime(2.8))    # 'cascade'
print(z_to_regime(4.0))    # 'turbulent'
```

---

## How to run the two-player benchmark

The two-player benchmark is the analytically tractable core of CHSE. It has a closed-form fixed point, an explicit oscillation condition, and a flip time formula.

```python
from chse.benchmark import TwoPlayerModel, OscillationAnalysis, flip_time
from chse.core.primitives import Params

# --- Deterministic ODE ---
model = TwoPlayerModel(mu=0.6, eta_bar=0.4, kappa_bar=0.4,
                       r_bar=1.0, h0=0.65, params=Params(HSI=1.0))
result = model.integrate(T=100, n_points=500, mode='best_response')
print(f"Fixed point: {model.fixed_point():.4f}")  # h* = 0.5 + (η̄ - κ̄r̄)/μ

# --- Stochastic integration (adds Gaussian noise) ---
result_stoch = model.integrate_stochastic(T=100, noise_std=0.06, seed=42)
print(f"Turnover count: {result_stoch.turnover_count}")
print(f"h range: [{result_stoch.h.min():.3f}, {result_stoch.h.max():.3f}]")

# --- Oscillation analysis ---
osc = OscillationAnalysis(mu=0.6, eta_bar=0.4, kappa_bar=0.4)
analysis = osc.analyse()
print(analysis.summary())
# Regime: oscillatory  disc=-0.28  period=23.75  decay=0.30
print(f"Oscillates: {osc.condition_holds()}")  # μ² < 4η̄κ̄

# --- Flip time ---
ft = flip_time(h0=0.75, mu=0.6, eta_bar=0.4, kappa_bar=0.4,
               r_bar=1.0, epsilon=0.01)
print(f"t* = {ft.t_star:.4f}")  # first passage time to h=0.5

# --- Sweep flip time across HSI values ---
from chse.benchmark.flip_threshold import flip_time_vs_hsi
import numpy as np
hsi_vals = np.linspace(0.3, 3.0, 50)
hsi_out, t_star_vals = flip_time_vs_hsi(
    hsi_vals, h0=0.75, mu=0.6, eta_bar=0.4, r_bar=1.0)
```

---

## How to work with networks

```python
from chse.core.network import CHSENetwork
import numpy as np

# --- Built-in topologies ---
net_2 = CHSENetwork.two_player(h0=0.75)
net_k4 = CHSENetwork.complete(4, initial_h=0.7)
net_path = CHSENetwork.path(6, initial_h=0.6)
net_star = CHSENetwork.star(5, initial_h=0.65)

# --- Custom topology ---
net = CHSENetwork(
    n_players=4,
    edges=[(0,1), (0,2), (1,3), (2,3)],
    initial_h={(0,1): 0.8, (0,2): 0.6, (1,3): 0.7, (2,3): 0.55}
)

# --- Reading beliefs (coherence is automatic) ---
print(net.belief(0, 1))   # h_01 = 0.8
print(net.belief(1, 0))   # h_10 = 0.2  (= 1 - h_01)

# --- Writing beliefs ---
net.set_belief(0, 1, 0.9)              # update single edge
net.set_belief_vector(np.array([0.8, 0.6, 0.7, 0.55]))  # update all at once

# --- Graph queries ---
print(net.neighbours(0))                    # [1, 2]
print(net.shortest_path_length(0, 3))       # 2
print(net.distance_decay(0, 3, decay_rate=1.0))   # exp(-2)≈0.135
print(net.expected_distance_decay(1.0))     # E[φ(d,G)] for PI

# --- Who leads on each edge ---
for e in net.canon_edges:
    leader = net.leader_on_edge(e[0], e[1])
    print(f"  edge {e}: leader = player {leader}")

# --- Copying (for simulation) ---
net_copy = net.copy()  # deep copy, independent beliefs
```

---

## How to build and analyse the propagation kernel

```python
from chse.core.network import CHSENetwork
from chse.core.anticipation import AnticipatState
from chse.core.kernel import TrustState, build_kernel, spectral_radius
from chse.core.kernel import expected_cascade_size, edge_fragility
from chse.core.primitives import Params
import numpy as np

net = CHSENetwork.complete(4, initial_h=0.65)
p = Params(alpha_R=0.4, PI=0.3)

# Initialise anticipation and trust histories
ant = AnticipatState.initialise(net, alpha0=1.0, beta0=1.0)
trust = TrustState.initialise(net, initial_trust=0.5, decay=0.9)

# Simulate 20 periods of anticipation signals to build up accuracy
rng = np.random.default_rng(42)
for _ in range(20):
    for e in net.canon_edges:
        h = net.belief(e[0], e[1])
        xi = float(rng.random() < h * 0.85 + (1 - h) * 0.15)
        ant.update(e[0], e[1], xi)

# Build the kernel
K = build_kernel(net, ant, trust, p, decay_rate=1.0)
print(f"K shape: {K.shape}")          # (6, 6) for K_4
rho = spectral_radius(K)
print(f"ρ(K) = {rho:.4f}")           # cascade potential

# Cascade bounds (Proposition 7.1)
if rho < 1.0:
    bound = expected_cascade_size(rho, p.alpha_R)
    print(f"E[cascade size] ≤ {bound:.4f}")
else:
    print("ρ(K) ≥ 1: unbounded cascade possible")

# Edge fragility — which edges are close to a leadership flip
frag = edge_fragility(net)
for e, f in sorted(frag.items(), key=lambda x: -x[1]):
    print(f"  edge {e}: h={net.h[e]:.2f}  fragility={f:.4f}")
```

---

## How to estimate the HOE

### Using BenchmarkSim (recommended)

```python
from chse.core.simulation import BenchmarkSim

# Three built-in regimes
for regime in ['stable', 'oscillatory', 'cascade']:
    sim = BenchmarkSim(regime=regime, T=300, burn_in=80, n_chains=4)
    result = sim.run(seed=42)
    s = result.hoe_stats
    print(f"{regime:12s}  E[h]={s.mean_h:.4f}  "
          f"Var={s.var_h:.4f}  tau={s.tau_hat:.4f}  "
          f"converged={result.stationarity_check()['converged']}")

# Custom regime
sim = BenchmarkSim(
    regime='custom',
    mu=0.8, eta_bar=0.5, kappa_bar=0.3,
    noise_std=0.05, HSI=1.8,
    T=400, burn_in=100, n_chains=6,
)
result = sim.run(seed=0)
```

### Using FullSim (full four-mechanism chain)

```python
from chse.core.simulation import FullSim
from chse.core.network import CHSENetwork
from chse.core.primitives import Params

net = CHSENetwork.two_player(h0=0.65)
p = Params(HSI=2.1)

fsim = FullSim(net, p, T=200, burn_in=60, n_chains=4)
sim_result, chain_results = fsim.run(seed=42)

# Access individual chains for detailed analysis
for i, chain in enumerate(chain_results):
    print(f"Chain {i+1}: flips={chain.turnover_count()}  "
          f"h_mean={chain.h_mean():.4f}  h_var={chain.h_variance():.6f}")
```

### Using HOEEstimator directly

```python
from chse.equilibrium.hoe import HOEEstimator, stationarity_test

estimator = HOEEstimator(net, p, T=300, burn_in=80, n_chains=4)
chains, stats = estimator.run()
print(stats.summary())

# Stationarity across time windows
for i, chain in enumerate(chains):
    st = stationarity_test(chain, burn_in=80, n_windows=4)
    print(f"Chain {i+1}: {st['window_means']}  converged={st['converged']}")
```

---

## How to verify ergodicity

```python
from chse.equilibrium.hoe import check_ergodicity_conditions
from chse.core.network import CHSENetwork
from chse.core.primitives import Params

net = CHSENetwork.two_player()
p = Params()  # default: rho_kappa=0.31, rho_mu=0.50

result = check_ergodicity_conditions(net, p)
print(f"Irreducible : {result['irreducible']}")
print(f"Aperiodic   : {result['aperiodic']}  "
      f"(ratio={result['rho_ratio']:.6f})")
print(f"Ergodic     : {result['ergodic']}")

for cond, val in result['irreducibility_checks'].items():
    print(f"  {'OK' if val else 'FAIL'}  {cond}")

# What breaks ergodicity:
p_bad = Params(rho_kappa=0.5)   # ratio = 0.5/0.5 = 1.0 (rational)
r_bad = check_ergodicity_conditions(net, p_bad)
print(f"\nWith rho_kappa=0.5: ergodic={r_bad['ergodic']}")  # False
```

---

## How to verify Lyapunov stability

```python
from chse.equilibrium.markov import run_chain
from chse.equilibrium.lyapunov import verify_lyapunov, estimate_orbit_support, lyapunov_V
from chse.core.network import CHSENetwork
from chse.core.primitives import Params

net = CHSENetwork.two_player(h0=0.65)
p = Params(discount=0.9)

# Run a long chain to get a good orbit estimate
chain = run_chain(net, p, T=300, seed=42)

# Estimate Ω* from post-burn-in states
support = estimate_orbit_support(chain, burn_in=80, n_support_points=40)

# Evaluate V at any state
V = lyapunov_V(chain.states[100], support, p, theta_kappa=1.0, theta_mu=1.0)
print(f"V(s_100) = {V:.6f}")

# Full verification: ΔV along trajectory
result = verify_lyapunov(chain, p, burn_in=80, n_support_points=40, Gamma=0.25)
print(result.summary())
# Theorem 8.2: Γ < (1-δ)/(1+δ)  ↔  Gamma < 0.053
# With Gamma=0.25 this is False — system is not Lyapunov stable by theorem bound
# But note: mean ΔV and frac_decreasing measure empirical stability

# To satisfy Theorem 8.2, use very small Gamma or large delta
result_stable = verify_lyapunov(chain, p, burn_in=80, Gamma=0.04)
print(f"Lyapunov stable (Gamma=0.04): {result_stable.lyapunov_stable}")  # True
```

---

## How to compute welfare distortions

```python
from chse.welfare.distortions import (
    compute_welfare_distortions,
    reframing_distortion, social_optimal_eta,
    resistance_distortion, social_optimal_kappa,
    clarity_distortion, total_clarity_gap,
    total_welfare,
)
from chse.core.network import CHSENetwork
from chse.core.primitives import Params

net = CHSENetwork.complete(3, initial_h=0.65)
p = Params(beta_R=0.15, zeta_II=0.3, c_mu=0.3, c_kappa=0.3)

# Full report
wd = compute_welfare_distortions(
    net, p,
    eta_eq=0.5,         # equilibrium reframing investment
    kappa_eq=0.6,       # equilibrium credibility investment
    u_L=10.0, u_F=2.0, # payoffs
    Gamma=0.4,
)
print(wd.summary())
print(f"Welfare loss: {wd.welfare_loss:.4f}")

# Individual components
excess1 = reframing_distortion(net, p, eta_eq=0.5, Gamma=0.4)
eta_so  = social_optimal_eta(net, p, eta_eq=0.5, Gamma=0.4)
excess2 = resistance_distortion(net, p, kappa_eq=0.6)
gap3    = total_clarity_gap(net, p)

# Sweep Gamma
import numpy as np
for Gamma in np.linspace(0, 0.8, 5):
    ex = reframing_distortion(net, p, 0.5, Gamma)
    so = social_optimal_eta(net, p, 0.5, Gamma)
    print(f"Γ={Gamma:.2f}: excess={ex:.4f}  η_SO={so:.4f}")
```

---

## How to demonstrate the Hierarchy Persistence Paradox

```python
from chse.welfare.paradox import calibrated_paradox_scan, paradox_from_simulation
import numpy as np

# Analytic scan (fast, uses closed-form cascade formula)
result = calibrated_paradox_scan(
    hsi_vals=np.linspace(0.3, 3.5, 300),
    alpha_R=0.5,
    trust_avg=0.65,   # average cross-edge trust at equilibrium
    phi_avg=0.60,     # average distance decay in moderate network
    acc_floor=0.50,   # Acc_ij at HSI→0 (weak leader barely predicts)
    acc_ceiling=0.92, # Acc_ij at HSI→∞ (strong leader almost always right)
)
print(result.summary())
print(f"∂E[cascade|collapse]/∂HSI > 0: {result.derivative_sign}")  # True

# Simulation-based (slower, uses full Markov chains)
result_sim = paradox_from_simulation(
    hsi_list=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    n_periods=300, burn_in=100,
)

# Plot (standard three-panel figure)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
valid = ~np.isnan(result.cascade_sizes)

axes[0].plot(result.hsi_vals, result.acc_vals, lw=2)
axes[0].set(xlabel='HSI', ylabel='Acc_ij',
            title='Anticipation accuracy increases with HSI')

axes[1].plot(result.hsi_vals, result.rho_K_vals, color='orange', lw=2)
axes[1].axhline(1.0, color='gray', ls='--')
axes[1].set(xlabel='HSI', ylabel='ρ(K)',
            title='ρ(K) inflates with HSI')

axes[2].plot(result.hsi_vals[valid], result.cascade_sizes[valid], color='red', lw=2)
axes[2].set(xlabel='HSI', ylabel='E[cascade|collapse]',
            title='Hierarchy Persistence Paradox')

plt.suptitle('∂E[cascade|collapse]/∂HSI > 0', y=1.02)
plt.tight_layout()
plt.show()
```

---

## How to run the empirical pipeline

```python
from chse.empirical.fdi import (
    FDIEstimate, build_paper_examples,
    hoe_statistics_from_series, predict_regime,
    persistence_paradox_test,
)
import numpy as np

# --- Step 1: Compute FDI from observable proxies ---
est = FDIEstimate(
    country='Brazil', period='2015-18',
    V_T=0.77,    # political capital proxy (fiscal pressure index)
    K_CB=0.85,   # CB independence score (Dincer-Eichengreen)
    lambda_R=1.0, rho_ratio=1.0,
)
print(f"FDI = {est.FDI:.4f}  →  {est.regime}")  # 0.9059 → contested
print(f"Implied HSI = {est.to_hsi():.4f}")        # 1.104

# --- Step 2: Reproduce Figure 5 ---
examples = build_paper_examples()
print(f"\n{'Country':<8} {'Period':<10} {'FDI':<6} {'Regime'}")
print('-' * 40)
for e in examples:
    print(f"{e.country:<8} {e.period:<10} {e.FDI:<6.2f} {e.regime}")

# --- Step 3: Estimate HOE statistics from observed h(t) ---
# (e.g. yield spread response to CB announcements as h proxy)
from chse.benchmark import TwoPlayerModel
from chse.core.primitives import Params
model = TwoPlayerModel(mu=0.6, eta_bar=0.4, kappa_bar=0.4,
                       r_bar=1.0, h0=0.5, params=Params(HSI=1.0))
observed_h = model.integrate_stochastic(T=500, noise_std=0.09, seed=42).h
hoe_data = hoe_statistics_from_series(observed_h)
print(f"\nObserved HOE: τ̂={hoe_data.tau_hat:.4f}  "
      f"Var(h)={hoe_data.var_h:.4f}  E[h]={hoe_data.mean_h:.4f}")

# --- Step 4: Phase diagram prediction test ---
print(f"\n{'Country':<8} {'FDI':<6} {'Predicted':<12} {'Consistent'}")
print('-' * 40)
for e in examples:
    pred = predict_regime(e, pi=0.0)
    print(f"{e.country:<8} {e.FDI:<6.2f} "
          f"{pred['predicted_regime']:<12} {pred['consistent']}")

# --- Step 5: Hierarchy Persistence Paradox test ---
test = persistence_paradox_test(examples)
print(f"\nCorrelation HSI ~ post-collapse volatility: {test['correlation']:.4f}")
print(f"Paradox confirmed: {test['paradox_confirmed']}")
print(f"{test['interpretation']}")
```

---

## How to add a custom network topology

```python
from chse.core.network import CHSENetwork

# Example: 5-player directed chain with asymmetric initial beliefs
net = CHSENetwork(
    n_players=5,
    edges=[(0,1), (1,2), (2,3), (3,4), (0,2), (1,3)],
    initial_h={
        (0,1): 0.80,  # player 0 leads 1 strongly
        (1,2): 0.55,  # contested
        (2,3): 0.70,
        (3,4): 0.60,
        (0,2): 0.75,
        (1,3): 0.50,  # exact ambiguity
    }
)

# Run the full chain on this network
from chse.equilibrium.markov import run_chain
from chse.core.primitives import Params
chain = run_chain(net, Params(), T=100, seed=42)
print(f"Edges: {net.canon_edges}")
print(f"Trajectory shape: {chain.h_trajectory.shape}")  # (101, 6)
for idx, e in enumerate(net.canon_edges):
    flips = chain.turnover_count(idx)
    print(f"  edge {e}: {flips} flips")
```
