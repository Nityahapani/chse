# API Reference

Complete reference for every public class and function. Organised by module. Every entry maps directly to an implemented object in `chse/`.

---

## `chse.core.primitives`

### `Params`

The master parameter container. All 15 raw model parameters plus the two composite indices.

```python
from chse.core.primitives import Params

p = Params(
    lambda_kappa=1.0,   # resistance efficiency: κ → ρ
    lambda_R=1.0,       # reframing efficiency: η → P_R
    lambda_sigma=1.0,   # suppression efficiency: δ → σ
    K_cap=10.0,         # credibility capital cap
    M_cap=10.0,         # manipulation capital cap
    rho_kappa=0.31,     # credibility replenishment rate
    rho_mu=0.50,        # manipulation replenishment rate
    mu_II=1.0,          # ambiguity mean-reversion strength
    zeta_II=0.3,        # ambiguity network spillover
    alpha_I=0.2,        # anticipation h-impact
    beta_I=0.1,         # predictability penalty
    alpha_R=0.3,        # reframe direct belief drop
    beta_R=0.1,         # reframe network spillover
    delta_kappa=0.5,    # credibility loss per reframe
    discount=0.95,      # discount factor δ ∈ (0,1)
    HSI=None,           # override HSI (skips per-state calculation)
    PI=None,            # override PI
    c_mu=0.5,           # marginal cost of reframing investment
    c_kappa=0.5,        # marginal cost of credibility investment
)
```

**Methods:**

`p.hsi(K_i, V_j)` → `float`  
Compute HSI_ij = λ_κ · K_i / (λ_R · V_j). Returns `p.HSI` if override is set.

`p.pi(Gamma, expected_phi)` → `float`  
Compute PI = Γ · E[φ(d, G)]. Returns `p.PI` if override is set.

`p.instability_index(K_i, V_j, Gamma, expected_phi)` → `float`  
Compute Z = HSI⁻¹ · (1 + 2·PI).

`p.regime(K_i, V_j, Gamma, expected_phi)` → `str`  
Return regime label: `'stable'`, `'oscillatory'`, `'cascade'`, or `'turbulent'`.

---

### `CapitalStocks`

Per-player resource stocks.

```python
from chse.core.primitives import CapitalStocks

stocks = CapitalStocks(kappa=5.0, mu=5.0)
```

`stocks.deplete_kappa(amount, params)` → `CapitalStocks`  
Return new stocks with κ reduced (floored at 0).

`stocks.deplete_mu(amount, params)` → `CapitalStocks`  
Return new stocks with μ reduced (floored at 0).

`stocks.replenish(params, kappa_depleted, mu_depleted)` → `CapitalStocks`  
Apply one period of replenishment: κ(t+1) = min(K_cap, κ + ρ_κ − depleted).

---

### `CHSEState`

Complete state for the two-player game at a single time step.

```python
from chse.core.primitives import CHSEState

s = CHSEState(h=0.75)
```

`s.h` — hierarchy belief h_12 ∈ [0, 1]  
`s.h_21` — coherent complement: 1 − h  
`s.leader()` → `int` — player with h > 0.5  
`s.stage_payoff(u_L, u_F, player)` → `float`  
`s.project()` → `CHSEState` — clips h to [0, 1]

---

## `chse.core.network`

### `CHSENetwork`

The network G = (N, E) with a hierarchy belief matrix.

```python
from chse.core.network import CHSENetwork

# Constructors
net = CHSENetwork.two_player(h0=0.75)          # 2 nodes, 1 edge
net = CHSENetwork.complete(n=4, initial_h=0.7) # K_n
net = CHSENetwork.path(n=6, initial_h=0.6)     # path graph
net = CHSENetwork.star(n=5, initial_h=0.65)    # star graph

# Custom
net = CHSENetwork(
    n_players=3,
    edges=[(0,1),(0,2),(1,2)],
    initial_h={(0,1):0.7, (0,2):0.6, (1,2):0.55}
)
```

**Key methods:**

`net.belief(i, j)` → `float` — h_ij (handles both orderings via coherence)  
`net.set_belief(i, j, value)` — update h_ij, clip to [0, 1]  
`net.belief_vector()` → `np.ndarray` — all beliefs in canonical edge order  
`net.set_belief_vector(v)` — set all beliefs from array  
`net.neighbours(i)` → `list[int]`  
`net.leader_on_edge(i, j)` → `int`  
`net.shortest_path_length(i, j)` → `int`  
`net.distance_decay(i, j, decay_rate=1.0)` → `float` — φ(d, G) = exp(−r·d)  
`net.expected_distance_decay(decay_rate=1.0)` → `float` — E[φ(d, G)] for PI computation  
`net.copy()` → `CHSENetwork`  
`net.canon_edges` — list of canonical (i<j) edge tuples

---

## `chse.core.mechanisms`

All functions return a float Δh — the additive change to h.

`ambiguity_push(h, gamma, params)` → `float`  
Mechanism II: Δ^II h_ij = −μ_II · γ · (h − ½)

`reframe_resistance(c, params)` → `float`  
ρ_i(τ) = 1 − exp(−λ_κ · c)

`reframe_success_prob(eta, rho, params)` → `float`  
P_R(η, ρ) = (1 − exp(−λ_R · η)) · (1 − ρ)

`reframing_investment(h, eta, rho, params)` → `float`  
Mechanism III: Δ^III h_ij = −α_R · P_R (always ≤ 0 when h > 0.5)

`commitment_resistance(h, kappa, r, params)` → `float`  
Resistance term κ · r in the two-player ODE.

`optimal_eta(h, params)` → `float`  
Follower's log-optimal reframing spend: (1/λ_R)·ln(α_R·λ_R·(½−h)/c_μ) if h < ½, else 0.

`optimal_kappa_spend(h, params)` → `float`  
Leader's log-optimal resistance spend: (1/λ_κ)·ln(α_R·λ_κ·(h−½)/c_κ) if h > ½, else 0.

---

## `chse.core.anticipation`

### `AnticipateBelief`

Beta distribution tracking anticipation successes for one directed pair.

```python
belief = AnticipateBelief(alpha=1.0, beta=1.0)  # uniform prior
belief = belief.update(xi=1.0)                  # observe success
print(belief.mean)      # posterior mean = α/(α+β)
print(belief.accuracy()) # Acc_ij
```

### `AnticipatState`

Tracks Beta-Binomial states for all edges in a network.

```python
ant = AnticipatState.initialise(net, alpha0=1.0, beta0=1.0)
ant.accuracy(i, j)             # returns Acc_ij ∈ [0,1]
ant.update(i, j, xi)           # update after observing signal xi ∈ {0,1}
```

`suppression_probability(delta, params)` → `float`  
σ(δ) = 1 − exp(−λ_σ · δ)

---

## `chse.core.kernel`

### `TrustState`

Cross-edge trust levels Trust_{ij→kl}(t).

```python
trust = TrustState.initialise(net, initial_trust=0.5, decay=0.9)
trust.get(e1, e2)                    # trust from edge e1 to e2
trust.update(e1, e2, delta1, delta2) # update based on belief co-movement
```

**Module-level functions:**

`build_kernel(network, ant_state, trust_state, params, decay_rate=1.0)` → `np.ndarray`  
Build the |E| × |E| propagation kernel matrix K.

`spectral_radius(K)` → `float`  
ρ(K) = max|eigenvalue|. The cascade threshold.

`expected_cascade_size(rho_K, alpha_R)` → `float`  
E[cascade size] ≤ α_R / (1 − ρ(K)) when ρ(K) < 1; returns `inf` otherwise.

`edge_fragility(network)` → `dict`  
{edge: 1 − |2h−1|} — fragility 1 at h=0.5, 0 at h∈{0,1}.

`optimal_cascade_seed(K, network, attacker)` → `dict`  
Optimal accuracy allocation for strategic cascade seeding (Prop. 7.2).

---

## `chse.core.simulation`

### `BenchmarkSim`

Multi-chain stochastic benchmark HOE estimation. Uses `TwoPlayerModel.integrate_stochastic` from diverse initial states.

```python
from chse.core.simulation import BenchmarkSim

# Built-in regimes: 'stable', 'oscillatory', 'cascade'
sim = BenchmarkSim(
    regime='stable',   # HSI=2.1, noise_std=0.03, h0∈{0.20,0.45,0.75,0.90}
    T=300,
    burn_in=80,
    n_chains=4,
)
result = sim.run(seed=42)  # → SimResult

# Custom regime
sim = BenchmarkSim(
    regime='custom',
    mu=0.8, eta_bar=0.5, kappa_bar=0.3, r_bar=1.0,
    noise_std=0.05, HSI=1.8,
    h0_spread=[0.3, 0.5, 0.7, 0.9],
    T=300, burn_in=80, n_chains=4,
)
```

### `FullSim`

Full 4-mechanism Markov chain simulation. Runs `equilibrium.markov.run_chain` from diverse initial states.

```python
from chse.core.simulation import FullSim
from chse.core.network import CHSENetwork
from chse.core.primitives import Params

fsim = FullSim(
    network=CHSENetwork.two_player(h0=0.65),
    params=Params(HSI=2.1),
    T=200,
    burn_in=60,
    n_chains=4,
)
sim_result, chain_results = fsim.run(seed=42)
```

### `SimResult`

Common output from both simulators.

```python
result.h_trajectories    # list[np.ndarray] — one per chain
result.pooled_h          # np.ndarray — post-burn-in, all chains
result.hoe_stats         # HOEFromData — HOE statistics
result.turnover_counts   # list[int] — flips per chain
result.stationarity_check(n_windows=4)  # dict with 'converged'
result.T, result.burn_in, result.n_chains, result.regime, result.HSI, result.mode
```

---

## `chse.benchmark`

### `TwoPlayerModel`

```python
from chse.benchmark import TwoPlayerModel

model = TwoPlayerModel(
    mu=0.6,           # mean-reversion strength μ > 0
    eta_bar=0.4,      # constant reframing rate η̄
    kappa_bar=0.4,    # constant resistance rate κ̄
    r_bar=1.0,        # reframing attack rate r̄
    h0=0.75,          # initial belief
    params=Params(HSI=1.0),
)

model.fixed_point()    # h* = ½ + (η̄ − κ̄r̄)/μ
model.integrate(T, n_points, mode)   # ODE integration → TwoPlayerResult
model.integrate_stochastic(T, noise_std, seed)  # stochastic → TwoPlayerResult
TwoPlayerModel.figure2_regimes(T=80)  # dict of stable/oscillatory/cascade
```

`TwoPlayerResult` fields: `t`, `h`, `eta`, `kappa`, `h_star`, `params`, `mode`,  
`final_h`, `leadership_periods`, `turnover_count`, `turnover_frequency`

### `OscillationAnalysis`

```python
from chse.benchmark import OscillationAnalysis

osc = OscillationAnalysis(mu=0.6, eta_bar=0.4, kappa_bar=0.4, r_bar=1.0)
result = osc.analyse()       # → OscillationResult
osc.condition_holds()        # True if μ² < 4η̄κ̄
osc.phase_portrait()         # (h_vals, dh_vals) for plotting
OscillationAnalysis.stability_scan(mu_vals, eta_vals, kappa_bar, r_bar)
```

`OscillationResult` fields: `oscillates`, `discriminant`, `eigenvalues`, `period`, `decay_rate`, `h_star`, `regime`

### `flip_time`

```python
from chse.benchmark import flip_time, FlipResult

result = flip_time(
    h0=0.75,
    mu=0.6,
    eta_bar=0.4,
    kappa_bar=0.4,
    r_bar=1.0,
    epsilon=0.01,
    params=Params(HSI=1.0),
    noise_std=0.0,    # 0 = deterministic; >0 uses half-period estimate
)
result.t_star       # None if stable, else float
result.summary()
```

`flip_time_vs_hsi(hsi_vals, h0, mu, eta_bar, r_bar, epsilon)` → `(hsi_vals, t_star_vals)`  
Sweep flip time across HSI values.

---

## `chse.phase`

### `PhaseDiagram`

```python
from chse.phase.phase_diagram import PhaseDiagram

pd = PhaseDiagram(
    hsi_min=0.1, hsi_max=3.0,
    pi_min=0.0,  pi_max=1.0,
    n_hsi=300,   n_pi=300,
)
grid = pd.compute()            # → RegimeGrid
curves = pd.boundary_curves()  # dict of boundary (hsi_arr, pi_arr) tuples
pd.verify_theorem_61(n_test=200)  # dict with 'fraction'
pd.verify_theorem_62()
pd.special_points()            # paper example points
```

`RegimeGrid` fields: `hsi_values`, `pi_values`, `Z` (2D), `regimes` (2D list of str)  
`grid.regime_at(hsi, pi)` → `str`  
`grid.fraction_in_regime(regime)` → `float`  
`grid.boundary_hsi(pi, boundary)` → `float`

`z_to_regime(Z)` → `str` — maps instability index to regime label  
`REGIME_COLOURS` — dict mapping regime names to hex colour strings

### `SystemJacobian`

```python
from chse.phase.jacobian import SystemJacobian

jac = SystemJacobian(
    network, params,
    mu=0.6, eta_bar=0.4, kappa_bar=0.4, r_bar=1.0,
    K=np.zeros((n_edges, n_edges)),  # optional propagation kernel
)
result = jac.analyse()   # → JacobianResult
jac.is_stable()          # Theorem 6.1: HSI·(1+2·PI) > 1
jac.cascade_threshold_rho_K()  # 1 − 1/HSI
jac.hopf_bifurcation_condition()  # μ² < 4η̄κ̄
```

`JacobianResult` fields: `rho_J`, `rho_J_belief`, `rho_K`, `eigenvalues`, `disc_2player`, `regime`, `Z`, `HSI`, `PI`, `n_edges`

### `CascadeAnalysis`

```python
from chse.phase.cascade import CascadeAnalysis

ca = CascadeAnalysis(params)
ca.cascade_probability(rho_K)      # sigmoid transition at ρ(K)=1
ca.analyse(rho_K)                  # → CascadeResult
ca.scan_rho_K(rho_K_vals)          # (rho_K, probs, sizes)
ca.persistence_paradox_scan(hsi_vals)  # (hsi, rho_K, cascade_sizes)
ca.cascade_size_distribution(rho_K, n_max)  # (sizes, probs) — geometric model
```

---

## `chse.equilibrium`

### `run_chain`

```python
from chse.equilibrium.markov import run_chain

chain = run_chain(
    network=CHSENetwork.two_player(h0=0.65),
    params=Params(),
    T=200,
    kappa0=np.array([5.0, 5.0]),  # optional initial capitals
    mu0=np.array([5.0, 5.0]),
    h0=np.array([0.65]),          # optional initial beliefs
    seed=42,
)
```

`ChainResult` fields: `h_trajectory` (T+1, n_edges), `kappa_traj` (T+1, n), `mu_traj` (T+1, n), `states`, `T`, `n_edges`, `n_players`  
`chain.turnover_count(edge_idx=0)`, `chain.turnover_frequency()`, `chain.h_mean()`, `chain.h_variance()`

### `HOEEstimator`

```python
from chse.equilibrium.hoe import HOEEstimator

est = HOEEstimator(network, params, T=300, burn_in=80, n_chains=4)
chains, stats = est.run()   # → (list[ChainResult], HOEStatistics)
```

`HOEStatistics` fields: `tau_hat`, `var_h`, `mean_h`, `expected_cascade`, `n_periods`, `n_chains`, `converged`, `stationarity_gap`

`stationarity_test(chain, burn_in, edge_idx, n_windows)` → `dict`  
`check_ergodicity_conditions(network, params)` → `dict`

### Lyapunov stability

```python
from chse.equilibrium.lyapunov import verify_lyapunov, estimate_orbit_support, lyapunov_V

support = estimate_orbit_support(chain, burn_in=80, n_support_points=40)
V_at_s = lyapunov_V(state, support, params, theta_kappa=1.0, theta_mu=1.0)
result = verify_lyapunov(chain, params, burn_in=80, n_support_points=40, Gamma=0.25)
```

`LyapunovResult` fields: `V_trajectory`, `delta_V`, `frac_decreasing`, `mean_delta_V`, `stability_cond`, `stability_bound`, `lyapunov_stable`

---

## `chse.welfare`

### `compute_welfare_distortions`

```python
from chse.welfare.distortions import compute_welfare_distortions

wd = compute_welfare_distortions(
    network, params,
    eta_eq=0.5,          # equilibrium reframing investment
    kappa_eq=0.6,        # equilibrium credibility investment
    u_L=10.0,            # leadership payoff
    u_F=2.0,             # followership payoff
    Gamma=0.4,           # propagation factor for distortion 1
    u_coordination=0.5,  # per-neighbour coordination benefit for distortion 2
)
```

`WelfareDistortions` fields: `reframing_excess`, `eta_SO`, `eta_eq`, `resistance_excess`, `clarity_gap`, `per_player_clarity`, `total_welfare_eq`, `total_welfare_SO`, `welfare_loss`, `policy_implications`

**Module-level functions:**

`total_welfare(network, u_L, u_F, investment_costs=None)` → `float`  
`reframing_distortion(network, params, eta_eq, Gamma)` → `float` — Distortion 1 magnitude  
`social_optimal_eta(network, params, eta_eq, Gamma)` → `float` — η_SO  
`resistance_distortion(network, params, kappa_eq, u_coordination)` → `float`  
`social_optimal_kappa(network, params, kappa_eq, u_coordination)` → `float`  
`clarity_distortion(network, params)` → `dict` — per-player under-investment  
`total_clarity_gap(network, params)` → `float` — Distortion 3 aggregate

### Hierarchy Persistence Paradox

```python
from chse.welfare.paradox import calibrated_paradox_scan, paradox_from_simulation

result = calibrated_paradox_scan(
    hsi_vals=np.linspace(0.3, 3.5, 300),
    alpha_R=0.5,
    trust_avg=0.65,
    phi_avg=0.60,
    acc_floor=0.50,
    acc_ceiling=0.92,
)

# Alternatively: simulation-based (slower)
result = paradox_from_simulation(hsi_list=[0.5, 1.0, 1.5, 2.0, 2.5], n_periods=300)
```

`ParadoxResult` fields: `hsi_vals`, `acc_vals`, `rho_K_vals`, `cascade_sizes`, `derivative_sign`, `n_below_one`

---

## `chse.empirical`

### `FDIEstimate`

```python
from chse.empirical.fdi import FDIEstimate, build_paper_examples

est = FDIEstimate(
    country='Turkey',
    period='2021-23',
    V_T=0.82,         # political capital proxy
    K_CB=0.45,        # CB independence score (Dincer-Eichengreen)
    lambda_R=1.0,
    rho_ratio=1.0,
)
est.FDI        # computed: V_T·λ_R / (K_CB·ρ_ratio)
est.regime     # 'monetary', 'contested', or 'fiscal'
est.to_hsi()   # 1/FDI

examples = build_paper_examples()  # 6 paper examples matching Figure 5
```

**Module-level functions:**

`hoe_statistics_from_series(h_series)` → `HOEFromData`  
Estimate τ̂, Var(h), E[h] from an observed h(t) time series.

`predict_regime(fdi_estimate, pi=0.0)` → `dict`  
Predict regime from FDI; verify consistency with phase diagram.

`persistence_paradox_test(fdi_estimates, collapse_volatility=None)` → `dict`  
Regress post-collapse volatility on pre-collapse HSI. Returns correlation and whether paradox is confirmed.

`HOEFromData` fields: `tau_hat`, `var_h`, `mean_h`, `h_above_half`, `stationarity_p`, `n_obs`
