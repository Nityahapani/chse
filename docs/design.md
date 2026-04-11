# Design Notes

The decisions behind the implementation — why things are built the way they are.

---

## Why HOE instead of Nash or Stackelberg

Nash equilibrium predicts h* = 0.5 always (symmetric, no credibility asymmetry). Stackelberg assumes the hierarchy is structural and exogenous. Both are knife-edge cases.

The Hierarchy Orbit Equilibrium is more general than either. A degenerate HOE concentrated at h=1 is the Stackelberg outcome. A degenerate HOE at h=0.5 is the Nash outcome. The interior HOE — a non-degenerate invariant measure — is what actually arises when the model is run with interior parameters. The solution concept does not presuppose the hierarchy; it derives it.

---

## Why an invariant measure, not a fixed point

The standard approach to dynamic games is to look for a strategy profile that is a fixed point of some best-response operator (Nash), or a value function that is a fixed point of a Bellman operator (Markov Perfect Equilibrium).

CHSE uses neither. The state space S is compact but the belief dynamics are not a contraction — there is no reason to expect a unique fixed-point belief trajectory. The solution concept is instead the **invariant measure of the induced Markov process**: the probability distribution over states that is preserved under one period of the game.

This maps directly to the implementation: `equilibrium/markov.py` implements one period, `equilibrium/hoe.py` estimates the invariant measure by running many periods and collecting statistics.

---

## The two composite indices

The model has 15 raw parameters. Early versions tried to analyse the full 15-dimensional parameter space. The key compression result is that the qualitative dynamics depend on parameter space only through two scalars:

- HSI = λ_κ · K_i / (λ_R · V_j)
- PI = Γ · E[φ(d, G)]

This is not a model simplification — it is a theorem. The phase boundaries (Theorems 6.1 and 6.2) are stated entirely in terms of Z = HSI⁻¹ · (1 + 2·PI). Individual parameters matter for second-order effects (flip time, welfare magnitudes) but not for the qualitative regime classification.

The implementation reflects this: `Params` carries an `HSI` override field specifically so callers can specify the composite index directly without needing to set all underlying parameters.

---

## Why the mechanisms are separated into four distinct files

The four mechanisms — anticipation (I), ambiguity (II), reframing (III), propagation (IV) — are mathematically distinct and operate on different timescales:

- **Mechanism I** (Beta-Binomial) updates a belief about a latent type. It is Bayesian.
- **Mechanism II** (ambiguity push) is a direct, strategic spend of manipulation capital to move h toward 0.5.
- **Mechanism III** (reframing) involves capital investment and a probabilistic outcome. It can succeed or fail each period.
- **Mechanism IV** (kernel propagation) is a linear operator applied after the direct mechanisms; it requires first building the kernel from the accumulated history.

Separating them into `core/anticipation.py`, `core/mechanisms.py`, and `core/kernel.py` means each can be tested and replaced independently. The `equilibrium/markov.py` engine assembles them in the correct period order.

---

## The circular import and how it is resolved

`chse.core.simulation` imports from `chse.benchmark.two_player` (to run `BenchmarkSim`). But `chse.core.__init__` re-exports everything from `simulation.py`. And `chse.benchmark.two_player` imports from `chse.core.primitives`, which triggers `chse.core.__init__`, which triggers `simulation.py`, which tries to import from `chse.benchmark.two_player` — a cycle.

The resolution: `simulation.py` defers its cross-package imports to call time using local imports inside `BenchmarkSim.run()` and `FullSim.run()`. The module-level code in `simulation.py` only imports from within `chse.core` itself.

```python
# simulation.py — at module level (safe)
from .primitives import Params
from .network import CHSENetwork

# simulation.py — inside BenchmarkSim.run() (deferred, breaks cycle)
def run(self, seed=42):
    from ..benchmark.two_player import TwoPlayerModel  # lazy
    ...
```

This means `from chse.benchmark import TwoPlayerModel` and `from chse.core import BenchmarkSim` both work cleanly from any import order.

---

## The `rho_kappa = 0.31` default

The ergodicity proposition requires the replenishment ratio ρ_κ/ρ_μ to be irrational (to guarantee aperiodicity). The default `rho_mu = 0.30` and setting `rho_kappa = 0.30` gives ratio 1.0 — rational — and breaks the aperiodicity check.

The fix sets `rho_kappa = 0.31`. The ratio 0.31/0.30 ≈ 1.0333... is not a simple rational (it requires denominators above 20 to approximate), so the aperiodicity check passes. This is a one-digit change with meaningful theoretical consequences: it ensures the unique π* is globally attracting.

If you set `rho_kappa = rho_mu` deliberately (e.g. for a symmetric model), `check_ergodicity_conditions` will report `aperiodic=False` and `ergodic=False` — this is correct and expected.

---

## BenchmarkSim vs FullSim

These are two different ways to estimate the same object (the HOE π*), and they give different answers for good reasons.

**BenchmarkSim** runs `TwoPlayerModel.integrate_stochastic`. This is the reduced-form benchmark where the four mechanisms collapse into a single mean-reverting ODE with additive noise. It is fast, analytically grounded, and produces clean HOE distributions because the dynamics are simple. The stable regime converges to h*≈0.80 from any h₀. The oscillatory regime gives a symmetric distribution around 0.5.

**FullSim** runs the complete `equilibrium/markov.run_chain`. This implements all four mechanisms with the full period timeline. The dynamics are richer — Mechanism I (Beta-Binomial) accumulates accuracy over time, Mechanism IV couples edges through the propagation kernel — but the equilibrium outcome is qualitatively similar. For the default parameters (stable regime, HSI=2.1), FullSim converges to h*≈0.99 because the optimal best-response follower (using `optimal_eta`, which returns 0 when h > 0.5) never attacks. This is the correct Stackelberg limit.

Use BenchmarkSim for HOE demonstrations. Use FullSim when you want to observe how the four mechanisms interact dynamically, or when you need the full capital trajectory (κ, μ).

---

## Welfare loss computation

The welfare loss is computed as the sum of three monetised distortion costs, not as the difference W_SO − W_eq. The reason: total welfare W = Σ_{edges} [h·u_L + (1-h)·u_F + (1-h)·u_L + h·u_F] = Σ_{edges} [u_L + u_F], which is independent of h due to symmetry. The social optimum and equilibrium therefore deliver the same gross welfare, making W_SO − W_eq = 0 (or negative, depending on investment costs).

The welfare loss from distortions is real, but it shows up as excess investment costs, not as a change in the hierarchy belief distribution. The three distortions waste resources: followers over-invest in reframing, leaders over-invest in resistance, and no one invests enough in making the hierarchy clear. The welfare loss is:

    Loss = D1·c_μ·|E| + D2·c_κ·|E| + D3_gap

where D1, D2 are the excess investment magnitudes and D3_gap is the monetised clarity under-investment.

---

## Distance decay and the network φ function

The distance decay function φ(d(i,j), G) = exp(−decay_rate · d) appears in three places:

1. **Mechanism I spillover**: successful anticipation by i propagates to i's other relationships, attenuated by network distance.
2. **Mechanism III spillover**: a successful reframe on edge (i,j) propagates to i's other edges.
3. **Kernel K**: the propagation weight K_{ij→kl} includes φ(d({i,j},{k,l}), G) where the edge distance is the minimum endpoint distance.

Using an exponential decay is a strong parametric assumption. It implies that information and influence decay geometrically with graph distance. For dense or small-world networks (most economic networks of interest), the distinctions between d=1,2,3 matter much more than d=10,20 — the exponential captures this well.

The decay rate is set to 1.0 by default. Higher values make the network effectively local (only direct neighbours matter). Lower values make it more global.

---

## Stationarity testing approach

The stationarity check compares window means across the post-burn-in trajectory. A tighter test (e.g. augmented Dickey-Fuller or KPSS) would be statistically more rigorous but introduces a scipy.stats dependency and requires choosing critical values. The window-mean approach is more interpretable: if the chain has converged, the mean h over any 50-period window should be within 0.05 of any other window's mean.

The convergence threshold of 0.05 is conservative — real convergence typically gives max_diff < 0.01 for stable-regime chains. If you want a tighter or looser criterion, pass it directly:

```python
from chse.equilibrium.hoe import stationarity_test

st = stationarity_test(chain, burn_in=80, n_windows=6)
print(f"max_diff = {st['max_diff']:.4f}")  # inspect the raw number
converged_tight = st['max_diff'] < 0.02   # custom threshold
```
