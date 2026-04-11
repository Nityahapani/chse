# Parameter Reference

Every parameter in CHSE, with default values, units, and economic interpretation. All parameters live in `Params` (`chse.core.primitives`).

---

## Composite indices

These two numbers carry essentially all predictive content. Everything else is a mechanism.

| Index | Formula | Interpretation |
|-------|---------|----------------|
| **HSI** | λ_κ · K_i / (λ_R · V_j) | Ratio of leader resistance capacity to follower attack capacity. HSI > 1: stability favoured. HSI < 1: turnover favoured. |
| **PI** | Γ · E[φ(d, G)] | Network-wide cascade potential. PI < 0.5: local dynamics dominate. PI > 0.5: contagion dominates. |
| **Z** | HSI⁻¹ · (1 + 2·PI) | Instability index. Determines regime (see phase diagram). |

---

## Efficiency parameters

These control how effectively each mechanism converts capital into belief changes.

| Parameter | Symbol | Default | Mechanism | Interpretation |
|-----------|--------|---------|-----------|----------------|
| `lambda_kappa` | λ_κ | 1.0 | III | Resistance efficiency: rate at which credibility capital spend converts to reframe-resistance ρ. Higher → same κ spend gives stronger protection. |
| `lambda_R` | λ_R | 1.0 | III | Reframing efficiency: rate at which narrative capital spend converts to attack power. Higher → same η spend gives higher P_R. |
| `lambda_sigma` | λ_σ | 1.0 | I | Suppression efficiency: rate at which opacity investment δ converts to signal suppression σ. Higher → smaller δ needed to suppress anticipation signals. |

---

## Capital caps and replenishment

| Parameter | Symbol | Default | Interpretation |
|-----------|--------|---------|----------------|
| `K_cap` | K_i | 10.0 | Maximum credibility capital per player. Hard cap — excess is wasted. |
| `M_cap` | M_i | 10.0 | Maximum manipulation capital per player. |
| `rho_kappa` | ρ_κ | 0.31 | Per-period credibility replenishment. Set to 0.31 (not 0.30) so ρ_κ/ρ_μ is irrational, ensuring chain aperiodicity. |
| `rho_mu` | ρ_μ | 0.50 | Per-period manipulation replenishment. Faster than credibility replenishment, reflecting that narrative capital is easier to rebuild than institutional credibility. |

**Note on aperiodicity:** The ergodicity proposition requires ρ_κ/ρ_μ to be irrational. With ρ_κ = 0.31 and ρ_μ = 0.30 (or 0.50), the ratio ≈ 1.033 (or 0.62) is not a simple rational — the aperiodicity condition is satisfied. Setting ρ_κ = ρ_μ breaks this.

---

## Mechanism-specific parameters

### Mechanism I — Anticipation

| Parameter | Symbol | Default | Interpretation |
|-----------|--------|---------|----------------|
| `alpha_I` | α_I | 0.2 | Direct impact of an anticipation success on h_ij. The belief Δ^I h_ij = α_I · ξ_ij upon a success. |
| `beta_I` | β_I | 0.1 | Predictability penalty on the opponent. The network spillover Δ^I h_jk = −β_I · ξ_ij · φ(d(j,k)) reduces j's credibility with their other partners. |

### Mechanism II — Role ambiguity

| Parameter | Symbol | Default | Interpretation |
|-----------|--------|---------|----------------|
| `mu_II` | μ_II | 1.0 | Ambiguity mean-reversion rate. Controls how strongly γ spend pushes h toward 0.5: Δ^II h = −μ_II · γ · (h − ½). Higher → ambiguity investment is more powerful. |
| `zeta_II` | ζ_II | 0.3 | Network spillover rate for role ambiguity. Ambiguity on edge (i,j) leaks to i's other edges: Δ^II h_ik = −ζ_II · γ · φ(d(i,k)). |

### Mechanism III — Retroactive reframing

| Parameter | Symbol | Default | Interpretation |
|-----------|--------|---------|----------------|
| `alpha_R` | α_R | 0.3 | Direct belief drop on a successful reframe: Δ^III h_ij = −α_R · P_R. The immediate damage to the leader's hierarchy belief. |
| `beta_R` | β_R | 0.1 | Network spillover of a successful reframe: Δ^III h_ik = −β_R · P_R · φ(d(i,k)). Reframing damage propagates to i's other relationships. |
| `delta_kappa` | δ_κ | 0.5 | Credibility capital lost per successful reframe: Δκ_i = −δ_κ · P_R. Reframing damages the leader's institutional capital, not just the belief. |

---

## Discount and cost parameters

| Parameter | Symbol | Default | Interpretation |
|-----------|--------|---------|----------------|
| `discount` | δ | 0.95 | Discount factor. Used in the Lyapunov stability condition: Γ < (1−δ)/(1+δ). Also affects the long-run value of holding leadership. |
| `c_mu` | c_μ | 0.5 | Marginal cost of reframing investment η. Enters the optimal best-response: η*(h) = (1/λ_R)·ln(α_R·λ_R·(½−h)/c_μ). Higher c_μ → follower attacks less aggressively. |
| `c_kappa` | c_κ | 0.5 | Marginal cost of credibility investment. Enters κ*(h) = (1/λ_κ)·ln(α_R·λ_κ·(h−½)/c_κ). Higher c_κ → leader defends less aggressively. |

---

## Override parameters

These bypass the per-state calculations and set the composite indices directly. Useful for the two-player benchmark and phase diagram exploration.

| Parameter | Default | When to use |
|-----------|---------|-------------|
| `HSI` | `None` | Set to a specific value (e.g. `HSI=2.1`) to force a regime without specifying K_i, V_j, λ_κ, λ_R individually. All methods that compute HSI return this value directly. |
| `PI` | `None` | Set to a specific value to force propagation intensity. When None, PI is computed from Γ and E[φ(d, G)]. |

---

## Regime thresholds

Derived from parameters, not themselves parameters:

| Threshold | Formula | Separates |
|-----------|---------|----------|
| Stability boundary | HSI · (1 + 2·PI) = 1 | Stable ↔ Oscillatory |
| Cascade boundary | ρ(K) = 1 − 1/HSI | Oscillatory ↔ Cascade |
| Turbulent boundary | Z = 3.5 | Cascade ↔ Turbulent |

---

## Calibration guidance

### Stable regime (HSI >> 1)

For a central bank with strong institutional independence:
```python
p = Params(
    lambda_kappa=2.0,  # strong resistance
    lambda_R=0.8,      # weaker reframing attacks
    K_cap=15.0,        # large credibility stock
    HSI=2.1,           # override for simplicity
)
```

### Oscillatory regime (HSI ≈ 1)

For a contested corporate hierarchy:
```python
p = Params(
    lambda_kappa=1.0,
    lambda_R=1.5,
    HSI=1.0,
)
```

### Cascade regime (HSI << 1)

For a fragile sovereign debtor under bilateral creditor pressure:
```python
p = Params(
    lambda_R=2.5,   # aggressive reframing
    K_cap=5.0,      # limited credibility stock
    HSI=0.4,
)
```

### Empirically calibrated (central bank)

Use the FDI formula to back out implied parameters:

```python
from chse.empirical.fdi import FDIEstimate

est = FDIEstimate(
    country='Turkey', period='2021-23',
    V_T=0.82,    # political capital index
    K_CB=0.45,   # Dincer-Eichengreen independence score
    lambda_R=1.0, rho_ratio=1.0,
)
print(f"Implied HSI ≈ {est.to_hsi():.3f}")
```
