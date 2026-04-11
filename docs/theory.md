# Formal Theory

This document presents the complete mathematical theory underlying CHSE. Every definition, assumption, theorem, and formula here corresponds directly to an implemented object in the codebase.

---

## 1. The core idea

The Stackelberg model assumes hierarchy as a structural primitive: one player commits first and the other best-responds. Real hierarchies are not like this. Central banks gain and lose independence. Corporate authority is contested and sometimes reversed overnight. Creditor seniority in sovereign debt restructuring is negotiated in real time.

CHSE formalises a class of dynamic games in which the hierarchical structure is **endogenous, fluid, and socially embedded**. The Stackelberg hierarchy — who leads, who follows — is not an input to the game but an output.

---

## 2. Primitives

### 2.1 Players and network

Let N = {1, …, n} be a set of players connected by an undirected graph G = (N, E).

### 2.2 Hierarchy belief

**Definition 2.1 (Hierarchy Belief).** For each directed pair (i, j) with {i, j} ∈ E, the *hierarchy belief* h_ij(t) ∈ [0, 1] is the network-wide probability assigned at time t to the event that i leads j.

The belief state H(t) = (h_ij(t)) satisfies the **coherence constraint**:

    h_ij(t) + h_ji(t) = 1    for all {i,j} ∈ E

Interpretation:
- h = 1 → pure Stackelberg leadership (i leads j)
- h = 0.5 → full role ambiguity (Nash-like, neither player commits)
- h = 0 → pure followership (j leads i)

### 2.3 Stage game payoff

    U_i(h_ij) = h_ij · u_i^L + (1 − h_ij) · u_i^F

where u_i^L > u_i^F (Assumption 2.1: Leadership Premium). The payoff is a convex combination of the leadership payoff and the followership payoff, weighted by the current hierarchy belief.

### 2.4 Resource stocks

Each player i holds two capital stocks:

| Symbol | Name | Funds | Cap | Replenishes |
|--------|------|-------|-----|-------------|
| κ_i(t) | Credibility capital | Commitment resistance (Mech. III) | K_i | ρ_κ per period |
| μ_i(t) | Manipulation capital | Ambiguity (Mech. II), suppression (Mech. I), reframing (Mech. III) | M_i | ρ_μ per period |

Capital evolves as:

    κ_i(t+1) = min(K_i,  κ_i(t) + ρ_κ − M_i^κ(t) − δ_κ · Σ_j P_{R,ij}(t))
    μ_i(t+1) = min(M_i,  μ_i(t) + ρ_μ − M_i^μ(t))

### 2.5 Composite indices

The full model has 15 raw parameters. Two composite indices carry essentially all predictive content.

**Definition 2.2 (Hierarchy Stability Index).** For edge (i, j) with i currently leading:

    HSI_ij = (λ_κ · K_i) / (λ_R · V_j)

HSI measures the ratio of i's resistance capacity to j's attack capacity. HSI > 1 favours stability; HSI < 1 favours turnover.

**Definition 2.3 (Propagation Intensity).**

    PI = Γ · E[φ(d, G)]

PI measures the network-wide cascade potential: the spectral contraction factor Γ times the average distance-decay, determining how strongly a belief shift on one edge propagates across the network.

**Instability Index:**

    Z = HSI⁻¹ · (1 + 2·PI)

---

## 3. The four mechanisms

### 3.1 Mechanism I — Anticipation as a public good

Each player has a latent predictive type θ_i ∈ Θ distributed with prior F(θ). Anticipation success ξ_ij(t) is a noisy signal of the comparison θ_i > θ_j. The hierarchy belief is the posterior:

    h_ij(t+1) = P(θ_i > θ_j | ξ_ij(1), …, ξ_ij(t))

**Beta-Binomial updating:**

    α_i^(t+1) / (α_i^(t+1) + β_i^(t+1)) = (α_i^(t) + ξ_ij(t)) / (α_i^(t) + β_i^(t) + 1)

Successful anticipation is publicly observable unless suppressed. The **signal externality** propagates to all network neighbours:

    Δ^I h_ik(t) = α_I · ξ_ij(t) · (1 − σ_ij) · φ(d(i,k), G)    ∀k ≠ i
    Δ^I h_jk(t) = −β_I · ξ_ij(t) · (1 − σ_ij) · φ(d(j,k), G)    ∀k ≠ j

**Suppression technology** (opacity investment δ ≥ 0):

    σ_ij(δ) = 1 − exp(−λ_σ · δ)

### 3.2 Mechanism II — Role ambiguity as a strategic instrument

Player i spends γ_ij(t) ≥ 0 units of manipulation capital on role ambiguity on edge (i, j). The mean-reverting belief shift:

    Δ^II h_ij(t) = −μ_II · γ_ij(t) · (h_ij(t) − ½)

Network-wide leadership credibility spillover:

    Δ^II h_ik(t) = −ζ_II · γ_ij(t) · φ(d(i,k), G)    ∀k ≠ j

The ambiguity distortion factor D interpolates between hierarchical and Nash payoffs:

    D(h_ij) = 1 − 4(h_ij − ½)²
    U_i^amb(h_ij) = (1 − D(h_ij)) · U_i(h_ij) + D(h_ij) · U_i^Nash

### 3.3 Mechanism III — Retroactive reframing

Follower j spends narrative capital η_ij(t) to attack leader i's past commitment. The probability of a successful reframe:

    P_R(η_ij, ρ_i) = (1 − exp(−λ_R · η_ij)) · (1 − ρ_i)

Leader i builds **reframe-resistance** by spending credibility capital c_i at the time of commitment:

    ρ_i(τ) = 1 − exp(−λ_κ · c_i)

Upon a successful reframe, belief and capital updates:

    Δ^III h_ij = −α_R · P_R
    Δ^III h_ik = −β_R · P_R · φ(d(i,k), G)
    Δ^III κ_i = −δ_κ · P_R    (credibility depletes)

**Optimal best-response functions** (from payoff maximisation):

    η*(h) = (1/λ_R) · ln(α_R · λ_R · (½ − h) / c_μ)    if h < ½,  else 0
    κ*(h) = (1/λ_κ) · ln(α_R · λ_κ · (h − ½) / c_κ)    if h > ½,  else 0

### 3.4 Mechanism IV — Contagious propagation (endogenous kernel)

The propagation kernel K is endogenous — a function of trust accumulated through past anticipation accuracy and network centrality:

    K_{ij→kl}(t) = Acc_ij(t) · Trust_{ij→kl}(t) · φ(d({i,j},{k,l}), G) / Z

where Acc_ij(t) is the time-averaged anticipation accuracy on edge (i,j) and Trust_{ij→kl}(t) is the cross-edge trust level (updated each period).

The total belief update from propagation:

    Δ^IV h_kl(t) = Σ_{(i,j)≠(k,l)} K_{ij→kl} · (Δ^I + Δ^II + Δ^III) h_ij

---

## 4. Full dynamic system

### 4.1 Complete belief update

    h_ij(t+1) = Proj_{[0,1]} { h_ij(t) + Δ^I h_ij + Δ^II h_ij + Δ^III h_ij + Δ^IV h_ij }

### 4.2 Period timeline

| Stage | Event | State updated |
|-------|-------|---------------|
| 1 | Investment portfolios chosen | — |
| 2 | Stage game resolves; payoffs realised | Payoffs |
| 3 | Anticipation signals ξ_ij(t) realised; suppression applied | Anticipation history |
| 4 | H(t+1) computed via equations above | H(t) |
| 5 | κ, μ updated via depletion and replenishment | κ(t), μ(t) |
| 6 | Kernel K applied; cascade check performed | H(t+1) adjusted |

### 4.3 Boundary behaviour

- As h_ij(t) → 1: Mechanisms I and III dominate; follower j intensifies reframing; leader i maximises commitment resistance. The boundary is absorbing only if HSI → ∞.
- As h_ij(t) → 0: Symmetric case with roles reversed.
- When κ_i(t) hits cap K_i: excess commitment capital is wasted, creating a saturation effect that weakens the leader at the top of their cycle.
- When μ_i(t) = 0: no ambiguity or reframing is feasible; i plays transparently, making them maximally predictable — potentially exploitable.

---

## 5. Two-player benchmark

With N = {1, 2}, one edge, write h = h_12. In the deterministic limit (suppressing stochastic anticipation), the belief evolution reduces to:

    ḣ(t) = −μ(h(t) − ½) + η(t) − κ(t)·r(t)

where μ > 0 is the mean-reversion strength, η(t) is player 2's reframing investment, and κ(t)·r(t) is player 1's commitment capital deployed as resistance.

**Fixed point:** Setting ḣ = 0:

    h* = ½ + (η̄ − κ̄·r̄) / μ

**Definition 3.1 (Oscillation Condition).** The two-player benchmark exhibits oscillatory dynamics iff:

    μ² < 4·η̄·κ̄

The characteristic equation of the linearised system is λ² + μλ + η̄κ̄ = 0. When the discriminant μ² − 4η̄κ̄ < 0, eigenvalues are complex and the system spirals around h*.

**Definition 3.2 (Flip Time).** Starting from h(0) = h₀ > ½, the first time h crosses ½ is:

    t* = (1/μ̃) · ln((h₀ − ½) / ε)

where μ̃ = |Re(λ)| is the effective decay rate and ε > 0 is the flip precision threshold. Flip time is decreasing in λ_R (reframing efficiency) and increasing in HSI.

---

## 6. Phase diagram

**Definition 6.1 (Regime Classification).**

| Z | Regime | Characterisation |
|---|--------|-----------------|
| Z < 1 | Stable Hierarchy | H(t) converges to fixed point; leadership permanent |
| 1 ≤ Z < 2 | Oscillatory | H(t) cycles; leadership alternates periodically |
| 2 ≤ Z < 3.5 | Cascade-Dominated | Large-amplitude oscillations; frequent network-wide cascades |
| Z ≥ 3.5 | Turbulent | Sensitive dependence on initial conditions |

**Theorem 6.1 (Phase Boundary I — Stability to Oscillation).** The boundary between stable and oscillatory regimes is:

    HSI · (1 + 2·PI) = 1

Below this curve, the two-player benchmark fixed point is globally stable. Above it, the oscillation condition of Definition 3.1 is satisfied.

**Theorem 6.2 (Phase Boundary II — Cascade Threshold).** Cascade-dominated dynamics emerge when:

    ρ(K) ≥ 1 − (1/HSI)

This boundary depends jointly on PI (through ρ(K)) and HSI (through the local resistance to flips).

---

## 7. Cascades as spectral threshold processes

**Proposition 7.1 (Spectral Cascade Condition).**

    ρ(K) < 1  →  no infinite cascade
    ρ(K) ≥ 1  →  cascade possible

When ρ(K) < 1, expected cascade size is bounded:

    E[cascade size] ≤ α_R / (1 − ρ(K))

**Proposition 7.2 (Strategic Cascade Manipulation).** Because K is endogenous, players can strategically manipulate cascade potential. Player i increases ρ(K) by building trust and accuracy on edges adjacent to opponents' fragile relationships (h near 0.5), thereby increasing cascade probability when a reframe succeeds.

---

## 8. Hierarchy Orbit Equilibrium

### 8.1 Induced Markov process

The CHSE dynamical system defines a Markov process on the compact state space:

    S = H × ∏_i [0, K_i] × [0, M_i]

Let P(s, B) denote the transition kernel: the probability that the state moves from s into set B ⊆ S in one period under optimal structural move portfolios.

### 8.2 Definition

**Definition 8.1 (Hierarchy Orbit Equilibrium).** A probability measure π* on (S, B(S)) is a Hierarchy Orbit Equilibrium if it is an invariant measure of the induced Markov process:

    π*(B) = ∫_S P(s, B) π*(ds)    for all B ∈ B(S)

and if the structural move portfolios supporting P(s, ·) are individually optimal at π*-almost every s ∈ S.

A fixed-point strategy profile (Nash, Stackelberg) corresponds to π* = δ_{s*} — a Dirac mass at a single point. HOE generalises this to non-degenerate invariant measures.

### 8.3 Existence

**Theorem 8.1 (Existence of HOE).** Under Assumptions 2.1 (resource boundedness), 2.2 (leadership premium), and 4.1 (propagation contraction), and with discount factor δ ∈ (0, 1), the induced Markov process on S has at least one invariant measure π*, constituting an HOE.

*Proof sketch.* S is compact (Tychonoff). The optimal best-response correspondence is upper hemicontinuous by Berge's Maximum Theorem. The transition kernel P(s, ·) is therefore Feller. By the Krylov-Bogolyubov theorem, every Feller Markov chain on a compact metric space admits an invariant probability measure. ∎

### 8.4 Ergodicity

**Proposition (Ergodicity).** If the induced Markov process is:

1. **Irreducible** — every open set B ⊂ S has positive probability of being reached from any s in finite steps (sufficient condition: all λ_R, λ_κ, λ_σ > 0 and ρ_κ, ρ_μ > 0 ensuring full state-space reachability).
2. **Aperiodic** — sufficient condition: ρ_κ/ρ_μ is irrational (i.e. not a rational p/q for small p, q).

Then π* is unique and the process converges from any initial condition:

    ‖P^t(s, ·) − π*‖_TV → 0    as t → ∞

The default parameters satisfy both conditions: ρ_κ = 0.31, ρ_μ = 0.30, ratio ≈ 1.033 (irrational).

### 8.5 Lyapunov stability

**Definition 8.2 (Lyapunov Function).**

    V(s) = d(s, Ω*)² + Σ_i [θ_κ(κ_i − κ_i*(s))² + θ_μ(μ_i − μ_i*(s))²]

where Ω* = supp(π*) is the HOE orbit support and s*(s) = argmin_{s*∈Ω*} ‖s − s*‖.

**Theorem 8.2 (Lyapunov Stability).** If the propagation factor satisfies:

    Γ < (1 − δ) / (1 + δ)

then π* is Lyapunov stable. With positive resource replenishment rates and sufficiently large efficiency parameters (λ_R, λ_κ, λ_σ > λ*), π* is asymptotically stable.

---

## 9. HOE empirical statistics

**Definition 10.1 (Leadership Turnover Frequency).**

    τ̂ = (1/|E|·T) · Σ_{(i,j)∈E} Σ_{t=1}^T 𝟙[h_ij(t) ≠ h_ij(t−1) > ½]

**Definition 10.2 (HOE as Stationary Distribution).** The HOE π* is identified empirically as the stationary distribution over the triple (τ̂, Var(h), E[cascade size]):

    HOE ≡ π*(τ̂,  Var(h),  E[cascade size])

---

## 10. Welfare distortions

In any interior HOE π*, relative to the social optimum, there are three distortions:

**Distortion 1 — Over-investment in reframing.** Each follower j ignores the negative externality of reframing on adjacent edges (network spillover −β_R · P_R · φ). Excess investment:

    Excess_1 ∝ β_R · Γ / (1 − Γ)

Policy fix: legal estoppel, institutional precedent.

**Distortion 2 — Over-investment in commitment resistance.** Each leader i ignores that reframe-resistant commitments are more legible to third parties (positive externality). Excess investment:

    Excess_2 ∝ Σ_k φ(d(i,k)) · u_k^coordination

Policy fix: legibility subsidies, transparent announcements.

**Distortion 3 — Under-investment in hierarchy clarity.** Hierarchy h_ij = 1 (or 0) is a public good — it reduces ambiguity costs for all network actors interacting with i or j. Under-investment:

    Deficit_3 ∝ ζ_II · |E_i|    (network-wide ambiguity spillover × degree)

Policy fix: public commitment requirements (central bank mandates, board resolutions).

---

## 11. Hierarchy Persistence Paradox

**Theorem (Hierarchy Persistence Paradox).** Conditional on a leadership collapse (h_ij crossing below 0.5), the expected cascade size is *increasing* in HSI:

    ∂E[cascade size | collapse] / ∂HSI > 0

*Causal chain:*

1. ∂Acc_ij / ∂HSI > 0 — stronger leaders accumulate more anticipation successes
2. ∂ρ(K) / ∂Acc_ij > 0 — higher accuracy inflates propagation weights in K
3. ∂E[cascade] / ∂ρ(K) = α_R / (1 − ρ(K))² > 0

A high-HSI leader's strength is precisely what makes their fall catastrophic.

*Testable predictions:*
- Long-tenured, high-credibility central banks produce larger financial market disruptions when independence is lost.
- Dominant firms produce larger supply-chain cascades when authority collapses.
- Hegemonic creditors produce larger debt restructuring crises than marginal ones.

---

## 12. Connections to existing equilibrium concepts

| Concept | HOE relation |
|---------|-------------|
| Nash Equilibrium | Special case: π* = δ_{Nash}, h* = 0.5 fixed |
| Stackelberg Equilibrium | Special case: π* = δ_{Stack}, HSI → ∞ |
| Markov Perfect Equilibrium | HOE generalises: adds endogenous h, contested structure |
| Self-confirming Equilibrium | HOE generalises: adds network embedding and structural belief updating |
| Evolutionary Stable Strategy | Special case of HOE in large-N symmetric network |
| Percolation / network cascades | HOE adds strategic manipulation of cascade threshold |

**Proposition 9.1 (HOE as Generalisation).**
- (a) Nash equilibrium is the HOE π* = δ_Nash in the limit HSI → 0, h* = 0.5.
- (b) Stackelberg equilibrium is the HOE π* = δ_Stack in the limit HSI → ∞, h* ∈ {0, 1}.
- (c) Markov Perfect Equilibrium is the HOE of a CHSE game with exogenous K (constant kernel, no endogenous propagation).
- (d) Every Evolutionarily Stable Strategy of a symmetric population game is an HOE of the corresponding large-N CHSE network.

---

## 13. Fiscal Dominance Index

**Corollary 11.1 (Endogenous Fiscal Dominance).** In the CB-Treasury CHSE game, fiscal dominance (h_{CB,T} < 0.5 recurrently in π*) occurs if and only if the Fiscal Dominance Index satisfies FDI > 1, where:

    FDI = V_T · λ_R / (K_CB · ρ_κ/ρ_ν)

| FDI | Regime |
|-----|--------|
| FDI < 0.5 | Robust monetary dominance |
| 0.5 ≤ FDI ≤ 1 | Contested regime with frequent oscillation |
| FDI > 1 | Recurrent fiscal dominance |
