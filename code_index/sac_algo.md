# Residual RL SAC — Algorithm Description

This document describes the **algorithmic details** of the Residual RL framework implemented in this codebase. It focuses on the mathematical formulations, optimization objectives, and design rationale behind every component. This is intended as a self-contained reference for understanding *what* the algorithms do and *why*, independent of the code structure.

---

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [Action Composition](#2-action-composition)
3. [Policy Representation](#3-policy-representation)
4. [Critic (Q-Function)](#4-critic-q-function)
5. [Algorithm 1: Residual SAC](#5-algorithm-1-residual-sac)
6. [Algorithm 2: GRPO / Q-Weighted Policy Gradient](#6-algorithm-2-grpo--q-weighted-policy-gradient)
7. [Algorithm 3: On-Policy PPO](#7-algorithm-3-on-policy-ppo)
8. [Algorithm 4: PARL (Policy-Agnostic RL)](#8-algorithm-4-parl-policy-agnostic-rl)
9. [Algorithm 5: GradQ](#9-algorithm-5-gradq)
10. [Temperature Auto-Tuning](#10-temperature-auto-tuning)
11. [BC Warmup](#11-bc-warmup)
12. [BC Regularization](#12-bc-regularization)
13. [Critic Update (Shared Across All Algorithms)](#13-critic-update-shared-across-all-algorithms)
14. [Data Collection (Rollout Strategy)](#14-data-collection-rollout-strategy)
15. [Reward Shaping](#15-reward-shaping)
16. [Numerical Stability Techniques](#16-numerical-stability-techniques)
17. [Diagnostics: Collapse and Dominance Detection](#17-diagnostics-collapse-and-dominance-detection)
18. [Training Loop Structure](#18-training-loop-structure)
19. [Hyperparameter Summary](#19-hyperparameter-summary)

---

## 1. Problem Formulation

We have a pre-trained, **frozen** base policy `π_base` (e.g., Pi-0.5, a vision-language-action diffusion model) that produces action chunks `a_base ∈ ℝ^{T×A}` given observations, where `T` is the chunk length (e.g., 10) and `A` is the action dimension (e.g., 7 for a robotic arm).

The base policy is good but imperfect. We want to learn a lightweight **residual policy** `π_θ` that outputs a small correction `δ ∈ ℝ^{T'×A}` (where `T' ≤ T` is the query frequency) to improve task success rate while staying close to the base policy's behavior.

The MDP is defined as:
- **State**: `s = (o, a_base)` — the environment observation `o` augmented with the base policy's proposed action
- **Action**: `δ` — the residual correction (or alternatively, the full executed action `a_exec` directly)
- **Transition**: The environment receives `a_exec = clip(a_base + α·δ, -1, 1)` and produces `s'`
- **Reward**: Sparse (`r=-1` per step, `r=0` on success) or dense (raw env reward)

The key constraint is that the base policy `π_base` is **never fine-tuned**. It is queried as a black box at each decision point, using its own internal stochasticity (diffusion noise) to produce diverse base actions.

---

## 2. Action Composition

### Default Mode: Delta Prediction (`predict_a_exec=False`)

The residual policy `π_θ(s)` outputs a delta `δ ∈ (-1, 1)^{T'×A}` (bounded by tanh squashing). The executed action is:

```
a_exec = clip(a_base[:T'] + α · δ, -1, 1)
```

where:
- `α ∈ ℝ+` is the **residual scaling factor** (default 0.1)
- `clip(·, -1, 1)` enforces environment action bounds
- Only the first `T'` steps of the base action chunk are used (the rest are executed open-loop from the base policy)

The parameter `α` controls the trust region. Small `α` (e.g., 0.1) constrains the residual to small perturbations, ensuring the agent cannot deviate far from the base policy. Larger `α` allows more aggressive corrections.

### Alternative Mode: Direct Execution Prediction (`predict_a_exec=True`)

The policy directly outputs `a_exec ∈ (-1, 1)^{T'×A}`:

```
a_exec = clip(π_θ(s), -1, 1)
```

In this mode, `α` is unused for composition (only logged for diagnostics). The policy must learn both the base behavior and any corrections from scratch (though BC warmup helps).

### Soft-Clip for Gradient Flow (Actor Only)

During the actor's forward pass in gradient computation, standard `clip()` has zero gradient at boundaries, which can trap the actor. Instead, a **differentiable soft-clip** is used:

```
soft_clip(x) = mid + half_range · tanh((x - mid) / half_range)
```

where `mid = 0`, `half_range = 1` for the `[-1, 1]` range. This smoothly saturates near the boundaries while preserving non-zero gradients everywhere. It is equivalent to identity in the interior and tanh-like at the edges.

This soft-clip is used **only in the actor update** (where gradients must flow through the composition). The critic update and data collection use standard hard `clip()`.

---

## 3. Policy Representation

### TanhNormal Policy

The residual policy is a **TanhNormal** (squashed Gaussian) distribution:

```
z ~ N(μ_θ(s), σ_θ(s)²)          (pre-tanh sample)
δ = tanh(z)                       (squashed to (-1, 1))
```

where `μ_θ(s)` and `σ_θ(s)` are neural network outputs:

```
h = MLP(flatten(s))               # s includes encoded pixels, state, base_action
μ = Linear(h)                     # (T' × A,) — mean
log_σ = clip(Linear(h), log_σ_min, log_σ_max)  # (T' × A,) — log std
σ = exp(log_σ)
```

Default bounds: `log_σ_min = -5.0`, `log_σ_max = 2.0`.

### Log-Probability Computation

The log-probability under TanhNormal includes the **change-of-variables Jacobian** for the tanh bijector:

```
log π(δ|s) = log N(z; μ, σ²) - Σᵢ log(1 - tanh²(zᵢ))
```

The second term is the log-determinant of the Jacobian of tanh. It is computed from the pre-tanh sample `z` (not via `atanh(δ)`, which would be numerically unstable near ±1).

This Jacobian correction is **critical** for the SAC actor loss. It provides gradient to the mean `μ` that pushes it away from the tanh saturation boundaries. Without it, `μ` can grow unboundedly, causing all actions to saturate at ±1.

### Fixed-Std Variant

When `learn_std=False`, the policy uses a fixed standard deviation:

```
σ = exp(fixed_log_std)            # e.g., exp(-0.5) ≈ 0.607
```

This prevents **std collapse** — a failure mode where the learned std shrinks to near-zero during BC warmup and fails to recover during RL (since RL requires exploration via stochasticity).

### Mode (Deterministic) Action

For evaluation and data collection:

```
δ_mode = tanh(μ_θ(s))
```

This is the most likely action under the policy. It is used for rollouts (not sampling) because exploration comes from the stochastic base policy, not the residual.

---

## 4. Critic (Q-Function)

### Ensemble Architecture

The critic is an ensemble of `N` independent Q-functions (default `N=10`):

```
Q_i(s, a_exec) = MLP_i(flatten(s) ⊕ flatten(a_exec))   for i = 1, ..., N
```

**Critical**: The critic takes the **composed executed action** `a_exec`, not the raw delta `δ`. The action composition `a_exec = clip(a_base + α·δ, -1, 1)` happens before the critic sees the action.

### Ensemble Reduction

Two strategies for aggregating across the ensemble:

- **`'min'`**: `Q(s,a) = min_i Q_i(s,a)` — conservative, underestimates (standard SAC)
- **`'mean'`**: `Q(s,a) = (1/N) Σ_i Q_i(s,a)` — less conservative, may overestimate (default for residual)

The mean reduction is preferred for residual RL because conservative underestimation can cause the residual to collapse to zero (why deviate from the base if the critic says everything is equally bad?).

### Observation Conditioning

The actor observes `base_action` (to condition the residual on what the base policy proposed). The critic can optionally **not** observe `base_action` (`pop_base_actions=True`, default), so it evaluates `Q(o, a_exec)` purely based on the observation and executed action. This prevents the critic from memorizing a shortcut based on the base action rather than learning the true state-action value.

---

## 5. Algorithm 1: Residual SAC

### Actor Loss

Standard SAC objective adapted for residual actions:

```
δ, log π(δ|s) = π_θ.sample_and_log_prob(s)
a_exec = soft_clip(a_base[:T'] + α · δ, -1, 1)
Q = reduce(Q_1(s, a_exec), ..., Q_N(s, a_exec))

L_actor = E_s[ α_temp · log π(δ|s) - Q(s, a_exec) ]
```

where:
- `α_temp` is the automatically-tuned temperature (entropy coefficient)
- `soft_clip` preserves gradients at action boundaries
- `log π(δ|s)` includes the full tanh Jacobian correction
- `log π(δ|s)` is clipped to `[-50, 50]` for numerical safety

The loss encourages the actor to:
1. **Maximize Q** — find residuals that improve the base policy
2. **Maximize entropy** — maintain exploration (weighted by `α_temp`)

### Gradient Path

The actor gradient flows through:
```
θ → μ_θ, σ_θ → z (reparameterized) → δ = tanh(z) → a_exec = soft_clip(base + α·δ) → Q(s, a_exec)
```

Every step in this chain has non-zero gradient (tanh has non-zero gradient except at ±∞, soft_clip has non-zero gradient everywhere, Q is differentiable).

### When predict_a_exec=True

```
a_exec = π_θ(s)                   # already in (-1, 1) from tanh
Q = reduce(Q_i(s, a_exec))

L_actor = E_s[ α_temp · log π(a_exec|s) - Q(s, a_exec) ]
```

No soft-clip is applied because the tanh output is already bounded. Applying soft-clip or clip would create a "double squashing" effect.

---

## 6. Algorithm 2: GRPO / Q-Weighted Policy Gradient

### Overview

Instead of differentiating through the critic (as in SAC), this algorithm uses **advantage-weighted regression**: sample multiple actions, evaluate their Q-values, and increase the probability of high-Q actions.

### Procedure

```
For each state s in the batch:
  1. Sample G actions from current policy (stop gradient):
     δ_g ~ π_θ(·|s)    for g = 1, ..., G

  2. Compute Q-values for each candidate:
     a_exec_g = clip(a_base + α · δ_g, -1, 1)
     Q_g = Q_target(s, a_exec_g)

  3. Compute advantages:
     GRPO mode:     A_g = Q_g - (1/G) Σ_g Q_g    (subtract group mean)
     Q-weighted PG: A_g = Q_g                       (raw Q, no baseline)

  4. [Optional] Normalize: A_g = (A_g - mean(A)) / (std(A) + ε)
  5. [Optional] Clip: A_g = clip(A_g, adv_min, adv_max)
  6. Stop gradient on A_g

  7. Actor loss (advantage-weighted log-prob):
     L_pg = -(1/(G·B)) Σ_{g,b} A_g · log π_θ(δ_g | s_b)

  8. Entropy bonus:
     H = -(1/(G·B)) Σ_{g,b} log π_θ(δ_g | s_b)
     L_entropy = -β · H

  9. Total: L_actor = L_pg + L_entropy + λ_bc · L_bc
```

### Key Design Decisions

- **No importance ratios**: Actions are sampled fresh from the current policy each call (stop-gradientd). There is no "old" vs "new" policy — the stop-gradient creates the target distribution and the differentiable forward pass creates the current distribution. This is NOT PPO despite the function name.
- **Target critic** for Q evaluation: Uses `Q_target` (slowly-updated copy) to avoid feedback loops.
- **GRPO baseline**: Subtracting the group mean `(1/G)Σ Q_g` reduces variance. Actions better than the group average get positive advantage; worse ones get negative. This is more stable than using raw Q values which may all be large negative numbers (in sparse reward settings).
- **Safe log-prob**: Before computing `log π(δ_g|s)`, actions are clamped to `(-1+ε, 1-ε)` to avoid `atanh(±1) = ±∞`.

---

## 7. Algorithm 3: On-Policy PPO

### Overview

True Proximal Policy Optimization with stored **importance weights**. Unlike Algorithm 2 (which samples fresh actions), this algorithm reuses the **actual actions taken during data collection** with their stored log-probabilities.

### Procedure

```
Batch is sampled from the MOST RECENTLY collected trajectory only.
Each transition includes:
  - δ_stored: the action that was actually taken
  - log π_old(δ_stored|s): the log-prob at collection time

For each (s, δ_stored, log π_old) in batch:
  1. Compose a_exec from stored action:
     a_exec = clip(a_base + α · δ_stored, -1, 1)

  2. Evaluate Q as advantage:
     A = Q_target(s, a_exec)     (bandit-style, no baseline subtraction)

  3. [Optional] Normalize advantages

  4. Compute current log-prob of stored action:
     log π_θ(δ_stored|s)
     (clamped to (-1+ε, 1-ε) before atanh to avoid ±∞)

  5. Importance ratio:
     log_ratio = log π_θ(δ_stored|s) - log π_old(δ_stored|s)
     log_ratio = clip(log_ratio, -20, 20)    (stability)
     ρ = exp(log_ratio)

  6. PPO clipped surrogate loss:
     L_unclipped = -ρ · A
     L_clipped   = -clip(ρ, 1-ε, 1+ε) · A
     L_pg = max(L_unclipped, L_clipped)      (pessimistic — take the worse case)

  7. Entropy bonus (via fresh sample, not stored action):
     δ_fresh, log π = π_θ.sample_and_log_prob(s)
     H = -mean(log π)
     L_entropy = -β · H

  8. Total: L_actor = L_pg + L_entropy + λ_bc · L_bc
```

### Key Design Decisions

- **On-policy**: Only uses data from the last trajectory, ensuring `π_old ≈ π_θ`.
- **Bandit-style advantages**: Uses raw Q values (no value function baseline). This works because each state is seen only once per trajectory.
- **Asymmetric clipping**: `clip(ρ, 1-ε·c_min, 1+ε·c_max)` allows different lower and upper multipliers.
- **Separate entropy computation**: Entropy is estimated from a **fresh sample** (not the stored action) to get proper TanhNormal entropy with the Jacobian correction. Computing entropy from stored actions would require `atanh(stored_action)` which is unstable at boundaries.
- **Log-ratio clipping**: Before exponentiation, `log_ratio` is clamped to `[-20, 20]` to prevent numerical overflow/underflow.

---

## 8. Algorithm 4: PARL (Policy-Agnostic RL)

### Overview

PARL completely decouples the actor from policy gradient. Instead of maximizing `E[Q(s, π(s))]` through the reparameterization trick, PARL:
1. Uses the critic to **find high-Q actions** via sampling + gradient ascent
2. Trains the actor to **imitate** those high-Q actions via supervised learning (MSE)

The actor never differentiates through the critic. This avoids all gradient issues from tanh squashing, action clipping, and critic approximation errors.

### Procedure

```
For each state s in the batch:
  1. SAMPLE: Draw N candidates from current actor (stop-gradient):
     δ_n ~ π_θ(·|s)    for n = 1, ..., N
     Convert to a_exec space:
     a_n = clip(a_base + α · δ_n, -1, 1)

  2. INCLUDE BASE: Add the base policy action as (N+1)-th candidate:
     a_{N+1} = clip(a_base, -1, 1)

  3. EVALUATE: Compute Q for all N+1 candidates:
     Q_n = Q(s, a_n)    for n = 1, ..., N+1

  4. SELECT ELITES: Keep the top-K candidates by Q-value:
     {a_k}_{k=1}^K = top-K({a_n}_{n=1}^{N+1}, by Q)

  5. REFINE: For each elite, gradient ascent on Q w.r.t. action:
     for step = 1, ..., M:
       a_k ← clip(a_k + η · ∇_a Q(s, a_k), -1, 1)

  6. RE-EVALUATE: Compute Q for all refined elites:
     Q_k^refined = Q(s, a_k^refined)

  7. SELECT BEST: Pick the single best refined action:
     a* = argmax_k Q_k^refined

  8. CONVERT TO ACTOR SPACE:
     If predict_a_exec=False: target_δ = (a* - a_base) / (α + ε)
     If predict_a_exec=True:  target = a*
     Clip target to (-1+ε, 1-ε) for TanhNormal feasibility

  9. DISTILL: Train actor to imitate a* via sample-based MSE:
     δ_sample ~ π_θ(·|s)
     L_actor = E_s[ ||δ_sample - target||² ]
```

### Key Design Decisions

- **Including the base action as a candidate** ensures the algorithm can never degrade below the base policy's performance. If the base action has the highest Q-value, it will be selected as the distillation target, and the actor will learn to reproduce the base (i.e., zero residual).
- **Gradient ascent on Q w.r.t. action** (step 5) is the key mechanism for finding improved actions. The critic's Q-landscape is used as a differentiable objective for action optimization. Parameters: `M = 5` steps, `η = 0.01` step size.
- **Sample-based MSE** (not mode-based) provides gradient to both mean and std via the reparameterization trick. If `dist.mode()` were used, the loss would be independent of std, causing std to collapse.
- **No entropy term in the loss**: Entropy is monitored but not optimized. The actor's std evolves only through the MSE loss gradient.

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `parl_num_samples` (N) | 16 | Candidates sampled from actor |
| `parl_num_elites` (K) | 4 | Top candidates kept for refinement |
| `parl_num_grad_steps` (M) | 5 | Gradient ascent iterations |
| `parl_step_size` (η) | 0.01 | Gradient ascent learning rate |

---

## 9. Algorithm 5: GradQ

### Overview

A simplified version of PARL that skips the Best-of-N selection. Instead of sampling N candidates and selecting elites, it samples a single action from the actor, refines it via gradient ascent on Q, and distills the result.

### Procedure

```
For each state s in the batch:
  1. Sample 1 action from current actor (stop-gradient):
     δ ~ π_θ(·|s)
     a = clip(a_base + α · δ, -1, 1)

  2. Gradient ascent on Q w.r.t. action:
     for step = 1, ..., M:
       a ← clip(a + η · ∇_a Q(s, a), -1, 1)

  3. Convert refined a back to actor output space:
     target_δ = (a - a_base) / (α + ε)
     Clip target to (-1+ε, 1-ε)

  4. Distill into actor via sample-based MSE:
     δ_sample ~ π_θ(·|s)
     L_actor = E_s[ ||δ_sample - target||² ]
```

### Comparison to PARL

| Aspect | PARL | GradQ |
|--------|------|-------|
| Candidates | N (default 16) + base | 1 |
| Elite selection | Top-K | None |
| Base policy safeguard | Yes (base included as candidate) | No |
| Compute cost | Higher (N forward passes + K refinements) | Lower (1 forward pass + 1 refinement) |
| Robustness | Higher (best of many) | Lower (single point) |

---

## 10. Temperature Auto-Tuning

### SAC Temperature

Used only by Algorithm 1 (Residual SAC). The entropy coefficient `α_temp` is learned:

```
α_temp = clip(exp(log_α), α_min, α_max)

L_temp = α_temp · (H_current - H_target)
```

where:
- `H_current = -E[log π(δ|s)]` — current policy entropy (TanhNormal, includes Jacobian)
- `H_target = -dim(δ)/2` — target entropy (default: negative half the action dimension)
- `α_min = 0.01`, `α_max = 2.0` — clipping range

**Dynamics**:
- If `H_current > H_target`: loss is positive → gradient decreases `log_α` → `α_temp` decreases → less entropy bonus → policy becomes more deterministic
- If `H_current < H_target`: loss is negative → gradient increases `log_α` → `α_temp` increases → more entropy bonus → policy explores more

**Temperature clipping** prevents:
- `α_temp → 0`: Would kill exploration entirely, causing premature convergence
- `α_temp → ∞`: Would dominate the Q-value signal, preventing any useful learning

---

## 11. BC Warmup

### Motivation

Before RL training begins, the residual policy is randomly initialized. If RL updates start immediately, the random residuals will corrupt the base policy's actions, causing poor data collection and potentially unrecoverable early failures.

BC warmup solves this by pre-training the actor to **reproduce the base policy's behavior** before RL kicks in. This ensures:
1. The actor starts near the identity mapping (zero residual)
2. The critic receives reasonable data to learn from
3. The transition from BC to RL is smooth

### BC Target

```
If predict_a_exec=False (delta mode):
  target = 0                      # learn zero residual → a_exec = a_base

If predict_a_exec=True (a_exec mode):
  target = clip(a_base, -1, 1)    # learn to reproduce base action
```

The clipping for `predict_a_exec=True` is important because Pi-0.5's quantile unnormalization can produce actions outside `[-1, 1]`, but the actor's tanh output is bounded to `(-1, 1)`. Without clipping, the MSE target would be unreachable, pushing the actor mean toward `±∞`.

### BC Loss

```
δ_sample ~ π_θ(·|s)              # reparameterized sample
L_bc = E_s[ (1/D) Σ_d (δ_sample_d - target_d)² ]
```

Uses **sample-based MSE** (not mode-based). The reparameterization trick (`δ_sample = tanh(μ + σ·ε), ε ~ N(0,I)`) provides gradient to both `μ` and `σ`:
- Gradient to `μ`: pushes mean toward target
- Gradient to `σ`: pushes std to match the spread of the target (for a point target like 0, std should decrease)

If `dist.mode()` were used instead, the loss `||tanh(μ) - target||²` would be independent of `σ`, and `σ` would receive no gradient, potentially remaining at its initialization or drifting.

### During BC Warmup

- **Critic**: Still performs TD learning (with aggressive update ratio, default 10 critic updates per step)
- **Actor**: BC loss only (no RL signal)
- **Data collection**: Forces zero residual (pure base policy rollouts)
- **Temperature**: Not updated

---

## 12. BC Regularization

### During RL Training (Optional)

After BC warmup ends, an optional BC regularization term can be added to the actor loss to prevent the residual from drifting too far from the stored behavior:

```
L_actor = L_rl + λ_bc · L_bc_reg
```

where:

```
δ_sample ~ π_θ(·|s)
L_bc_reg = E_s[ (1/D) Σ_d (δ_sample_d - δ_stored_d)² ]
```

### Success-Only BC

When `bc_on_success_only=True`, the BC loss is computed only on transitions from successful episodes:

```
L_bc_reg = Σ_b [success_b · MSE_b] / Σ_b success_b
```

This biases the actor toward imitating its own successful behavior rather than all behavior (including failures).

### Success Replay Buffer

A separate buffer stores only transitions from successful trajectories. During actor updates, a fraction `success_buffer_ratio` (e.g., 0.2) of the actor batch is drawn from this buffer, with the rest from the main buffer. This provides a natural curriculum: the actor sees more success examples as they accumulate.

---

## 13. Critic Update (Shared Across All Algorithms)

All five algorithms share the same critic update (TD learning):

### Target Q Computation

```
δ', log π(δ'|s') = π_θ.sample_and_log_prob(s')
a_exec' = clip(a_base' + α · δ', -1, 1)

Q_target_i = Q_target_i(s', a_exec')    for i = 1, ..., N
Q̄' = reduce(Q_target_1, ..., Q_target_N)

y = r + γ^T' · mask · Q̄'

[Optional entropy backup]:
y -= γ^T' · mask · α_temp · log π(δ'|s')
```

where:
- `γ^T'` is the discount factor raised to the query frequency (precomputed at data insertion time). This accounts for the fact that one SAC "step" spans `T'` environment steps.
- `mask = 0` at terminal states, `mask = 1` otherwise
- `Q_target` is the slowly-updated target network (Polyak-averaged)
- Entropy backup (including `log π` in the target) is off by default

### Critic Loss

```
Q_i(s, a_exec)    for i = 1, ..., N    (current critic)

MSE:   L_critic = (1/N) Σ_i E_s[ (Q_i(s, a_exec) - y)² ]
Huber: L_critic = (1/N) Σ_i E_s[ huber(Q_i(s, a_exec) - y, δ_h) ]
```

where the Huber loss is:
```
huber(x, δ_h) = { 0.5·x²           if |x| ≤ δ_h
                 { δ_h·(|x| - 0.5·δ_h)  otherwise
```

Huber loss (optional, `use_huber_loss=True`) reduces sensitivity to large TD errors, which can occur in sparse reward settings where the critic's estimates are initially far from the true values.

### Target Network Update

```
θ_target ← τ · θ_critic + (1 - τ) · θ_target
```

With `τ = 0.005`, the target network slowly tracks the critic. This stabilizes training by providing a slowly-moving target for the Bellman backup.

---

## 14. Data Collection (Rollout Strategy)

### Per-Query-Step Procedure

At each decision point (every `T'` environment steps):

```
1. Observe environment → o
2. Query frozen base policy: a_base = π_base(o)  [uses internal random noise]
3. Build augmented observation: s = (o, a_base)
4. If BC warmup or first trajectory:
     δ = 0                          # zero residual
   Else:
     δ = mode(π_θ(s))              # deterministic (no sampling)
5. a_exec = clip(a_base[:T'] + α · δ, -1, 1)
6. Execute a_exec[0], a_exec[1], ..., a_exec[T'-1] in environment
7. Record (s, δ, a_base, reward, ...) for replay buffer
```

### Exploration Strategy

A critical design decision: **the residual policy uses deterministic (mode) actions during rollouts**. Exploration comes entirely from the **stochastic base policy** — Pi-0.5 uses internal diffusion noise that produces different base actions each query, even for the same observation.

This is the opposite of standard SAC where the actor samples stochastically during rollouts. The rationale:
- The base policy's stochasticity provides sufficient exploration in action space
- Adding sampling noise from the residual policy would add jitter on top of the base policy's variation, potentially degrading performance
- The residual's role is to systematically correct the base policy, not to explore

### NaN Guard

If the actor produces NaN/Inf actions during rollout:
```
if not all_finite(δ):
    δ = zeros                       # fall back to zero residual
```

---

## 15. Reward Shaping

### Sparse Reward (Default)

```
Successful episode:
  r_t = -1    for t = 0, ..., T-2      (every non-terminal query step)
  r_T = 0     for t = T-1               (terminal step)
  mask_T = 0                             (terminal)

Failed episode:
  r_t = -1    for all t                  (every query step)
  mask_t = 1  for all t                  (no terminal — infinite penalty stream)
```

This creates a strong incentive to succeed quickly:
- Success: total return = `-(T-1)` (shorter episodes are better)
- Failure: total return = `-∞` (due to `mask=1`, the discounted sum never terminates)

### Dense Reward

Uses raw environment rewards directly. Terminal mask is set at episode end.

### Query-Step Granularity

Rewards are assigned per **query step** (every `T'` environment steps), not per environment step. One SAC "transition" spans `T'` environment steps, and the discount factor is `γ^{T'}` to account for this.

---

## 16. Numerical Stability Techniques

### Gradient Clipping

Both actor and critic use `clip_by_global_norm` before Adam:

```
optimizer = chain(
    clip_by_global_norm(max_norm),    # default max_norm = 1.0
    adam(lr)
)
```

This prevents gradient explosions that can occur when:
- The critic's TD error is very large (early training with sparse rewards)
- The actor's log-prob produces extreme values near action boundaries
- The composition `base + α·δ` amplifies gradients

### NaN Guard on Gradients

```
grads = tree_map(λ x: nan_to_num(x, nan=0, posinf=0, neginf=0), grads)
```

Applied to all algorithm variants (GRPO, PPO, PARL, GradQ, BC warmup, critic). Replaces NaN/Inf gradient values with zeros, preventing a single bad sample from corrupting the entire parameter update.

The Residual SAC actor update does **not** mask NaN gradients — it logs them and lets the program crash. This is by design: SAC is the primary algorithm and NaN gradients indicate a fundamental issue that should be diagnosed, not silently suppressed.

### Temperature Clipping

```
α_temp = clip(exp(log_α), 0.01, 2.0)
```

Prevents the entropy coefficient from:
- Collapsing to 0 (killing exploration, causing premature convergence to suboptimal residuals)
- Exploding (dominating the Q-value signal, preventing learning)

### Log-Probability Clipping

```
log π(δ|s) = clip(log π(δ|s), -50, 50)
```

Prevents extreme log-prob values (which can occur when the policy is very certain about an action) from destabilizing the loss.

### Safe Action Clamping for Log-Prob

```
safe_δ = clip(δ, -1+ε, 1-ε)       # ε = 1e-6
log π(safe_δ|s)                     # avoids atanh(±1) = ±∞
```

When computing `log π` of stored/clipped actions (not fresh samples), the action may sit exactly at ±1.0 due to hard clipping during collection. The TanhNormal log-prob requires `atanh(δ)` which diverges at ±1. Clamping to `(-1+ε, 1-ε)` avoids this.

### Soft-Clip for Actor Gradient Flow

As described in Section 2, `soft_clip(x) = tanh(x)` (rescaled) replaces hard `clip` in the actor update to preserve non-zero gradients at action boundaries.

---

## 17. Diagnostics: Collapse and Dominance Detection

A core challenge in residual RL is balancing the residual's contribution against the base policy. Two failure modes:

### Residual Collapse

The residual shrinks to near-zero, and the agent degenerates to the base policy. Indicators:
- `collapse/delta_norm_mean → 0`: Residual magnitude vanishes
- `collapse/eff_delta_norm_mean → 0`: Effective perturbation `|α·δ|` vanishes
- `collapse/ratio_eff_delta_to_base_mean → 0`: Residual is negligible relative to base
- `collapse/near_zero_frac → 1`: Most delta dimensions are near zero

### Residual Dominance

The residual overwhelms the base policy, effectively ignoring it. Indicators:
- `collapse/ratio_eff_delta_to_base_mean → ∞`: Residual dominates base
- `collapse/clip_frac → 1`: Nearly all actions are at the boundaries ±1
- `collapse/cos_base_delta_mean → -1`: Residual is fighting the base policy

### Diagnostic Metrics

| Metric | Formula | Healthy Range |
|--------|---------|---------------|
| `collapse/delta_norm_mean` | `E[||δ||]` | `> 0.01` |
| `collapse/base_norm_mean` | `E[||a_base||]` | Stable |
| `collapse/eff_delta_norm_mean` | `E[|α| · ||δ||]` | `0.01–1.0` |
| `collapse/ratio_eff_delta_to_base_mean` | `E[|α·δ| / |a_base|]` | `0.05–0.5` |
| `collapse/ratio_exec_change_to_base_mean` | `E[||a_exec - a_base|| / ||a_base||]` | `0.01–0.3` |
| `collapse/near_zero_frac` | `frac(|δ_d| < 0.001)` | `< 0.5` |
| `collapse/clip_frac` | `frac(|a_exec| ≥ 0.999)` | `< 0.3` |
| `collapse/cos_base_delta_mean` | `cos(a_base, δ)` | Near 0 (orthogonal) |
| `collapse/cos_base_exec_change_mean` | `cos(a_base, a_exec - a_base)` | Near 0 |
| `collapse/alpha_temp` | `α_temp` | `0.01–2.0` |

---

## 18. Training Loop Structure

### Outer Loop (Trajectory-Wise Alternation)

```
for each training iteration:
  1. Collect one full trajectory using current policy + base policy
  2. Insert transitions into replay buffer (+ success buffer if successful)
  3. Compute number of gradient steps: N = traj_len × UTD_ratio
  4. For each gradient step (inner loop):
       [Phase detection: BC warmup vs RL]

       Critic updates (C times, default C=2):
         batch = sample(replay_buffer)
         update_critic(batch)

       Actor updates (A times, default A=4):
         If BC warmup:
           batch = sample(replay_buffer)
           update_actor_bc(batch)
         Elif on-policy PPO:
           batch = sample_from_last_traj(replay_buffer)
           update_actor_onpolicy(batch)
         Else (off-policy):
           batch = sample_mixed(replay_buffer, success_buffer)
           update_actor(batch)
```

### Update Ratios

The **Update-To-Data (UTD) ratio** is `multi_grad_step` (outer) × `num_critic_updates` (inner critic) × `num_actor_updates` (inner actor).

For example with defaults:
- `multi_grad_step = 1`: 1 outer gradient step per collected transition
- `num_critic_updates = 2`: 2 critic updates per outer step
- `num_actor_updates = 4`: 4 actor updates per outer step

This means the actor trains 4× more frequently than the data rate, and the critic trains 2× more than the data rate. The asymmetry (more actor updates) is intentional: the actor needs many gradient steps to make progress in the high-dimensional action space, while the critic converges faster on the value estimation.

### BC Warmup Phase

During BC warmup (`i < bc_warmup_steps`):
- Update ratios change: `bc_warmup_num_critic_updates=10`, `bc_warmup_num_actor_updates=1`
- This aggressively trains the critic (which benefits from the base policy's data) while gently training the actor (just enough to converge on zero residual)

---

## 19. Hyperparameter Summary

### Action Composition
| Parameter | Default | Description |
|-----------|---------|-------------|
| `residual_alpha` | 0.1 | Scaling factor for `a_exec = base + α·δ` |
| `predict_a_exec` | False | If True, actor outputs `a_exec` directly |
| `query_freq` | varies | Steps between re-querying SAC |
| `chunk_len` | 10 | Base policy action horizon |

### Networks
| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dims` | (256, 256, 256) | MLP hidden layer sizes |
| `latent_dim` | 200 | Encoder bottleneck dimension |
| `num_qs` | 10 | Q-function ensemble size |
| `critic_reduction` | 'mean' | Ensemble aggregation |
| `log_std_min` / `log_std_max` | -5.0 / 2.0 | Policy log-std bounds |
| `learn_std` | True | Learned vs fixed std |
| `fixed_log_std` | -0.5 | Fixed std value (when `learn_std=False`) |

### Optimization
| Parameter | Default | Description |
|-----------|---------|-------------|
| `actor_lr` | 1e-4 | Actor learning rate |
| `critic_lr` | 3e-4 | Critic learning rate |
| `temp_lr` | 3e-4 | Temperature learning rate |
| `discount` | 0.999 | Discount factor |
| `tau` | 0.005 | Target network Polyak rate |
| `max_grad_norm` | 1.0 | Gradient clipping norm |
| `batch_size` | 16 | Mini-batch size |
| `multi_grad_step` | 1 | Outer UTD ratio |
| `num_critic_updates` | 2 | Critic updates per outer step |
| `num_actor_updates` | 4 | Actor updates per outer step |

### Temperature
| Parameter | Default | Description |
|-----------|---------|-------------|
| `init_temperature` | 1.0 | Initial entropy coefficient |
| `target_entropy` | -dim/2 | Target entropy for auto-tuning |
| `clip_temp` | True | Whether to clip temperature |
| `clip_min_temp` / `clip_max_temp` | 0.01 / 2.0 | Clipping range |

### GRPO / Q-Weighted PG
| Parameter | Default | Description |
|-----------|---------|-------------|
| `grpo_num_samples` | 8 | G: number of action samples per state |
| `entropy_coeff` | 1e-3 | Entropy bonus coefficient (β) |
| `use_grpo_baseline` | True | Subtract group mean from Q |
| `normalize_advantages` | False | Normalize advantages |
| `adv_clip_min` / `adv_clip_max` | None | Optional advantage clipping |
| `log_prob_clip` | 50.0 | Log-prob safety clipping |

### On-Policy PPO
| Parameter | Default | Description |
|-----------|---------|-------------|
| `clip_epsilon` | 0.2 | PPO clip epsilon (ε) |
| `clip_min_epsilon_multiplier` | 1.0 | Asymmetric lower clip multiplier |
| `clip_max_epsilon_multiplier` | 1.0 | Asymmetric upper clip multiplier |
| `log_ratio_clip` | 20.0 | Log importance ratio clip |

### PARL
| Parameter | Default | Description |
|-----------|---------|-------------|
| `parl_num_samples` | 16 | N: candidates from actor |
| `parl_num_elites` | 4 | K: elites for refinement |
| `parl_num_grad_steps` | 5 | M: gradient ascent iterations |
| `parl_step_size` | 0.01 | η: gradient ascent learning rate |

### GradQ
| Parameter | Default | Description |
|-----------|---------|-------------|
| `gradq_num_grad_steps` | 5 | Gradient ascent iterations |
| `gradq_step_size` | 0.01 | Gradient ascent learning rate |

### BC Warmup
| Parameter | Default | Description |
|-----------|---------|-------------|
| `bc_warmup_steps` | 0 | Gradient steps of BC warmup (0 = disabled) |
| `bc_warmup_num_critic_updates` | 10 | Aggressive critic training during warmup |
| `bc_warmup_num_actor_updates` | 1 | Light actor BC during warmup |

### BC Regularization
| Parameter | Default | Description |
|-----------|---------|-------------|
| `bc_reg_coeff` | 0.0 | BC regularization weight (λ_bc, 0 = disabled) |
| `bc_on_success_only` | False | Only regularize on success transitions |
| `success_buffer_ratio` | 0.0 | Fraction of actor batch from success buffer |
| `success_buffer_min_size` | 100 | Min samples before using success buffer |

### Stability
| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_huber_loss` | False | Huber loss for critic |
| `huber_delta` | 1.0 | Huber loss threshold |
| `backup_entropy` | False | Include entropy in critic target |

### Reward
| Parameter | Default | Description |
|-----------|---------|-------------|
| `reward_type` | 'sparse' | `sparse` (-1/0) or `dense` (raw env) |
