# Residual RL SAC Training Pipeline — Detailed Code Explanation

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Key Insight: SAC Predicts Residual Actions on Top of a Frozen Base Policy](#2-key-insight-sac-predicts-residual-actions-on-top-of-a-frozen-base-policy)
3. [Entry Point and Configuration](#3-entry-point-and-configuration)
4. [Environment Setup](#4-environment-setup)
5. [Agent Initialization (PixelSACResidualLearner)](#5-agent-initialization-pixelsacresiduallearner)
6. [Network Architectures](#6-network-architectures)
   - [Encoder](#61-encoder)
   - [Actor (Policy Network)](#62-actor-policy-network)
   - [Critic (Q-Network Ensemble)](#63-critic-q-network-ensemble)
   - [Temperature (Entropy Coefficient)](#64-temperature-entropy-coefficient)
7. [Replay Buffer and Success Buffer](#7-replay-buffer-and-success-buffer)
8. [Training Loop](#8-training-loop)
   - [BC Warmup Phase](#81-bc-warmup-phase)
   - [Trajectory Collection](#82-trajectory-collection)
   - [Reward Shaping](#83-reward-shaping)
   - [Gradient Updates (Separated Critic/Actor)](#84-gradient-updates-separated-criticactor)
9. [Algorithm Variants](#9-algorithm-variants)
   - [Residual SAC (residual_sac)](#91-residual-sac-residual_sac)
   - [GRPO / Q-Weighted PG (residual_grpo / q_weighted_pg)](#92-grpo--q-weighted-pg-residual_grpo--q_weighted_pg)
   - [On-Policy PPO (on_policy_ppo)](#93-on-policy-ppo-on_policy_ppo)
   - [PARL — Policy-Agnostic RL (residual_parl)](#94-parl--policy-agnostic-rl-residual_parl)
   - [GradQ (residual_gradq)](#95-gradq-residual_gradq)
10. [Critic Update](#10-critic-update)
11. [Actor Update — Residual SAC](#11-actor-update--residual-sac)
12. [BC Warmup Update](#12-bc-warmup-update)
13. [Stability Mechanisms](#13-stability-mechanisms)
14. [Evaluation](#14-evaluation)
15. [Observation Processing Pipeline](#15-observation-processing-pipeline)
16. [File Reference Map](#16-file-reference-map)

---

## 1. High-Level Overview

This codebase implements **Residual RL** with multiple actor optimization algorithms. The core idea:

1. A pre-trained **frozen base policy** (Pi-0 or Pi-0.5, a vision-language-action diffusion model) generates base action chunks in environment action space.
2. A lightweight **residual policy** (SAC-based) predicts a small correction (delta) on top of the base actions.
3. The executed action is composed as: **`a_exec = clip(base_action + alpha * delta, -1, 1)`**
4. The critic learns `Q(s, a_exec)` — value of the *composed* action.
5. The actor is optimized to maximize Q via one of several algorithms: SAC, GRPO, PPO, PARL, or GradQ.

Unlike DSRL (which predicts noise in diffusion model latent space), Residual RL operates entirely in **environment action space**. The base policy is queried as a black box; its internal noise process provides stochastic exploration.

**Framework**: JAX + Flax (neural networks) + Optax (optimizers) + Distrax (probability distributions)

---

## 2. Key Insight: SAC Predicts Residual Actions on Top of a Frozen Base Policy

The action space of the SAC agent is the **environment action space**, not a latent noise space.

- **SAC action space**: `(query_freq, action_dim)` — e.g., `(8, 7)` for an 8-step chunk of 7-DOF actions
- **Base policy output**: `(chunk_len, action_dim)` — e.g., `(10, 7)` for Pi-0.5's full action horizon
- **Composition**: `a_exec = clip(base_action[:query_freq] + residual_alpha * delta, -1, 1)`

```
observation → Pi-0.5 produces base_actions (internally uses its own random noise)
observation + base_actions → SAC predicts delta (residual)
a_exec = clip(base + alpha * delta, -1, 1) → executed in environment
```

The `residual_alpha` parameter (default 0.1) controls how much the residual can perturb the base action. Small alpha keeps the agent close to the base policy.

### predict_a_exec Mode

An alternative mode where the actor predicts `a_exec` directly instead of delta:
- **predict_a_exec=False** (default): Actor outputs delta, composed as above
- **predict_a_exec=True**: Actor outputs the full executed action; `residual_alpha` is ignored for composition (only used for logging diagnostics)

### Observation Space

The SAC agent's observation includes the base policy's actions:

```
observations = {
    'pixels': (1, H, W, 3*num_cameras, 1),     # images (or ignored if use_vlm_embedding)
    'state': (1, state_dim, 1),                  # proprioceptive state [optional]
    'vlm_embedding': (1, vlm_embedding_dim, 1),  # VLM hidden states [optional]
    'base_action': (1, chunk_len, action_dim, 1), # base policy actions [always present]
}
```

The trailing `1` dimensions are frame-stacking placeholders.

---

## 3. Entry Point and Configuration

### Entry Script: `examples/launch_train_sim_residual.py`

Parses CLI arguments and assembles training configuration. Key hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `algo` | `residual_sac` | Algorithm: `sac`, `residual_sac`, `q_weighted_pg`, `residual_grpo`, `residual_parl`, `residual_gradq` |
| `residual_alpha` | `0.1` | Scaling factor for residual actions |
| `predict_a_exec` | `False` | Actor predicts `a_exec` directly instead of delta |
| `actor_lr` | `1e-4` | Actor learning rate |
| `critic_lr` | `3e-4` | Critic learning rate |
| `temp_lr` | `3e-4` | Temperature learning rate |
| `hidden_dims` | `(256, 256, 256)` | MLP hidden layer sizes for actor/critic |
| `latent_dim` | `200` | Bottleneck dimension after encoder |
| `discount` | `0.999` | Discount factor (high due to sparse rewards) |
| `tau` | `0.005` | Soft target update rate |
| `critic_reduction` | `'mean'` | How to aggregate Q-values (`'mean'` or `'min'`) |
| `num_qs` | `10` | Number of Q-functions in ensemble |
| `num_critic_updates` | `2` | Critic updates per gradient step |
| `num_actor_updates` | `4` | Actor updates per gradient step |
| `max_grad_norm` | `1.0` | Gradient clipping norm |
| `query_freq` | varies | How often SAC re-predicts residual during rollout |
| `chunk_len` | `10` | Base policy action horizon |
| `bc_warmup_steps` | `0` | Number of BC warmup gradient steps before RL |
| `bc_reg_coeff` | `0.0` | BC regularization coefficient during RL |
| `success_buffer_ratio` | `0.0` | Fraction of actor batch from success buffer |
| `use_vlm_embedding` | `False` | Use VLM embeddings instead of raw pixels |
| `learn_std` | `True` | Whether policy std is learned (vs fixed) |
| `log_std_min` / `log_std_max` | `-5.0` / `2.0` | Bounds on policy log-std |
| `clip_temp` | `True` | Whether to clip temperature |
| `clip_min_temp` / `clip_max_temp` | `0.01` / `2.0` | Temperature clipping range |
| `use_huber_loss` | `False` | Use Huber loss for critic instead of MSE |
| `reward_type` | `sparse` | `sparse` (-1/0) or `dense` (raw env reward) |

### Main Function: `examples/train_sim_residual.py::main_residual(variant)`

1. Sets up JAX sharding across available GPUs
2. Disables TensorFlow GPU usage
3. Creates the environment (LIBERO, Aloha, or CartPole)
4. Creates a WandB logger
5. Creates `DummyEnvResidual` to define observation/action spaces (includes `base_action`)
6. Loads the frozen Pi-0/Pi-0.5 base policy (`agent_dp`)
7. Monkey-patches openpi `Policy.infer` to fix tokenized_prompt dimension issue
8. Creates the residual SAC agent based on `algo` selection:
   - `residual_sac` → `PixelSACResidualLearner`
   - `q_weighted_pg` / `residual_grpo` → `PixelPPOResidualLearner`
   - `residual_parl` → `PixelPARLResidualLearner`
   - `residual_gradq` → `PixelGradQResidualLearner`
   - `sac` → `PixelSACLearner` (no residual, baseline)
9. Creates replay buffer (and optional success replay buffer)
10. Launches the training loop via `trajwise_alternating_training_loop_residual()`

---

## 4. Environment Setup

### Supported Environments

1. **LIBERO** (primary): Robotic manipulation benchmark
   - Observation: `agentview_image` (224x224), `robot0_eye_in_hand_image`, robot state (8-dim)
   - Task max reward: 1 (binary success)
   - Max timesteps: 500

2. **Aloha Cube Transfer**: Bimanual robot sim
   - Observation: top camera pixel, agent_pos (14-dim)
   - Task max reward: 4
   - Max timesteps: 500

3. **CartPole**: Classic control (used with `ZeroBasePolicy` as base)
   - Observation: image + state

### DummyEnvResidual (Space Definition)

`DummyEnvResidual` (in `examples/train_sim_residual.py`) defines the observation and action spaces:

```
Observation space:
  - 'pixels': Box(0, 255, shape=(resize_image, resize_image, 3*num_cameras, 1), uint8)
  - 'state': Box(-1, 1, shape=(state_dim, 1), float32)  [optional]
  - 'vlm_embedding': Box(-inf, inf, shape=(vlm_embedding_dim, 1), float32)  [optional]
  - 'base_action': Box(-inf, inf, shape=(chunk_len, action_dim, 1), float32)  [always present]

Action space:
  - Box(-1, 1, shape=(query_freq, action_dim), float32)  — residual delta (or a_exec if predict_a_exec)
```

**Key difference from DSRL**: The action space is `(query_freq, action_dim)` (environment action dimensions), not `(1, 32)` (diffusion noise dimensions).

---

## 5. Agent Initialization (PixelSACResidualLearner)

**File**: `jaxrl2/agents/pixel_sac/pixel_sac_residual_learner.py`
**Class**: `PixelSACResidualLearner` (inherits from `Agent`)

### Constructor Flow

```
PixelSACResidualLearner.__init__(seed, observations, actions, **kwargs):
  1. Record action_dim = prod(actions.shape[-2:])
  2. Record action_chunk_shape = actions.shape[-2:]
  3. Store residual_alpha, predict_a_exec, use_huber_loss, num_critic_updates, num_actor_updates, etc.
  4. Select encoder based on encoder_type
  5. Build actor: PixelMultiplexer(encoder, policy_def, pop_base_actions=False)
     - policy_def = LearnedStdTanhNormalPolicy (if learn_std=True) or FixedStdTanhNormalPolicy
  6. Build critic: PixelMultiplexer(encoder, StateActionEnsemble, pop_base_actions=configurable)
  7. Build temperature: Temperature(init_temp, clip_temp, min_temp, max_temp)
  8. Initialize all as Flax TrainState with optax.chain(clip_by_global_norm, adam)
  9. Deep-copy critic params for target network
  10. Set target_entropy = -action_dim/2 if 'auto'
```

Key differences from non-residual SAC:
- Actor sees `base_action` in observations (`pop_base_actions=False`)
- Critic optionally removes `base_action` from observations (`pop_base_actions` configurable, default `True`)
- Both actor and critic use `optax.chain(clip_by_global_norm(max_grad_norm), adam)` (gradient clipping)
- Temperature supports clipping: `clip(exp(log_alpha), min_temp, max_temp)`
- Separate `update_critic()`, `update_actor()`, `update_actor_bc()` methods (not a fused `update()`)

---

## 6. Network Architectures

### 6.1 Encoder

Multiple encoder options are available. The default for residual is `'small'` CNN.

#### Small CNN Encoder (`encoder_type='small'`)
**File**: `jaxrl2/networks/encoders/networks.py`
```
Input: (B, H, W, C, 1) -> normalize to [0,1] -> reshape to (B, H, W, C)
-> Conv(64, 3x3, stride=2) -> ReLU
-> Conv(64, 3x3, stride=1) -> ReLU
-> Conv(64, 3x3, stride=1) -> ReLU
-> Conv(64, 3x3, stride=1) -> ReLU
-> Flatten to (B, D)
```

CNN features default to `(64, 64, 64, 64)` for residual (vs `(32, 32, 32, 32)` for DSRL).

#### ResNet Encoders (`resnet_18_v1`, `resnet_34_v1`, etc.)
**File**: `dsrl_pi0/jaxrl2/networks/encoders/resnet_encoderv1.py`

Standard ResNet with:
- 7x7 conv stem with stride 2
- Group/batch/layer normalization
- Spatial softmax output — produces 2D keypoint coordinates per feature map channel

**Spatial Softmax** (`dsrl_pi0/jaxrl2/networks/encoders/spatial_softmax.py`):
- Feature maps `(B, H, W, C)` → softmax over spatial dims → expected `(x, y)` per channel → output `(B, 2*C)`
- Optional learnable temperature

### PixelMultiplexer (Encoder Wrapper)

**File**: `jaxrl2/networks/encoders/networks.py`

Updated for residual RL with two key flags:

```
PixelMultiplexer(encoder, network, latent_dim, use_bottleneck,
                 pop_base_actions, use_vlm_embedding)

Input: observations dict {'pixels': ..., 'state': ..., 'base_action': ..., 'vlm_embedding': ...}
  1. If use_vlm_embedding: skip encoder, use vlm_embedding directly as visual representation
  2. Otherwise: pass pixels through encoder -> Dense(latent_dim) -> LayerNorm -> tanh
  3. If pop_base_actions=True: remove 'base_action' from observation dict (critic doesn't see it)
  4. If pop_base_actions=False: flatten base_action (B, chunk_len, action_dim, 1) -> (B, chunk_len*action_dim) and keep in dict
  5. Pass to downstream network (actor/critic MLP)
```

- **Actor** always has `pop_base_actions=False` (sees base_action to condition residual)
- **Critic** default has `pop_base_actions=True` (evaluates Q(s, a_exec) without seeing base_action explicitly)

### 6.2 Actor (Policy Network)

**File**: `jaxrl2/networks/learned_std_normal_policy.py`

Two policy types available:

#### LearnedStdTanhNormalPolicy (default when `learn_std=True`)
```
Input: observation dict -> _flatten_dict -> concatenated vector
-> MLP(hidden_dims, activate_final=True)  # e.g., (256, 256, 256) with ReLU
-> Dense(action_dim) -> means
-> Dense(action_dim) -> log_stds (clipped to [log_std_min, log_std_max])
-> TanhMultivariateNormalDiag(means, exp(log_stds))
```

#### FixedStdTanhNormalPolicy (when `learn_std=False`)
```
Same MLP architecture but:
-> Dense(action_dim) -> means
-> log_std = fixed_log_std (not learned, e.g., -0.5)
-> TanhMultivariateNormalDiag(means, exp(fixed_log_std))
```

The fixed-std variant prevents std collapse after BC warmup (the learned std can collapse to near-zero during BC and fail to recover during RL).

**TanhMultivariateNormalDiag**: `distrax.Transformed` distribution with Tanh bijector.
- `sample_and_log_prob()` includes correct log-det-Jacobian for tanh squashing
- `mode()` returns `tanh(means)` (deterministic action for evaluation)

### 6.3 Critic (Q-Network Ensemble)

**File**: `dsrl_pi0/jaxrl2/networks/values/state_action_ensemble.py`
**Class**: `StateActionEnsemble`

Uses Flax's `nn.vmap` to create `num_qs` (default 10) independent Q-functions:

```python
VmapCritic = nn.vmap(StateActionValue,
                     variable_axes={'params': 0},
                     split_rngs={'params': True},
                     in_axes=None,
                     out_axes=0,
                     axis_size=num_qs)
```

Each Q-function takes `(observations, a_exec_flat)` and outputs a scalar Q-value.

**Critical**: The critic receives **executed actions** `a_exec` (not delta/residual actions). The composition `a_exec = clip(base + alpha * delta, -1, 1)` happens before passing to the critic.

Output shape: `(num_qs, batch_size)` — one Q-value per ensemble member per batch item.

### 6.4 Temperature (Entropy Coefficient)

**File**: `jaxrl2/agents/pixel_sac/temperature.py`

Updated with clipping:

```python
class Temperature(nn.Module):
    initial_temperature: float = 1.0
    clip_temp: bool = True
    min_temp: float = 0.01
    max_temp: float = 2.0

    def __call__(self):
        log_temp = self.param('log_temp', ...)
        temp = jnp.exp(log_temp)
        if self.clip_temp:
            temp = jnp.clip(temp, self.min_temp, self.max_temp)
        return temp
```

Clipping prevents temperature from collapsing to zero (killing exploration) or exploding (dominating the Q-value signal).

---

## 7. Replay Buffer and Success Buffer

### Main Replay Buffer

**File**: `dsrl_pi0/jaxrl2/data/replay_buffer.py`
**Class**: `ReplayBuffer`

```python
data = {
    'observations': {
        'pixels': np.array,
        'state': np.array,
        'base_action': np.array,     # (capacity, chunk_len, action_dim, 1) — from frozen policy
        'vlm_embedding': np.array,    # (capacity, vlm_dim, 1) [optional]
    },
    'next_observations': { ... same structure ... },
    'actions': np.array,        # (capacity, query_freq, action_dim) — delta or a_exec
    'next_actions': np.array,
    'rewards': np.array,        # (capacity,)
    'masks': np.array,          # (capacity,) — 1.0 if not terminal, 0.0 if terminal
    'discount': np.array,       # (capacity,) — precomputed discount^query_freq
    'success_flag': np.array,   # (capacity,) — 1.0 if episode was successful
    'old_log_probs': np.array,  # (capacity,) — log prob under behavior policy (for PPO)
}
```

### Success Replay Buffer

When `success_buffer_ratio > 0`, a separate `ReplayBuffer` stores only transitions from successful trajectories. During actor updates, a fraction `success_buffer_ratio` of the actor batch is drawn from this buffer.

**Sampling logic** (in `_sample_actor_batch()`):
```python
if success_buffer_ready:
    success_batch = success_buffer.sample(int(batch_size * success_buffer_ratio))
    main_batch = main_buffer.sample(batch_size - success_batch_size)
    batch = concat_recursive([main_batch, success_batch])
else:
    batch = main_buffer.sample(batch_size)
```

This biases the actor toward imitating successful behavior while still learning from failures.

---

## 8. Training Loop

**File**: `examples/train_utils_sim_residual.py`
**Function**: `trajwise_alternating_training_loop_residual()`

The training follows a **trajectory-wise alternating** pattern with **separate critic and actor updates**:

```
while gradient_steps <= max_steps:
    1. Collect one full trajectory using current policy + base policy
    2. Add trajectory to replay buffer (+ success buffer if successful)
    3. Perform N gradient updates:
       N = len(trajectory) * multi_grad_step (or num_online_gradsteps_batch)
       For each step:
         a. num_critic_updates critic updates (sample from replay buffer)
         b. num_actor_updates actor updates:
            - BC warmup phase: MSE distillation
            - On-policy PPO: sample from last trajectory
            - Off-policy (SAC/GRPO/PARL/GradQ): sample from buffer (+ success buffer)
    4. Log metrics, run evaluation, save checkpoints periodically
```

### 8.1 BC Warmup Phase

When `bc_warmup_steps > 0`, the first `bc_warmup_steps` gradient steps use **behavioral cloning** instead of RL for the actor:

- **Critic**: Still does TD learning (with aggressive `bc_warmup_num_critic_updates` per step, default 10)
- **Actor**: MSE loss to reproduce the base policy's behavior:
  - `predict_a_exec=False`: target = zeros (learn zero residual, so `a_exec = a_base`)
  - `predict_a_exec=True`: target = `clip(base_action, -1, 1)` (learn to predict base action directly)

This warm-starts the actor near the identity mapping before RL training begins, ensuring a smooth transition.

During BC warmup, trajectory collection forces **zero residual** (pure base policy rollouts), ensuring the critic learns reasonable Q-values before the actor starts perturbing.

### 8.2 Trajectory Collection

**Function**: `collect_traj_residual(variant, agent, env, i, agent_dp)`

Step-by-step for each timestep `t`:

1. Convert raw observation to image and robot state
2. **Every `query_freq` steps** (when `t % query_freq == 0`):
   a. Query frozen Pi-0.5: `agent_dp.infer(obs_pi_zero)` → `base_actions` (chunk_len, action_dim)
   b. Optionally extract VLM embedding: `infer_result["vlm_embedding"]` → mean-pool → `(W,)`
   c. Build `obs_dict` with pixels/vlm_embedding, state, and `base_action`
   d. **If first trajectory or BC warmup**: force zero residual (pure base policy)
   e. **Otherwise**: `agent.eval_actions(obs_dict)` — **deterministic mode** (not sampling)
      - Exploration comes from the *stochastic base policy* (Pi-0.5 uses internal random noise each query)
      - SAC residual uses deterministic mode to avoid adding jitter on top
   f. For on-policy PPO: also compute `log_prob` of the action for importance weighting
   g. Compose: `a_exec = clip(base[:query_freq] + alpha * delta, -1, 1)` (or just clip delta if predict_a_exec)
3. Execute `actions[t % query_freq]` in environment
4. Record reward; break if done
5. After rollout, compute rewards (sparse or dense) and masks

**Key difference from DSRL**: DSRL uses `agent.sample_actions()` (stochastic noise prediction). Residual RL uses `agent.eval_actions()` (deterministic delta) because the base policy already provides stochasticity.

### 8.3 Reward Shaping

Two reward modes:

**Sparse** (default, `reward_type='sparse'`):
```python
if is_success:
    rewards = [-1, -1, ..., -1, 0]   # last step gets 0
    masks   = [ 1,  1, ...,  1, 0]   # terminal at last step
else:
    rewards = [-1, -1, ..., -1]       # all -1
    masks   = [ 1,  1, ...,  1]       # no terminal
```

**Dense** (`reward_type='dense'`):
```python
rewards = raw_env_rewards              # e.g., for CartPole: +1 per step
masks   = [1, 1, ..., 1, 0]          # terminal only on done
```

Rewards are computed **per query step** (every `query_freq` environment steps), not per environment step.

### 8.4 Gradient Updates (Separated Critic/Actor)

Unlike DSRL's fused `_update_jit` (one critic + one actor update per call), residual SAC uses separate JIT-compiled functions with configurable update ratios:

```python
for _ in range(num_gradsteps):
    # Critic updates
    for _ in range(num_critic_updates):  # default 2
        batch = next(replay_buffer_iterator)
        critic_info = agent.update_critic(batch)

    # Actor updates
    for _ in range(num_actor_updates):  # default 4
        actor_batch = _sample_actor_batch(...)  # may mix in success buffer
        actor_info = agent.update_actor(actor_batch)
```

Three JIT-compiled entry points:
- `_update_critic_jit()` — critic TD update + target network update
- `_update_actor_jit()` — actor RL update + temperature update
- `_update_actor_bc_jit()` — actor BC warmup update (no critic, no temperature)

---

## 9. Algorithm Variants

All algorithms share the same critic (TD learning on `Q(s, a_exec)`), but differ in how the actor is trained.

### 9.1 Residual SAC (`residual_sac`)

**Learner**: `PixelSACResidualLearner`
**Actor update**: `update_actor_residual()`

Standard SAC objective adapted for residual actions:

```
1. Sample delta ~ pi(s)
2. Compose a_exec = soft_clip(base + alpha * delta, -1, 1)  [or a_exec = delta if predict_a_exec]
3. Evaluate Q(s, a_exec) using critic
4. actor_loss = mean(alpha_temp * log_prob - Q)
5. Optional BC regularization: actor_loss += bc_reg_coeff * MSE(pi_sample, stored_delta)
6. Temperature update: temp_loss = alpha_temp * (entropy - target_entropy)
```

Key detail: Uses `_soft_clip()` (tanh-based differentiable clip) instead of `jnp.clip` when composing `a_exec` during the actor update. This preserves gradients when `base + alpha * delta` hits the action bounds. Standard `jnp.clip` has zero gradient at boundaries, which can trap the actor.

### 9.2 GRPO / Q-Weighted PG (`residual_grpo` / `q_weighted_pg`)

**Learner**: `PixelPPOResidualLearner`
**Actor update**: `update_actor_residual_ppo()`

Advantage-weighted regression with group sampling:

```
1. Sample G delta candidates from current actor (stop-gradient)
2. For each candidate: compose a_exec, evaluate Q(s, a_exec) using target critic
3. Compute advantages:
   - GRPO (residual_grpo): A_g = Q_g - mean(Q)  (subtract group baseline)
   - Q-weighted PG (q_weighted_pg): A_g = Q_g  (raw Q, no baseline)
4. Stop-gradient advantages
5. Re-evaluate log_prob(delta_g | s) under current actor params
6. actor_loss = -mean(A_g * log_prob_g) - entropy_coeff * mean_entropy
7. Optional BC regularization, advantage clipping/normalization
```

This is NOT PPO (no importance ratios). Actions are sampled fresh from the current policy each call, so ratios are always ~1. The term "PPO" in the function name is historical.

### 9.3 On-Policy PPO (`on_policy_ppo`)

**Learner**: `PixelPPOResidualLearner` with `on_policy_ppo=True`
**Actor update**: `update_actor_residual_ppo_onpolicy()`

True PPO with stored `old_log_probs`:

```
1. Sample batch from the LAST collected trajectory (not full replay buffer)
2. Stored actions = delta actions taken by behavior policy
3. old_log_probs = log_prob at collection time (stored in buffer)
4. Compose a_exec from stored actions, evaluate Q as advantage
5. Compute current log_prob of stored actions under current policy
6. ratio = exp(current_log_prob - old_log_prob)
7. PPO clipped loss: L = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
8. Entropy bonus via fresh sample (with Jacobian correction)
```

Uses `online_replay_buffer.sample_from_last_traj(batch_size)` to get on-policy data.

### 9.4 PARL — Policy-Agnostic RL (`residual_parl`)

**Learner**: `PixelPARLResidualLearner`
**Actor update**: `update_actor_residual_parl()`

Decouples actor training from policy gradient entirely:

```
1. Sample N action candidates from current actor (frozen)
2. Add base policy action as (N+1)-th candidate (ensures no degradation below base)
3. Evaluate Q(s, a_exec) for all N+1 candidates
4. Keep top-K elites by Q-value
5. Refine elites via gradient ascent on Q w.r.t. action:
   a_exec <- clip(a_exec + step_size * grad_a Q(s, a_exec), -1, 1)
   (repeated for parl_num_grad_steps iterations)
6. Pick the best refined action as distillation target a*
7. Convert a* back to actor output space (delta or a_exec)
8. Train actor via MSE: loss = mean((pi_sample - a*)^2)
```

The actor **never differentiates through the critic**. The critic provides stop-gradient targets only. This avoids issues with tanh squashing and action clipping killing gradients.

Including the base policy action as a candidate ensures the residual can never perform worse than the base policy.

### 9.5 GradQ (`residual_gradq`)

**Learner**: `PixelGradQResidualLearner`
**Actor update**: `update_actor_residual_gradq()`

Simplified PARL — skips Best-of-N selection:

```
1. Sample 1 action from current actor (frozen)
2. Convert to a_exec space
3. Refine via gradient ascent: a <- a + step_size * grad_a Q(s, a)
4. Use refined action as distillation target
5. Train actor via MSE
```

Same benefits as PARL (no critic-through-actor gradients) but cheaper (no ensemble of candidates).

---

## 10. Critic Update

**File**: `jaxrl2/agents/pixel_sac/residual_critic_updater.py`
**Function**: `update_critic_residual()`

TD learning on composed actions:

```
1. Extract base_action and next_base_action from observations
2. Compose current a_exec:
   - predict_a_exec=False: a_exec = clip(base + alpha * stored_delta, -1, 1)
   - predict_a_exec=True: a_exec = clip(stored_a_exec, -1, 1)
3. Sample next_delta from actor: next_delta, next_log_probs = actor(next_obs)
4. Compose next_a_exec from next_base_action + alpha * next_delta
5. Target Q: next_qs = target_critic(next_obs, next_a_exec)
   - Reduce ensemble: min or mean
   - target_q = rewards + discount_from_buffer * masks * next_q
   - Optional entropy backup: target_q -= discount * masks * alpha_temp * next_log_probs
6. Critic loss:
   - MSE: (qs - target_q)^2
   - Or Huber loss: huber(qs - target_q, delta) if use_huber_loss=True
7. NaN guard on gradients, then apply
```

The `discount` field in the batch is `gamma^query_freq` (precomputed when inserting into buffer).

Logged metrics include Q-value statistics, TD error stats, and residual-specific diagnostics (delta_norm, base_action_norm, a_exec stats, clipping rate).

---

## 11. Actor Update — Residual SAC

**File**: `jaxrl2/agents/pixel_sac/residual_actor_updater.py`
**Function**: `update_actor_residual()`

```
1. Forward pass: dist = actor(observations)
2. Sample: delta, log_probs = dist.sample_and_log_prob()
3. Reshape delta to chunk: (B, query_freq, action_dim)
4. Compose a_exec:
   - predict_a_exec=False: a_exec = _soft_clip(base + alpha * delta, -1, 1)
   - predict_a_exec=True: a_exec = delta (already in tanh range)
5. Flatten a_exec and evaluate: qs = critic(obs, a_exec_flat)
6. Reduce ensemble: q = qs.min(0) or qs.mean(0)
7. Clip log_probs to [-50, 50] for numerical safety
8. RL loss = mean(-q + alpha_temp * log_probs)
9. Optional BC regularization:
   bc_loss = compute_bc_loss_residual(dist, batch, ...)  # sample-based MSE
   actor_loss = rl_loss + bc_reg_coeff * bc_loss
10. Compute gradients, NaN detection logging, apply gradients
```

**Important design choices**:

- **`_soft_clip()` for gradient flow**: Uses tanh-based saturation at boundaries instead of hard clip. Inside `[lo+margin, hi-margin]` it's identity; outside, it smoothly saturates. This ensures the actor always receives non-zero gradient, even when actions are near the bounds.

- **Full TanhNormal log_prob**: The `sample_and_log_prob()` includes the Jacobian correction `-log(1-tanh^2(z))`. This provides gradient to the mean that pushes it away from tanh saturation boundaries. Without it, means can grow unboundedly, causing all actions to saturate at +/-1.

- **`_safe_clip_for_log_prob()`**: Clamps actions to `(-1+eps, 1-eps)` before computing `log_prob()` of stored actions. This avoids `atanh(+/-1) = +/-inf` which would produce NaN gradients.

### BC Regularization (`compute_bc_loss_residual()`)

```python
policy_sample = dist.sample(seed=key)  # reparameterized for gradient to mean AND std
mse_per_sample = mean((policy_sample - stored_actions)^2, axis=-1)  # (B,)
if bc_on_success_only:
    bc_loss = sum(mse * success_mask) / sum(success_mask)
else:
    bc_loss = mean(mse_per_sample)
```

Uses **sample-based MSE** (not mode-based) to provide gradient signal to both mean and std via the reparameterization trick. Using `dist.mode()` would make the loss independent of std, causing std collapse.

### Collapse/Dominance Diagnostics

The actor update logs extensive diagnostics to detect residual collapse (delta -> 0) or dominance (delta overwhelms base):

| Metric | Meaning |
|--------|---------|
| `collapse/delta_norm_mean` | Norm of residual delta |
| `collapse/base_norm_mean` | Norm of base action |
| `collapse/eff_delta_norm_mean` | `|alpha * delta|` — effective perturbation magnitude |
| `collapse/ratio_eff_delta_to_base_mean` | Collapse (~0) vs dominance (~1+) |
| `collapse/near_zero_frac` | Fraction of delta dims < 1e-3 |
| `collapse/clip_frac` | Fraction of a_exec at +/-1 (saturation) |
| `collapse/cos_base_delta_mean` | Cosine similarity between base and delta (alignment) |
| `collapse/cos_base_exec_change_mean` | Cosine between base and (a_exec - base) |
| `collapse/alpha_temp` | Current SAC temperature |

---

## 12. BC Warmup Update

**File**: `jaxrl2/agents/pixel_sac/residual_actor_updater.py`
**Function**: `update_actor_bc_residual()`

```
1. Construct BC target:
   - predict_a_exec=False: target = zeros (learn zero residual)
   - predict_a_exec=True: target = clip(base_action, -1, 1) (learn to predict base action)
2. Forward pass: dist = actor(observations)
3. Sample-based MSE: loss = mean((dist.sample() - target)^2)
4. NaN guard on gradients, apply
```

Logs `bc_warmup/*` metrics including MSE statistics, policy distribution stats, and `base_out_of_range_frac` (fraction of Pi-0.5 outputs outside [-1, 1] before clipping — diagnostically important since Pi-0.5's quantile unnormalization can produce unbounded values).

---

## 13. Stability Mechanisms

Several mechanisms prevent training instability:

### Gradient Clipping
```python
actor_optimizer = optax.chain(
    optax.clip_by_global_norm(max_grad_norm),  # default 1.0
    optax.adam(learning_rate=actor_lr),
)
```
Both actor and critic use gradient clipping.

### NaN Guard on Gradients
```python
def _nan_to_num_tree(tree):
    return jax.tree_util.tree_map(
        lambda x: jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
        tree
    )
```
Applied to gradients in GRPO, on-policy PPO, PARL, GradQ, BC warmup, and critic updates. The residual SAC actor update logs NaN/Inf detection but does NOT mask (lets it crash for diagnosis).

### Temperature Clipping
```python
temp = clip(exp(log_temp), min_temp=0.01, max_temp=2.0)
```
Prevents entropy coefficient from collapsing to zero or exploding.

### Log-Prob Clipping
```python
log_probs_clipped = jnp.clip(log_probs, -50.0, 50.0)
```
Prevents extreme log_prob values from destabilizing the actor loss.

### Soft-Clip for Actor Gradient Flow
```python
def _soft_clip(x, lo=-1.0, hi=1.0, margin=0.05):
    mid = (hi + lo) / 2.0
    half_range = (hi - lo) / 2.0
    x_norm = (x - mid) / half_range
    return mid + half_range * jnp.tanh(x_norm)
```
Used only in the actor's forward pass (not critic). Preserves gradients at action boundaries.

### Safe Clip for Log-Prob Computation
```python
def _safe_clip_for_log_prob(x, eps=1e-6):
    return jnp.clip(x, -1.0 + eps, 1.0 - eps)
```
Prevents `atanh(+/-1) = +/-inf` when evaluating log_prob of stored/clipped actions.

### Huber Loss (Optional)
```python
def _huber_loss(x, delta=1.0):
    return where(|x| <= delta, 0.5 * x^2, delta * (|x| - 0.5 * delta))
```
Optional for critic TD error. Reduces sensitivity to large TD errors.

### NaN Guard on Rollout Actions
```python
if not np.all(np.isfinite(raw_actions)):
    raw_actions = np.nan_to_num(raw_actions, nan=0.0, posinf=0.0, neginf=0.0)
```
During trajectory collection, NaN/Inf actions are replaced with zeros.

---

## 14. Evaluation

**Function**: `perform_control_eval_residual()` in `examples/train_utils_sim_residual.py`

Runs `eval_episodes` rollouts using the **deterministic policy** (`agent.eval_actions()` = distribution mode):

```python
actions_flat = agent.eval_actions(obs_dict)  # mode (deterministic)
```

At step `i=0`, evaluation uses zero residual to measure the **base policy's performance** before any RL training.

Metrics logged:
- `evaluation/avg_return`: Mean episode return
- `evaluation/success_rate`: Fraction of successful episodes
- `evaluation/avg_episode_len`: Mean episode length
- `evaluation/delta_norm_mean`: Mean residual magnitude across eval
- `evaluation/delta_norm_std`: Std of residual magnitude
- `evaluation/base_norm_mean`: Mean base action magnitude
- `evaluation/clipping_rate_mean`: Fraction of actions at bounds
- `evaluation/Reward >= r`: Fraction achieving reward >= r
- `eval_video/{rollout_id}`: Video per evaluation rollout

---

## 15. Observation Processing Pipeline

### Raw Env Observation -> SAC Input

```
Raw obs (env-specific dict)
  |
obs_to_img(obs, variant)
  -> For LIBERO: obs["agentview_image"][::-1, ::-1]  (flip)
  -> Resize to (resize_image, resize_image) if specified
  -> Returns: (H, W, 3) uint8 image
  |
obs_to_qpos(obs, variant)
  -> [eef_pos, quat->axisangle, gripper_qpos]  (8-dim for LIBERO)
  |
obs_to_pi_zero_input(obs, variant) -> agent_dp.infer(obs_pi_zero)
  -> base_actions: (chunk_len, action_dim)
  -> vlm_embedding: (S, W) -> mean_pool -> (W,) [optional]
  |
Build obs_dict:
  obs_dict = {
      'pixels': curr_image[np.newaxis, ..., np.newaxis],      # (1, H, W, 3, 1)
      'state': qpos[np.newaxis, ..., np.newaxis],              # (1, 8, 1) [optional]
      'vlm_embedding': embedding[np.newaxis, ..., np.newaxis], # (1, W, 1) [optional]
      'base_action': base_actions[np.newaxis, ..., np.newaxis],# (1, chunk_len, action_dim, 1)
  }
  |
SAC agent processes:
  1. Encoder (or VLM embedding bypass): pixels -> latent vector
  2. _flatten_dict: concat [encoded_pixels/vlm_embedding, state, base_action] -> flat vector
  3. MLP -> action distribution (delta or a_exec)
```

### Pi-0.5 Input Format (LIBERO)

```python
obs_pi_zero = {
    "observation/image": resized_agentview (224x224),
    "observation/wrist_image": resized_wrist_cam (224x224),
    "observation/state": [eef_pos, axisangle, gripper_qpos],
    "prompt": task_description_string,
}
```

---

## 16. File Reference Map

### Core Residual SAC Agent
| File | Purpose |
|------|---------|
| `jaxrl2/agents/pixel_sac/pixel_sac_residual_learner.py` | Main Residual SAC learner class, JIT-compiled critic/actor/BC update functions |
| `jaxrl2/agents/pixel_sac/residual_actor_updater.py` | All actor update variants: SAC, GRPO, on-policy PPO, PARL, GradQ, BC warmup |
| `jaxrl2/agents/pixel_sac/residual_critic_updater.py` | Critic (Q-function) TD update with residual action composition |
| `jaxrl2/agents/pixel_sac/pixel_ppo_residual_learner.py` | GRPO / Q-weighted PG / on-policy PPO learner class |
| `jaxrl2/agents/pixel_sac/pixel_parl_residual_learner.py` | PARL (Policy-Agnostic RL) learner class |
| `jaxrl2/agents/pixel_sac/pixel_gradq_residual_learner.py` | GradQ learner class |
| `jaxrl2/agents/pixel_sac/temperature_updater.py` | Entropy temperature auto-tuning |
| `jaxrl2/agents/pixel_sac/temperature.py` | Temperature parameter module (with clipping) |
| `jaxrl2/agents/agent.py` | Base agent class with `eval_actions`, `sample_actions`, `compute_log_prob` |
| `jaxrl2/agents/common.py` | JIT-compiled action sampling/evaluation helpers |

### Network Architecture
| File | Purpose |
|------|---------|
| `jaxrl2/networks/encoders/networks.py` | `Encoder` (small CNN), `PixelMultiplexer` (encoder wrapper with `pop_base_actions`, `use_vlm_embedding`) |
| `dsrl_pi0/jaxrl2/networks/encoders/resnet_encoderv1.py` | ResNet18/34 encoder with spatial softmax |
| `dsrl_pi0/jaxrl2/networks/encoders/spatial_softmax.py` | Spatial softmax layer |
| `jaxrl2/networks/learned_std_normal_policy.py` | `LearnedStdTanhNormalPolicy`, `FixedStdTanhNormalPolicy`, `TanhMultivariateNormalDiag` |
| `dsrl_pi0/jaxrl2/networks/values/state_action_ensemble.py` | `StateActionEnsemble` (vmapped Q-ensemble) |
| `dsrl_pi0/jaxrl2/networks/values/state_action_value.py` | `StateActionValue` (single Q-function) |
| `dsrl_pi0/jaxrl2/networks/mlp.py` | `MLP` with `_flatten_dict` observation processing |
| `dsrl_pi0/jaxrl2/networks/constants.py` | Weight initializers (orthogonal, xavier, kaiming) |

### Data
| File | Purpose |
|------|---------|
| `dsrl_pi0/jaxrl2/data/replay_buffer.py` | `ReplayBuffer` with dynamic resizing, trajectory tracking, `sample_from_last_traj()` |
| `dsrl_pi0/jaxrl2/data/dataset.py` | Base `Dataset` class with sampling logic, `concat_recursive()` |
| `dsrl_pi0/jaxrl2/data/augmentations.py` | Random crop, color jitter |

### Training Loop
| File | Purpose |
|------|---------|
| `examples/train_sim_residual.py` | Main entry: env setup, base policy loading, agent creation, buffer creation |
| `examples/launch_train_sim_residual.py` | CLI argument parsing, hyperparameter defaults |
| `examples/train_utils_sim_residual.py` | Training loop, trajectory collection, reward shaping, evaluation, success buffer mixing |

### Utilities
| File | Purpose |
|------|---------|
| `dsrl_pi0/jaxrl2/utils/target_update.py` | Soft (Polyak) target update |
| `dsrl_pi0/jaxrl2/utils/launch_util.py` | Argument parsing into `AttrDict` |
| `dsrl_pi0/jaxrl2/utils/general_utils.py` | `AttrDict`, `add_batch_dim` |
| `dsrl_pi0/jaxrl2/utils/wandb_logger.py` | WandB logging wrapper |
| `dsrl_pi0/jaxrl2/types.py` | Type aliases (`PRNGKey`, `Params`, `DataType`) |

### Shell Scripts
| File | Purpose |
|------|---------|
| `examples/scripts/run_libero_residual.sh` | LIBERO Residual SAC |
| `examples/scripts/run_libero_residual_grpo.sh` | LIBERO Residual GRPO |
| `examples/scripts/run_libero_residual_parl.sh` | LIBERO Residual PARL |
| `examples/scripts/run_libero_residual_ppo.sh` | LIBERO Residual PPO |
| `examples/scripts/run_libero_residual_bc_warmup.sh` | LIBERO Residual SAC with BC warmup |
| `examples/scripts/run_libero_residual_grpo_bc_warmup.sh` | LIBERO Residual GRPO with BC warmup |
| `examples/scripts/run_libero_residual_onpolicy_ppo.sh` | LIBERO Residual on-policy PPO |
| `examples/scripts/run_cartpole_residual_sac.sh` | CartPole Residual SAC |
| `examples/scripts/run_cartpole_parl.sh` | CartPole PARL |
| `examples/scripts/run_cartpole_grpo.sh` | CartPole GRPO |
| `examples/scripts/run_cartpole_qwpg.sh` | CartPole Q-weighted PG |
