# plots_summary.md — Residual SAC diagnostics plots (implementation spec)

This document is written to be handed to an LLM that has access to the codebase. It specifies **what to compute** and **how to plot it** for each diagnostic plot requested.  
Each plot below is separated by a blank line (per request).

---

## Global notation / shared conventions

**Trajectory data (evaluation rollouts):**
- For each evaluation episode `e`, you have a sequence of length `T_e`:
  - `obs_t` (full observation dict; includes pixels/state and base_action chunk)
  - `reward_t`
  - `terminated_t` (true terminal; environment-defined success/failure)
  - `truncated_t` (time-limit or external cutoff)
  - `a_base_t`: the **base** environment action actually proposed at timestep `t` (typically the first element of the base action chunk for that timestep or decision point).
  - `a_exec_t`: the **executed** environment action at timestep `t` (after residual composition + clip).
  - `delta_t`: the residual policy output in environment action space (if `predict_a_exec=False`), such that:
    - `a_exec = clip(a_base + residual_alpha * delta, -1, 1)` for each step (or per chunk if using action chunks).
    - If `predict_a_exec=True`, then `a_exec` is directly predicted by the residual actor; `delta` can be defined for logging as `(a_exec - a_base)/residual_alpha` where unclipped.

**Critic ensemble outputs:**
- `Q_i(s, a)` for `i=1..N` critics (e.g. `N=num_qs`).
- `Q_mean(s,a) = mean_i Q_i(s,a)`
- `Q_std(s,a) = std_i Q_i(s,a)`
- `Q_min(s,a) = min_i Q_i(s,a)` (optional; if you want a conservative view).
- `Q_targ_i(s,a)` are the target critic heads (Polyak / EMA).
- Unless stated otherwise:
  - use **online critics** for the main y-axis, and
  - use **target critics** for bootstraps / “targ” comparisons.

**State input `s`:**
- The residual RL MDP state is `s = (obs, base_action_chunk)` (base action is part of the observation dict).
- When plotting “Q(s,a_base)” at timestep `t`, ensure the critic sees the same `s_t` that was used to produce (or store) that base action.

**Action representation:**
- If actions are chunked as `(query_freq, action_dim)`, define a consistent flattening:
  - `a_flat = a.reshape(-1)` for embedding / UMAP and for gradient norms.
- For per-timestep plots, prefer the actual per-environment-step action vector `(action_dim,)`.  
  If your stored action is chunked, either:
  1) unroll it to per-step actions for plotting, or  
  2) plot per-decision-point with `x = decision_index` rather than env step.

**Temperature / entropy term (SAC):**
- Let `alpha_temp` be the entropy coefficient (temperature).
- For diagnostics that need a soft state value, define:

  `V_soft(s) = E_{u ~ pi(u|s)}[ Q_reduce(s, a_exec(u,s)) - alpha_temp * log_pi(u|s) ]`

  where:
  - `u` is the stochastic variable sampled from the actor (often the residual delta pre-squash / post-squash depending on implementation),
  - `a_exec(u,s)` is how you map the actor sample to an executed env action (compose with base and clip/soft-clip),
  - `Q_reduce` is either `min` or `mean` across critics, whichever your SAC backup uses (make this a flag).
- Estimate `V_soft` with Monte-Carlo sampling using `K=10` samples by default.

**Styling conventions:**
- Time axis: `t = 0..T-1`.
- Where requested, include **mean across critics as the line** and **std as a shaded band**.
- When showing multiple trajectories, either:
  - plot a small panel per trajectory (facets), or
  - aggregate with median + percentile bands across episodes.

---

# Q function diagnostics

## Plot Q01 — Multi-step consistency curve (“Q vs n-step bootstrapped return”)

**Goal:** Measure critic correctness with a TD-consistent target that does *not* suffer from the “Q vs truncated MC return” mismatch.

**Inputs:**
- One (or many) evaluation trajectories: `(s_t, a_exec_t, r_t, terminated_t, truncated_t)`.
- Current actor `pi` (residual policy) with ability to sample actions and compute log-prob (or at least sample actions).
- Target critic ensemble `Q_targ`.
- Discount `gamma`.
- Temperature `alpha_temp`.
- Hyperparameters: `K=10` (samples for `V_soft`), and a set of `n` values, e.g. `n_list = [1, 2, 4, 8, 16, 32]`.

**Computation:**
1. Pick a set of time indices `t` (e.g., all timesteps from all eval trajectories, or a subsample like 2000 points).
2. For each chosen `t` and each `n`:
   - Let `t_n = min(t+n, T-1)` (or stop earlier if episode ends).
   - Compute the n-step discounted reward sum:
     `R_t^(n) = sum_{k=0..n-1} gamma^k * r_{t+k}`  (only over valid steps).
   - Determine if you should bootstrap at `s_{t+n}`:
     - If the episode **terminated** before `t+n`, bootstrap term is 0.
     - If the episode is **truncated** at `t+n` (time-limit) OR you intentionally cut the trajectory at `t+n`, bootstrap with `V_soft(s_{t+n})`.
   - Estimate `V_soft(s_{t+n})` using `K` samples:
     - For `k=1..K`:
       - sample actor variable `u_k ~ pi(u|s_{t+n})` (e.g., residual delta sample)
       - map to executed env action `a_k = a_exec(u_k, s_{t+n})`
       - compute `q_k = Q_reduce_targ(s_{t+n}, a_k)` (reduce across target critics)
       - compute `ent_k = - alpha_temp * log_pi(u_k | s_{t+n})` (if log_prob available; else set ent_k=0 and note this)
       - contribution: `v_k = q_k + ent_k`
     - `V_soft_hat = mean_k v_k`
   - Define the n-step bootstrapped return:
     `G_t^(n) = R_t^(n) + gamma^n * bootstrap_mask * V_soft_hat`
     where `bootstrap_mask = 1` if not terminated before `t+n`, and (optionally) `bootstrap_mask = 1` only for truncation/time-limit (configurable).
3. For each `(t,n)` also compute critic prediction:
   - `Q_pred = Q_reduce_online(s_t, a_exec_t)` and optionally keep full ensemble vector for mean/std.

**Plot:**
- X-axis: `n` (log-scale is often useful).
- Y-axis: error statistics between `Q_pred` and `G_t^(n)`:
  - primary: `MAE_n = mean_t |Q_pred(t) - G_t^(n)|`
  - optional: `Bias_n = mean_t (Q_pred - G_t^(n))`
  - optional: `RMSE_n`
- Show a curve for each of:
  - `Q_reduce = mean` and/or `min` (if both are relevant).
- Add an inset / second panel: scatter `Q_pred` vs `G_t^(n*)` for a fixed `n*` (e.g., `n*=16`) with a `y=x` line.

**Interpretation cues to include in plot captions:**
- Errors should generally decrease as training progresses.
- If 1-step matches but large-n deviates, suspect reward/mask/termination handling or target drift.
- If all n are bad, critic is not learning or inputs are wrong.

---

## Plot Q02 — Q(s,a) vs Q_targ(s,a) on initial buffer trajectories

**Goal:** Check target network tracking and critic stability early (and over time).

**Inputs:**
- A set of transitions or short trajectories from the **initial replay buffer** (pre-training) OR from a fixed “probe” dataset.
- Online critic ensemble `Q_online`.
- Target critic ensemble `Q_targ`.
- Use the same action `a` as stored in the buffer (executed action).

**Computation:**
- For each sampled `(s,a)`:
  - `q_online = Q_reduce_online(s,a)` (or keep per-head)
  - `q_targ = Q_reduce_targ(s,a)`
- Optionally also compute `q_std_online` and `q_std_targ`.

**Plot:**
- Scatter plot:
  - X-axis: `q_targ`
  - Y-axis: `q_online`
  - Add reference diagonal `y=x`.
- Optionally:
  - color points by training step (if plotting over time),
  - or by TD-error magnitude,
  - or by episode outcome (success vs failure).
- Annotate with correlation and slope from a linear fit.

**Expected behavior:**
- Early: noisy but correlated; as training stabilizes, points cluster near y=x.

---

## Plot Q03 — TD-error across time within a single evaluation trajectory

**Goal:** Visualize where the critic is “surprised” along a rollout (large TD errors often coincide with failure modes, OOD states, or reward/termination bugs).

**Inputs:**
- One evaluation trajectory `(s_t, a_exec_t, r_t, s_{t+1}, terminated_t, truncated_t)`.
- Target policy sampling (for SAC backup): action sample at `s_{t+1}`.
- Target critic ensemble.
- Temperature `alpha_temp`, discount `gamma`.

**Computation (per timestep t):**
1. Sample next action (or residual variable) from actor at `s_{t+1}` to form `a_next`:
   - `a_next = a_exec(u_next, s_{t+1})`, where `u_next ~ pi(.|s_{t+1})`.
2. Compute target:
   - `y_t = r_t + gamma * not_terminal_mask * ( Q_reduce_targ(s_{t+1}, a_next) - alpha_temp * log_pi(u_next|s_{t+1}) )`
   - `not_terminal_mask` should be 0 for true termination; for truncation it should usually be 1 (configurable).
3. Compute prediction:
   - `q_pred_t = Q_reduce_online(s_t, a_exec_t)`
4. TD error:
   - `td_t = q_pred_t - y_t`
   - Also compute `td_mean_t`, `td_std_t` across critics if you want per-head TD.

**Plot:**
- Line plot over time `t`:
  - primary line: `td_mean_t`
  - shaded band: `± td_std_t` across critics
- Add horizontal zero line.
- Optionally add markers when `terminated` or `truncated` happens.
- Put the trajectory’s keyframe image (robot camera view) as a thin strip above the plot, or at least show image at a few key timesteps (start/mid/end).

---

## Plot Q04 — Q-values of base action across a trajectory (with critic ensemble mean±std, plus MC returns row)

**Goal:** See what the critic thinks about the *base* policy’s action along the rollout.

**Inputs:**
- One evaluation trajectory with per-timestep `s_t`.
- Base action to evaluate: `a_base_t` (per timestep).
- Online critic ensemble.
- Returns `G_t` (either truncated MC return or your preferred bootstrapped return; include which one in the plot title).

**Computation:**
- For each timestep `t`:
  - `q_base_heads_t = [Q_i_online(s_t, a_base_t)]_{i=1..N}`
  - `q_base_mean_t = mean_i q_base_heads_t`
  - `q_base_std_t = std_i q_base_heads_t`
  - `return_t = MC_return(t)` (or `bootstrapped_return(t)`; compute consistently for the whole trajectory)

**Plot layout (3-row panel for a single trajectory):**
- Row 1: “Return over time”
  - plot `return_t` as a line.
- Row 2: “Q(base) over time”
  - plot `q_base_mean_t` with shaded `± q_base_std_t`.
- Row 3: optional: “Ensemble disagreement”
  - plot `q_base_std_t` alone, or plot `max-min` across heads.

**Axes:**
- X-axis shared: timestep `t`.
- Y-axis row1: returns; row2: Q; row3: std.

---

## Plot Q05 — Q-values of executed/edit action across a trajectory (with critic ensemble mean±std, plus MC returns row)

**Goal:** Same as Q04, but for the actual executed residual-composed action.

**Inputs/Computation:**
- Identical to Q04, but use `a_exec_t`.

**Plot layout:**
- Same 3-row panel as Q04:
  - Row 1: returns over time
  - Row 2: `Q(a_exec)` mean±std
  - Row 3: disagreement metric (optional)

**Recommended extra overlay:**
- In Row 2, also overlay `Q(base)` mean as a thinner line so you can directly compare in the same chart:
  - executed line thicker, base line thinner/dashed.

---

## Plot Q06 — Gradient norm of Q w.r.t executed action along the trajectory (mean±std across critics)

**Goal:** Detect action-sensitivity (if gradients are ~0, critic may ignore action; if exploding, critic may be unstable). Also used later for “grad ascent” landscape plots.

**Inputs:**
- One evaluation trajectory `(s_t, a_exec_t)`.
- Online critic ensemble.
- Autodiff over action input.

**Computation (per timestep t):**
- For each critic head `i`:
  - compute `g_i_t = ∇_a Q_i_online(s_t, a_exec_t)` (gradient w.r.t action; hold s fixed)
  - `gnorm_i_t = ||g_i_t||_2` (if action is chunked, gradient is on flattened action)
- Aggregate:
  - `gnorm_mean_t = mean_i gnorm_i_t`
  - `gnorm_std_t = std_i gnorm_i_t`

**Plot:**
- Line over time: `gnorm_mean_t` with shaded `± gnorm_std_t`.
- Optional log-scale on y-axis if ranges are large.

---

## Plot Q07 — Gradient norm of Q w.r.t base action along the trajectory (mean±std across critics)

**Goal:** Same as Q06 but for base action; compares sensitivity in base vs executed regions.

**Inputs/Computation:**
- Same as Q06 but evaluate gradients at `a_base_t`.

**Plot:**
- Same format as Q06.
- Optional: overlay Q06 and Q07 in the same plot with two lines (executed vs base) for direct comparison.

---

## Plot Q08 — Histogram of TD errors across all timesteps in evaluation trajectories

**Goal:** Global view of TD-error distribution (heavy tails imply instability, bad masking, reward scale issues, etc.).

**Inputs:**
- A batch of evaluation trajectories.
- TD error definition exactly as in Q03.

**Computation:**
- For every timestep across all trajectories:
  - compute `td_t` (either using 1-sample SAC backup or using mean of K samples; choose one and keep it consistent).
- Optionally keep separate TD distributions for:
  - successes vs failures
  - terminated vs truncated episodes
  - early vs late timesteps

**Plot:**
- Histogram of `td_t` (and optionally `|td_t|`).
- Add vertical lines for mean, median, and 95th percentile.
- Optional: multiple histograms side-by-side (facets) for the splits listed above.

---

# Q landscape (UMAP action-space visualization)

## Shared setup for UMAP plots (L01–L05)

**Goal:** Visualize the learned Q landscape around actions actually taken, in a 2D embedding, with trajectories and gradients.

**Inputs:**
- A set of evaluation trajectories (recommend 5–20 episodes).
- For each timestep, collect:
  - `s_t`
  - `a_base_t`
  - `a_exec_t`
- Critic ensemble (online, or target—choose one and label it; online is more “current”).
- UMAP library (`umap-learn`).

**Action dataset for fitting UMAP:**
1. Collect all action vectors you want to embed:
   - all `a_base_t` and `a_exec_t` from chosen trajectories.
2. Optionally add “local random actions” around each base action:
   - For each `a_base_t`, sample `M` actions:
     - `a_rand = clip(a_base_t + radius * noise, -1, 1)`
     - noise can be `Normal(0, I)` or uniform; recommend normal then normalize direction.
   - Suggested: `M=10`, `radius ∈ {0.05, 0.1, 0.2}` (configurable).
3. Stack to matrix `A` shape `(num_points, action_dim)` (or flattened chunk size).

**UMAP fit:**
- Optionally standardize actions before UMAP (mean-0, std-1 per dimension).
- Fit a single UMAP model once per evaluation checkpoint:
  - `umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=0).fit(A)`
- Transform all points to 2D coordinates `Z`.

**Q coloring for embedded points:**
- For each point action `a` paired with a state `s`:
  - compute `Q_mean(s,a)` (mean across critics)
- For random actions sampled around `a_base_t`, use the *same state s_t* when evaluating Q, so the color reflects local action landscape at that state.

**Robot image panel:**
- For each plotted trajectory, select an image from the environment observation (e.g., `obs_t['pixels']` or a chosen camera) to show above the UMAP plot.  
  Use either:
  - the first frame,
  - the frame where failure occurs, or
  - a small strip of 3 keyframes.

---

## Plot L01 — UMAP trajectory scatter: base vs executed vs random-nearby actions, colored by Q

**Goal:** “Bird’s-eye view” of where base and executed actions lie in action manifold and whether residual moves toward higher-Q regions.

**Per-trajectory procedure (repeat for each chosen eval trajectory):**
1. Create a two-row figure:
   - Top: robot image (from a representative timestep in that trajectory).
   - Bottom: UMAP scatter plot.
2. For each timestep `t` in the trajectory:
   - Plot the 2D point for `a_base_t` using marker style A (e.g., circle).
   - Plot the 2D point for `a_exec_t` using marker style B (e.g., triangle).
   - Draw a small arrow from base point to executed point.
   - Next to the arrow (or in a side text box), display `delta = a_exec_t - a_base_t` summary:
     - either L2 norm `||delta||`,
     - or per-dimension small text for a subset of dims.
3. For each timestep `t`, also plot the random-nearby actions (sampled around base):
   - use faint small dots.
4. Color all points by `Q_mean(s_t, action)` using a continuous colormap.
5. Include a colorbar labeled “Q(s_t, a)”.

**Plot details:**
- Legend entries: base actions, executed actions, random-nearby.
- Optional: connect base points over time with a faint line to show progression.

---

## Plot L02 — Gradient-ascent path starting from base action (in UMAP), colored by Q

**Goal:** Show whether following `∇_a Q(s,a)` from the base action moves into a higher-Q region, and how that path relates to the executed residual action.

**Inputs:**
- Choose a specific state `s_t` (or a few `t`s) within an eval trajectory.
- Starting action: `a0 = a_base_t`.
- Step size `eta` and number of ascent steps `S` (e.g., `eta=0.01`, `S=20`).
- Gradient computed using an agreed reducer:
  - `Q_reduce_online` or `Q_reduce_targ` (choose; label in title).

**Computation:**
1. Initialize `a = a0`.
2. For step `k=0..S-1`:
   - compute gradient `g = ∇_a Q_reduce(s_t, a)` (w.r.t action)
   - optionally normalize `g = g / (||g|| + eps)` for stable steps
   - update `a <- clip(a + eta * g, -1, 1)`
   - record `(a_k, Q(s_t, a_k))`
3. Embed all `a_k` using the same fitted UMAP model (use `.transform()`).

**Plot:**
- Two-row figure: top robot image, bottom UMAP plot.
- Plot the gradient ascent path as a connected line through the embedded points.
- Color the path points by `Q(s_t, a_k)` (or annotate with values).
- Also plot the original base action point and the executed action point at that same timestep for context.

---

## Plot L03 — Gradient-ascent path starting from executed action (in UMAP), colored by Q

**Goal:** Same as L02, but start from the executed action to see if residual already sits near a local optimum or can still be improved.

**Inputs/Computation:**
- Same as L02 but start `a0 = a_exec_t`.

**Plot:**
- Same layout and overlays as L02.

---

## Plot L04 — 1D gradient line at the base action (in UMAP), colored by Q

**Goal:** Probe local Q landscape along the *gradient direction at base*, without iterating (simpler and more stable).

**Inputs:**
- Pick a state `s_t` and `a0 = a_base_t`.
- Compute gradient direction at `a0`: `g0 = ∇_a Q_reduce(s_t, a0)`.
- Normalize `g0`.

**Computation:**
- Sample a set of points along the gradient direction:
  - choose scalars `lambda ∈ [-L, ..., 0, ..., +L]` (e.g., 21 points)
  - `a(lambda) = clip(a0 + lambda * g0, -1, 1)`
  - compute `Q(s_t, a(lambda))`
  - embed `a(lambda)` to UMAP.

**Plot:**
- Two-row figure: robot image on top, UMAP on bottom.
- Plot the embedded line points connected in order of lambda.
- Color points by Q value.
- Highlight lambda=0 point as the base action.

**Optional second panel (very useful):**
- A simple 1D plot: x-axis lambda, y-axis Q(s_t, a(lambda)).

---

## Plot L05 — 1D gradient line at the executed action (in UMAP), colored by Q

**Goal:** Same as L04, but evaluate local landscape at executed action.

**Inputs/Computation:**
- Same as L04 but with `a0 = a_exec_t`.

**Plot:**
- Same as L04.

---

# Edit actor (residual policy) diagnostics

## Plot E01 — Delta-Q along eval trajectories: Q(s, executed_action) − Q(s, base_action)

**Goal:** Measure if residual edits are improving value *according to the critic* along the rollout, and how confident the critic ensemble is about that improvement.

**Inputs:**
- Evaluation trajectory: `s_t`, `a_base_t`, `a_exec_t`.
- Online critic ensemble.

**Computation (per timestep t):**
- For each critic head i:
  - `dq_i_t = Q_i(s_t, a_exec_t) - Q_i(s_t, a_base_t)`
- Aggregate:
  - `dq_mean_t = mean_i dq_i_t`
  - `dq_std_t = std_i dq_i_t`

**Plot:**
- Line plot over time:
  - line: `dq_mean_t`
  - shaded: `± dq_std_t`
- Additionally, include a compact “error bar summary” panel:
  - compute `dq_mean_episode = mean_t dq_mean_t`
  - compute `dq_std_episode = mean_t dq_std_t` (or std over timesteps)
  - show a bar with error bar for each episode, or for success vs failure groups.

**Optional overlay:**
- Also show success indicator (binary) or reward on a secondary axis.

---

## Plot E02 — Action traces per dimension across a trajectory, overlaying residual delta distribution

**Goal:** Inspect per-dimension action behavior and whether the residual actor’s mean/std makes sense (e.g., does it saturate, does it spike near failures, does it collapse to ~0).

**Inputs:**
- One evaluation trajectory.
- For each timestep:
  - base action vector `a_base_t[d]`
  - executed action vector `a_exec_t[d]` (optional overlay)
  - residual policy distribution parameters per dimension:
    - `mu_delta_t[d]` and `std_delta_t[d]` for the residual (if Gaussian/TanhNormal)
- Residual scaling `residual_alpha`.

**Computation:**
- If residual policy outputs deltas in (-1,1):
  - `delta_mean_t = tanh(mu_pre_t)` or directly the post-squash mean depending on implementation.
  - `delta_std_t` should be mapped consistently (if you only have pre-tanh std, note it; otherwise approximate).
- Compute optional “scaled delta”:
  - `delta_scaled_mean_t = residual_alpha * delta_mean_t`
  - `delta_scaled_std_t = residual_alpha * delta_std_t`

**Plot layout (for a single trajectory):**
- Top row: robot image (representative frame or a small strip).
- Below: one subplot per action dimension `d=1..action_dim` stacked vertically, sharing x-axis (timestep).
  For each dimension subplot:
  - Plot base action `a_base_t[d]` as a solid line.
  - Overlay residual delta mean as a second line:
    - either raw `delta_mean_t[d]` (label “delta”),
    - or scaled `delta_scaled_mean_t[d]` (label “alpha*delta”) — recommended so units match actions.
  - Shade the delta uncertainty band:
    - `delta_mean ± delta_std` (or scaled variant).
  - Optional: also plot executed action `a_exec_t[d]` as a third line (helps verify composition).
- Add horizontal lines at `-1` and `+1` (action bounds).
- Optionally annotate points where hard clipping occurred (e.g., markers when |a_exec| ≈ 1).

**Caption interpretation cues:**
- If delta mean collapses to 0 everywhere, residual is not contributing.
- If delta mean saturates and action frequently clips, residual is too aggressive or actor/logprob is broken.
- Std collapsing too early can indicate exploration issues / BC warmup issues.

---
