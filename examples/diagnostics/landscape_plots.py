"""UMAP action-landscape diagnostic plots L01-L05.

Visualizes the learned Q landscape around actions in a 2D UMAP embedding,
with trajectories, gradient ascent paths, and gradient line probes.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import umap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from examples.diagnostics.jax_compute import (
    compute_q_reduced,
    gradient_ascent_on_q,
    probe_q_along_gradient_line,
)
from examples.diagnostics.video_writer import save_figures_as_mp4


# ============================================================================
# Shared UMAP fitting
# ============================================================================

def fit_umap_model(all_traj_data, variant, M=10, radius=0.1):
    """Fit a UMAP model on base + exec + random-nearby actions from all eval trajs.

    Args:
        all_traj_data: List of EvalTrajectoryData.
        variant: Config with query_freq, action_dim.
        M: Number of random-nearby samples per base action.
        radius: Std for random perturbation.

    Returns:
        reducer: Fitted umap.UMAP model.
        action_mean: (action_dim_flat,) mean for standardization.
        action_std: (action_dim_flat,) std for standardization.
    """
    qf = variant.query_freq
    action_vectors = []

    for traj in all_traj_data:
        T_query = len(traj.obs_dicts) - 1
        for t in range(T_query):
            a_base_flat = np.squeeze(traj.base_actions[t])[:qf].flatten()
            a_exec_flat = traj.a_exec[t].flatten()
            action_vectors.append(a_base_flat)
            action_vectors.append(a_exec_flat)

            # Random nearby actions
            rng = np.random.RandomState(t * 1000 + len(action_vectors))
            for _ in range(M):
                noise = rng.randn(*a_base_flat.shape)
                a_rand = np.clip(a_base_flat + radius * noise, -1, 1)
                action_vectors.append(a_rand)

    A = np.stack(action_vectors)
    action_mean = A.mean(axis=0)
    action_std = A.std(axis=0) + 1e-8
    A_norm = (A - action_mean) / action_std

    reducer = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1,
        metric='euclidean', random_state=0,
    )
    reducer.fit(A_norm)
    return reducer, action_mean, action_std


def _transform_actions(reducer, actions, action_mean, action_std):
    """Transform action vectors to 2D UMAP coordinates."""
    A_norm = (actions - action_mean) / action_std
    return reducer.transform(A_norm)


def _prepare_obs(obs_dict):
    out = {}
    for k, v in obs_dict.items():
        if isinstance(v, np.ndarray):
            out[k] = jnp.asarray(v)
        else:
            out[k] = v
    return out


def _get_keyframe_strip(images, num_frames=4):
    if not images or len(images) == 0:
        return None
    n = len(images)
    indices = np.linspace(0, n - 1, num_frames, dtype=int)
    frames = [images[i] for i in indices]
    if frames[0].ndim == 3:
        return np.concatenate(frames, axis=1)
    return None


def _get_frame_at_step(images, query_step, query_freq):
    """Get the camera frame corresponding to a given query step."""
    if not images or len(images) == 0:
        return None
    env_step = max(0, min(query_step * query_freq, len(images) - 1))
    img = images[env_step]
    if img.ndim == 3:
        return img
    return None


# ============================================================================
# L01 — UMAP trajectory scatter
# ============================================================================

def plot_l01_umap_scatter(traj, traj_idx, agent_internals, variant,
                          reducer, action_mean, action_std, save_dir):
    """L01: UMAP scatter of base vs exec vs random-nearby, colored by Q, animated.

    Saved to {save_dir}/L01_umap_scatter.mp4
    """
    ai = agent_internals
    qf = variant.query_freq
    M_nearby = 5  # fewer for plotting clarity

    T_query = len(traj.obs_dicts) - 1

    # Collect all data per timestep
    base_pts_2d = []
    exec_pts_2d = []
    nearby_pts_2d = []
    base_q_vals = []
    exec_q_vals = []
    nearby_q_vals = []

    rng_np = np.random.RandomState(42)

    for t in range(T_query):
        obs_t = _prepare_obs(traj.obs_dicts[t])
        a_base_flat = np.squeeze(traj.base_actions[t])[:qf].flatten()
        a_exec_flat = traj.a_exec[t].flatten()

        # Q values
        q_base = float(compute_q_reduced(
            ai['critic_params'], ai['critic_apply_fn'],
            obs_t, jnp.asarray(a_base_flat[None]), ai['critic_reduction']).squeeze())
        q_exec = float(compute_q_reduced(
            ai['critic_params'], ai['critic_apply_fn'],
            obs_t, jnp.asarray(a_exec_flat[None]), ai['critic_reduction']).squeeze())

        base_q_vals.append(q_base)
        exec_q_vals.append(q_exec)

        # UMAP transform
        base_2d = _transform_actions(reducer, a_base_flat[None], action_mean, action_std)
        exec_2d = _transform_actions(reducer, a_exec_flat[None], action_mean, action_std)
        base_pts_2d.append(base_2d[0])
        exec_pts_2d.append(exec_2d[0])

        # Random nearby
        for _ in range(M_nearby):
            noise = rng_np.randn(*a_base_flat.shape)
            a_rand = np.clip(a_base_flat + 0.1 * noise, -1, 1)
            q_rand = float(compute_q_reduced(
                ai['critic_params'], ai['critic_apply_fn'],
                obs_t, jnp.asarray(a_rand[None]), ai['critic_reduction']).squeeze())
            rand_2d = _transform_actions(reducer, a_rand[None], action_mean, action_std)
            nearby_pts_2d.append(rand_2d[0])
            nearby_q_vals.append(q_rand)

    base_pts_2d = np.array(base_pts_2d)
    exec_pts_2d = np.array(exec_pts_2d)
    nearby_pts_2d = np.array(nearby_pts_2d) if nearby_pts_2d else np.zeros((0, 2))
    all_q = np.concatenate([base_q_vals, exec_q_vals, nearby_q_vals])
    vmin, vmax = all_q.min(), all_q.max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    figures = []
    step_interval = max(1, T_query // 15)

    for end_t in range(step_interval, T_query + 1, step_interval):
        fig, axes = plt.subplots(2, 1, figsize=(10, 10),
                                 gridspec_kw={'height_ratios': [1, 4]})
        cur_frame = _get_frame_at_step(traj.images, end_t - 1, qf)
        if cur_frame is not None:
            axes[0].imshow(cur_frame)
        axes[0].axis('off')
        axes[0].set_title(f'L01: UMAP scatter - Traj {traj_idx}, t={end_t-1} (success={traj.is_success})')

        ax = axes[1]

        # Random nearby (faint)
        n_nearby = end_t * M_nearby
        if n_nearby > 0 and len(nearby_pts_2d) > 0:
            ax.scatter(nearby_pts_2d[:n_nearby, 0], nearby_pts_2d[:n_nearby, 1],
                       c=nearby_q_vals[:n_nearby], cmap=cmap, norm=norm,
                       s=8, alpha=0.2, marker='.')

        # Base actions (circles)
        ax.scatter(base_pts_2d[:end_t, 0], base_pts_2d[:end_t, 1],
                   c=base_q_vals[:end_t], cmap=cmap, norm=norm,
                   s=40, alpha=0.8, marker='o', edgecolors='black', linewidths=0.5,
                   label='base')

        # Exec actions (triangles)
        ax.scatter(exec_pts_2d[:end_t, 0], exec_pts_2d[:end_t, 1],
                   c=exec_q_vals[:end_t], cmap=cmap, norm=norm,
                   s=40, alpha=0.8, marker='^', edgecolors='black', linewidths=0.5,
                   label='exec')

        # Arrows from base to exec
        for t in range(end_t):
            dx = exec_pts_2d[t, 0] - base_pts_2d[t, 0]
            dy = exec_pts_2d[t, 1] - base_pts_2d[t, 1]
            ax.annotate('', xy=exec_pts_2d[t], xytext=base_pts_2d[t],
                        arrowprops=dict(arrowstyle='->', color='gray',
                                        alpha=0.4, lw=0.8))

        # Connect base points over time
        if end_t > 1:
            ax.plot(base_pts_2d[:end_t, 0], base_pts_2d[:end_t, 1],
                    'k-', alpha=0.15, linewidth=0.5)

        ax.legend(loc='upper right')
        ax.set_xlabel('UMAP dim 1')
        ax.set_ylabel('UMAP dim 2')
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='Q(s,a)')

        plt.tight_layout()
        figures.append(fig)

    filepath = os.path.join(save_dir, 'L01_umap_scatter.mp4')
    save_figures_as_mp4(figures, filepath, fps=2)


# ============================================================================
# L02 — Gradient ascent from base action in UMAP
# ============================================================================

def plot_l02_grad_ascent_base(traj, traj_idx, agent_internals, variant,
                              reducer, action_mean, action_std, save_dir):
    """L02: Gradient ascent path starting from base action, animated over timesteps.

    Saved to {save_dir}/L02_grad_ascent_base.mp4
    """
    _plot_grad_ascent_umap(
        traj, traj_idx, agent_internals, variant,
        reducer, action_mean, action_std, save_dir,
        use_exec=False, plot_name='L02', filename='L02_grad_ascent_base.mp4',
    )


# ============================================================================
# L03 — Gradient ascent from exec action in UMAP
# ============================================================================

def plot_l03_grad_ascent_exec(traj, traj_idx, agent_internals, variant,
                              reducer, action_mean, action_std, save_dir):
    """L03: Gradient ascent path starting from executed action, animated.

    Saved to {save_dir}/L03_grad_ascent_exec.mp4
    """
    _plot_grad_ascent_umap(
        traj, traj_idx, agent_internals, variant,
        reducer, action_mean, action_std, save_dir,
        use_exec=True, plot_name='L03', filename='L03_grad_ascent_exec.mp4',
    )


def _plot_grad_ascent_umap(traj, traj_idx, agent_internals, variant,
                           reducer, action_mean, action_std, save_dir,
                           use_exec, plot_name, filename):
    """Shared gradient ascent plot for L02/L03."""
    ai = agent_internals
    qf = variant.query_freq
    T_query = len(traj.obs_dicts) - 1

    # Pick ~5 evenly-spaced timesteps
    n_selected = min(5, T_query)
    selected_ts = np.linspace(0, T_query - 1, n_selected, dtype=int)

    figures = []

    for t in selected_ts:
        obs_t = _prepare_obs(traj.obs_dicts[t])

        if use_exec:
            a_start = jnp.asarray(traj.a_exec[t].flatten()[None])
            start_label = 'a_exec'
            other_label = 'a_base'
        else:
            a_start = jnp.asarray(np.squeeze(traj.base_actions[t])[:qf].flatten()[None])
            start_label = 'a_base'
            other_label = 'a_exec'

        # Also get the other action for context
        a_base_flat = np.squeeze(traj.base_actions[t])[:qf].flatten()
        a_exec_flat = traj.a_exec[t].flatten()

        # Run gradient ascent
        path_actions, path_q_values = gradient_ascent_on_q(
            ai['critic_params'], ai['critic_apply_fn'], obs_t,
            a_start, step_size=0.01, num_steps=20,
            reduction=ai['critic_reduction'])

        path_actions_np = np.array(path_actions)  # (S+1, action_dim_flat)
        path_q_np = np.array(path_q_values)  # (S+1,)

        # Transform to UMAP
        path_2d = _transform_actions(reducer, path_actions_np, action_mean, action_std)
        base_2d = _transform_actions(reducer, a_base_flat[None], action_mean, action_std)
        exec_2d = _transform_actions(reducer, a_exec_flat[None], action_mean, action_std)

        # Determine start/other 2d positions
        if use_exec:
            start_2d = exec_2d[0]
            other_2d = base_2d[0]
            start_color, start_marker = 'blue', '^'
            other_color, other_marker = 'red', 'o'
        else:
            start_2d = base_2d[0]
            other_2d = exec_2d[0]
            start_color, start_marker = 'red', 'o'
            other_color, other_marker = 'blue', '^'

        fig, axes = plt.subplots(2, 1, figsize=(10, 10),
                                 gridspec_kw={'height_ratios': [1, 4]})
        cur_frame = _get_frame_at_step(traj.images, t, qf)
        if cur_frame is not None:
            axes[0].imshow(cur_frame)
        axes[0].axis('off')
        axes[0].set_title(f'{plot_name}: Grad ascent from {start_label} '
                          f'- Traj {traj_idx}, t={t}')

        ax = axes[1]

        # Gradient path colored by Q
        norm = Normalize(vmin=path_q_np.min(), vmax=path_q_np.max())
        cmap = cm.viridis

        # Draw gradient path arrows (skip first segment — drawn separately below)
        for i in range(1, len(path_2d) - 1):
            ax.annotate('', xy=path_2d[i + 1], xytext=path_2d[i],
                        arrowprops=dict(arrowstyle='->', lw=1.5,
                                        color=cmap(norm(path_q_np[i]))))
        ax.scatter(path_2d[1:, 0], path_2d[1:, 1], c=path_q_np[1:],
                   cmap=cmap, norm=norm, s=30, zorder=5, edgecolors='black',
                   linewidths=0.5)

        # Prominent outgoing arrow from start action to first gradient step
        ax.annotate('', xy=path_2d[1], xytext=start_2d,
                    arrowprops=dict(arrowstyle='->', lw=3.0,
                                    color=start_color, alpha=0.9))

        # Start action marker (prominent)
        ax.scatter(start_2d[0], start_2d[1], c=start_color, s=180,
                   marker=start_marker, zorder=10,
                   label=f'{start_label} (start)', edgecolors='black', linewidths=1.5)
        # Other action marker (context, smaller)
        ax.scatter(other_2d[0], other_2d[1], c=other_color, s=80,
                   marker=other_marker, zorder=9,
                   label=other_label, edgecolors='black', linewidths=0.5, alpha=0.6)
        # Arrow from base to exec for context
        ax.annotate('', xy=(exec_2d[0, 0], exec_2d[0, 1]),
                    xytext=(base_2d[0, 0], base_2d[0, 1]),
                    arrowprops=dict(arrowstyle='->', color='gray',
                                    alpha=0.4, lw=0.8, linestyle='--'))

        ax.legend()
        ax.set_xlabel('UMAP dim 1')
        ax.set_ylabel('UMAP dim 2')
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                     label='Q(s_t, a)')

        plt.tight_layout()
        figures.append(fig)

    filepath = os.path.join(save_dir, filename)
    save_figures_as_mp4(figures, filepath, fps=1)


# ============================================================================
# L04 — 1D gradient line at base action in UMAP
# ============================================================================

def plot_l04_grad_line_base(traj, traj_idx, agent_internals, variant,
                            reducer, action_mean, action_std, save_dir):
    """L04: 1D gradient line probe at base action, animated over timesteps.

    Saved to {save_dir}/L04_grad_line_base.mp4
    """
    _plot_grad_line_umap(
        traj, traj_idx, agent_internals, variant,
        reducer, action_mean, action_std, save_dir,
        use_exec=False, plot_name='L04', filename='L04_grad_line_base.mp4',
    )


# ============================================================================
# L05 — 1D gradient line at executed action in UMAP
# ============================================================================

def plot_l05_grad_line_exec(traj, traj_idx, agent_internals, variant,
                            reducer, action_mean, action_std, save_dir):
    """L05: 1D gradient line probe at executed action, animated.

    Saved to {save_dir}/L05_grad_line_exec.mp4
    """
    _plot_grad_line_umap(
        traj, traj_idx, agent_internals, variant,
        reducer, action_mean, action_std, save_dir,
        use_exec=True, plot_name='L05', filename='L05_grad_line_exec.mp4',
    )


def _plot_grad_line_umap(traj, traj_idx, agent_internals, variant,
                         reducer, action_mean, action_std, save_dir,
                         use_exec, plot_name, filename):
    """Shared 1D gradient line plot for L04/L05."""
    ai = agent_internals
    qf = variant.query_freq
    T_query = len(traj.obs_dicts) - 1

    n_selected = min(5, T_query)
    selected_ts = np.linspace(0, T_query - 1, n_selected, dtype=int)

    figures = []

    for t in selected_ts:
        obs_t = _prepare_obs(traj.obs_dicts[t])

        if use_exec:
            a0 = jnp.asarray(traj.a_exec[t].flatten()[None])
            anchor_label = 'a_exec'
        else:
            a0 = jnp.asarray(np.squeeze(traj.base_actions[t])[:qf].flatten()[None])
            anchor_label = 'a_base'

        lambdas, q_values, line_actions = probe_q_along_gradient_line(
            ai['critic_params'], ai['critic_apply_fn'], obs_t,
            a0, num_points=21, line_range=0.3,
            reduction=ai['critic_reduction'])

        lambdas_np = np.array(lambdas)
        q_values_np = np.array(q_values)
        line_actions_np = np.array(line_actions)

        # UMAP transform
        line_2d = _transform_actions(reducer, line_actions_np, action_mean, action_std)

        norm = Normalize(vmin=q_values_np.min(), vmax=q_values_np.max())
        cmap = cm.viridis

        fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                                 gridspec_kw={'height_ratios': [1, 4],
                                              'width_ratios': [1, 1]})
        # Top row: current frame
        cur_frame = _get_frame_at_step(traj.images, t, qf)
        if cur_frame is not None:
            axes[0, 0].imshow(cur_frame)
        axes[0, 0].axis('off')
        axes[0, 0].set_title(f'{plot_name}: Grad line at {anchor_label} - t={t}')
        axes[0, 1].axis('off')

        # Bottom left: UMAP
        ax_umap = axes[1, 0]
        ax_umap.plot(line_2d[:, 0], line_2d[:, 1], 'k-', alpha=0.3, linewidth=1)
        sc = ax_umap.scatter(line_2d[:, 0], line_2d[:, 1], c=q_values_np,
                             cmap=cmap, norm=norm, s=30, zorder=5,
                             edgecolors='black', linewidths=0.3)
        # Highlight lambda=0 (anchor)
        mid_idx = len(lambdas_np) // 2
        ax_umap.scatter(line_2d[mid_idx, 0], line_2d[mid_idx, 1],
                        c='red', s=100, marker='*', zorder=10, label=anchor_label)
        ax_umap.legend()
        ax_umap.set_xlabel('UMAP dim 1')
        ax_umap.set_ylabel('UMAP dim 2')
        ax_umap.set_title('UMAP embedding')
        fig.colorbar(sc, ax=ax_umap, label='Q(s_t, a)')

        # Bottom right: 1D Q(lambda)
        ax_1d = axes[1, 1]
        ax_1d.plot(lambdas_np, q_values_np, 'o-', color='steelblue', linewidth=1.5)
        ax_1d.axvline(0, color='red', linestyle='--', alpha=0.7, label=f'lambda=0 ({anchor_label})')
        ax_1d.set_xlabel('lambda (along gradient direction)')
        ax_1d.set_ylabel('Q(s_t, a(lambda))')
        ax_1d.set_title('Q along gradient direction')
        ax_1d.legend()
        ax_1d.grid(True, alpha=0.3)

        plt.tight_layout()
        figures.append(fig)

    filepath = os.path.join(save_dir, filename)
    save_figures_as_mp4(figures, filepath, fps=1)
