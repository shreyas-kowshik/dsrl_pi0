"""Edit actor (residual policy) diagnostic plots E01-E02.

E01: Delta-Q along eval trajectories (Q(exec) - Q(base))
E02: Action traces per dimension with residual delta distribution
"""

import os
import numpy as np
import jax.numpy as jnp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from examples.diagnostics.jax_compute import (
    compute_q_values_all_heads,
    get_actor_distribution_params,
)
from examples.diagnostics.video_writer import save_figures_as_mp4


def _get_keyframe_strip(images, num_frames=4):
    """Select evenly-spaced keyframes and concatenate horizontally."""
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


def _prepare_obs(obs_dict):
    out = {}
    for k, v in obs_dict.items():
        if isinstance(v, np.ndarray):
            out[k] = jnp.asarray(v)
        else:
            out[k] = v
    return out


def _get_a_exec_flat(traj, t, qf):
    return jnp.asarray(traj.a_exec[t].reshape(1, -1))


def _get_a_base_flat(traj, t, qf):
    base = np.squeeze(traj.base_actions[t])
    return jnp.asarray(base[:qf].reshape(1, -1))


# ============================================================================
# E01 — Delta-Q along trajectory
# ============================================================================

def plot_e01_delta_q(traj, traj_idx, agent_internals, variant, save_dir):
    """E01: Delta-Q = Q(s, a_exec) - Q(s, a_base) per head, animated.

    Saved to {save_dir}/E01_delta_q.mp4
    """
    ai = agent_internals
    qf = variant.query_freq

    T_query = len(traj.obs_dicts) - 1
    dq_means = []
    dq_stds = []

    for t in range(T_query):
        obs_t = _prepare_obs(traj.obs_dicts[t])
        a_exec_flat = _get_a_exec_flat(traj, t, qf)
        a_base_flat = _get_a_base_flat(traj, t, qf)

        qs_exec = np.array(compute_q_values_all_heads(
            ai['critic_params'], ai['critic_apply_fn'], obs_t, a_exec_flat))
        qs_base = np.array(compute_q_values_all_heads(
            ai['critic_params'], ai['critic_apply_fn'], obs_t, a_base_flat))

        dq = qs_exec[:, 0] - qs_base[:, 0]  # (num_qs,)
        dq_means.append(dq.mean())
        dq_stds.append(dq.std())

    dq_means = np.array(dq_means)
    dq_stds = np.array(dq_stds)

    figures = []
    step_interval = max(1, T_query // 20)

    for end_t in range(step_interval, T_query + 1, step_interval):
        fig, axes = plt.subplots(2, 1, figsize=(10, 7),
                                 gridspec_kw={'height_ratios': [1, 3]})
        cur_frame = _get_frame_at_step(traj.images, end_t - 1, qf)
        if cur_frame is not None:
            axes[0].imshow(cur_frame)
        axes[0].axis('off')
        axes[0].set_title(f'Traj {traj_idx}, t={end_t-1} (success={traj.is_success})')

        t_range = np.arange(end_t)
        axes[1].plot(t_range, dq_means[:end_t], color='teal', linewidth=1.5,
                     label='DQ mean')
        axes[1].fill_between(t_range,
                             dq_means[:end_t] - dq_stds[:end_t],
                             dq_means[:end_t] + dq_stds[:end_t],
                             alpha=0.25, color='teal')
        axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Query step t')
        axes[1].set_ylabel('Q(a_exec) - Q(a_base)')
        axes[1].set_title('E01: Delta-Q (residual value improvement)')
        axes[1].set_xlim(0, T_query)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Summary bar
        dq_mean_ep = dq_means[:end_t].mean()
        axes[1].text(0.02, 0.98, f'Avg DQ = {dq_mean_ep:.4f}',
                     transform=axes[1].transAxes, verticalalignment='top',
                     fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        figures.append(fig)

    filepath = os.path.join(save_dir, 'E01_delta_q.mp4')
    save_figures_as_mp4(figures, filepath, fps=2)


# ============================================================================
# E02 — Action traces per dimension
# ============================================================================

def plot_e02_action_traces(traj, traj_idx, agent_internals, variant, save_dir):
    """E02: Per-dimension action traces with residual delta distribution, animated.

    Shows base action, delta mean±std, and executed action per action dimension.
    Saved to {save_dir}/E02_action_traces.mp4
    """
    ai = agent_internals
    qf = variant.query_freq
    action_dim = variant.action_dim
    residual_alpha = float(ai['residual_alpha'])
    predict_a_exec = variant.get('predict_a_exec', False)

    T_query = len(traj.obs_dicts) - 1

    # Collect per-step distribution parameters and actions
    all_base = []  # (T_query, action_dim)
    all_exec = []  # (T_query, action_dim)
    all_delta_mean = []  # (T_query, action_dim)
    all_delta_std = []   # (T_query, action_dim)

    for t in range(T_query):
        obs_t = _prepare_obs(traj.obs_dicts[t])

        # Get actor distribution parameters (pre-tanh)
        pre_tanh_mean, pre_tanh_std = get_actor_distribution_params(
            ai['actor_apply_fn'], ai['actor_params'],
            ai['actor_batch_stats'], obs_t)
        pre_tanh_mean = np.array(pre_tanh_mean)  # (qf * action_dim,)
        pre_tanh_std = np.array(pre_tanh_std)

        # Post-tanh mean approximation: tanh(mean)
        post_tanh_mean = np.tanh(pre_tanh_mean)
        # For the std, approximate: std * (1 - tanh(mean)^2)
        post_tanh_std = pre_tanh_std * (1 - post_tanh_mean ** 2 + 1e-6)

        # Reshape to (qf, action_dim) and take first step
        delta_mean = post_tanh_mean.reshape(qf, action_dim)[0]  # first query step
        delta_std = post_tanh_std.reshape(qf, action_dim)[0]

        # Scaled delta
        if not predict_a_exec:
            delta_mean_scaled = residual_alpha * delta_mean
            delta_std_scaled = residual_alpha * delta_std
        else:
            delta_mean_scaled = delta_mean
            delta_std_scaled = delta_std

        # Base and exec at first step of chunk
        base_t = np.squeeze(traj.base_actions[t])[0]  # (action_dim,)
        exec_t = traj.a_exec[t][0]  # (action_dim,)

        all_base.append(base_t)
        all_exec.append(exec_t)
        all_delta_mean.append(delta_mean_scaled)
        all_delta_std.append(delta_std_scaled)

    all_base = np.array(all_base)  # (T_query, action_dim)
    all_exec = np.array(all_exec)
    all_delta_mean = np.array(all_delta_mean)
    all_delta_std = np.array(all_delta_std)

    # Determine number of dimensions to plot (cap at 8 for readability)
    n_dims = min(action_dim, 8)

    figures = []
    step_interval = max(1, T_query // 15)

    for end_t in range(step_interval, T_query + 1, step_interval):
        fig, axes = plt.subplots(n_dims + 1, 1, figsize=(12, 2.5 * (n_dims + 1)),
                                 gridspec_kw={'height_ratios': [1] + [2] * n_dims})
        cur_frame = _get_frame_at_step(traj.images, end_t - 1, qf)
        if cur_frame is not None:
            axes[0].imshow(cur_frame)
        axes[0].axis('off')
        axes[0].set_title(f'E02: Action traces - Traj {traj_idx}, t={end_t-1} (success={traj.is_success})')

        t_range = np.arange(end_t)

        for d in range(n_dims):
            ax = axes[d + 1]
            ax.plot(t_range, all_base[:end_t, d], color='gray', linewidth=1.5,
                    label='a_base')
            ax.plot(t_range, all_exec[:end_t, d], color='steelblue', linewidth=1.5,
                    label='a_exec')

            # Delta mean ± std (centered at base for visual overlay)
            delta_center = all_base[:end_t, d] + all_delta_mean[:end_t, d]
            ax.plot(t_range, delta_center, color='coral', linewidth=1, linestyle='--',
                    label='base + delta_mean')
            ax.fill_between(t_range,
                            delta_center - all_delta_std[:end_t, d],
                            delta_center + all_delta_std[:end_t, d],
                            alpha=0.2, color='coral')

            ax.axhline(-1, color='black', linestyle=':', alpha=0.3)
            ax.axhline(1, color='black', linestyle=':', alpha=0.3)
            ax.set_ylabel(f'dim {d}')
            ax.set_xlim(0, T_query)
            ax.grid(True, alpha=0.2)
            if d == 0:
                ax.legend(loc='upper right', fontsize=7)

        axes[-1].set_xlabel('Query step t')
        plt.tight_layout()
        figures.append(fig)

    filepath = os.path.join(save_dir, 'E02_action_traces.mp4')
    save_figures_as_mp4(figures, filepath, fps=2)
