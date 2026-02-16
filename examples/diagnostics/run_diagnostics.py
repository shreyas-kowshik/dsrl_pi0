"""Top-level orchestrator for generating all diagnostic plots during evaluation.

Called from perform_control_eval_residual after trajectories are collected.
Generates Q-function (Q01-Q10), actor (E01-E02), and UMAP landscape (L01-L05)
diagnostic plots as animated .mp4 files.
"""

import os
import traceback
import numpy as np
import jax.numpy as jnp

from examples.diagnostics.q_plots import (
    plot_q01_multistep_consistency,
    plot_q02_q_vs_qtarg,
    plot_q03_td_error_trajectory,
    plot_q04_q_base_trajectory,
    plot_q05_q_exec_trajectory,
    plot_q06_grad_norm_exec,
    plot_q07_grad_norm_base,
    plot_q08_td_histogram,
    plot_q09_q_variance_base,
    plot_q10_q_variance_exec,
)
from examples.diagnostics.actor_plots import (
    plot_e01_delta_q,
    plot_e02_action_traces,
)
from examples.diagnostics.landscape_plots import (
    fit_umap_model,
    plot_l01_umap_scatter,
    plot_l02_grad_ascent_base,
    plot_l03_grad_ascent_exec,
    plot_l04_grad_line_base,
    plot_l05_grad_line_exec,
)


def _get_agent_batch_stats(agent):
    """Safely get actor batch_stats from agent."""
    actor = agent._actor
    if hasattr(actor, 'batch_stats'):
        return actor.batch_stats
    return None


def _build_agent_internals(agent, variant):
    """Extract agent components into a dict for passing to plot generators.

    Handles compatibility across SAC, PARL, and GradQ learners.
    """
    # Temperature: SAC has _temp, PARL/GradQ may not
    has_temp = hasattr(agent, '_temp') and agent._temp is not None
    if has_temp:
        temp_apply_fn = agent._temp.apply_fn
        temp_params = agent._temp.params
    else:
        # Create a dummy temperature that returns 0 (no entropy term)
        class _DummyTemp:
            @staticmethod
            def apply_fn(params):
                return 0.0
        temp_apply_fn = _DummyTemp.apply_fn
        temp_params = {}

    # Determine num_qs from critic output
    num_qs = getattr(agent, 'num_qs', 2)
    if not hasattr(agent, 'num_qs'):
        # Try to infer from critic init params
        num_qs = 2  # default

    return {
        'actor_apply_fn': agent._actor.apply_fn,
        'actor_params': agent._actor.params,
        'actor_batch_stats': _get_agent_batch_stats(agent),
        'critic_apply_fn': agent._critic.apply_fn,
        'critic_params': agent._critic.params,
        'target_critic_params': agent._target_critic_params,
        'temp_apply_fn': temp_apply_fn,
        'temp_params': temp_params,
        'residual_alpha': agent._residual_alpha,
        'critic_reduction': getattr(agent, 'critic_reduction', 'mean'),
        'num_qs': num_qs,
    }


def generate_all_diagnostics(agent, all_traj_data, step_i, variant):
    """Generate all diagnostic plots and save as .mp4 files.

    Directory structure:
        {variant.outputdir}/diagnostics/step_{step_i}/
            aggregate/
                Q01_multistep_consistency.mp4
                Q02_q_vs_qtarg.mp4
                Q08_td_histogram.mp4
            traj_{rollout_id}/
                Q03_td_error.mp4
                Q04_q_base.mp4
                Q05_q_exec.mp4
                Q06_grad_norm_exec.mp4
                Q07_grad_norm_base.mp4
                Q09_q_variance_base.mp4
                Q10_q_variance_exec.mp4
                E01_delta_q.mp4
                E02_action_traces.mp4
                L01_umap_scatter.mp4
                L02_grad_ascent_base.mp4
                L03_grad_ascent_exec.mp4
                L04_grad_line_base.mp4
                L05_grad_line_exec.mp4

    Args:
        agent: Residual RL agent (SAC, PARL, or GradQ).
        all_traj_data: List of EvalTrajectoryData from evaluation rollouts.
        step_i: Current training step.
        variant: Training config.
    """
    if not all_traj_data:
        print('[Diagnostics] No trajectory data, skipping.')
        return

    base_dir = os.path.join(variant.outputdir, 'diagnostics', f'step_{step_i}')
    aggregate_dir = os.path.join(base_dir, 'aggregate')
    os.makedirs(aggregate_dir, exist_ok=True)

    print(f'[Diagnostics] Generating plots for step {step_i} '
          f'({len(all_traj_data)} trajectories) -> {base_dir}')

    agent_internals = _build_agent_internals(agent, variant)

    # ------------------------------------------------------------------
    # Per-trajectory plots: Q03-Q07, E01-E02
    # ------------------------------------------------------------------
    all_td_errors_per_traj = []

    for traj_idx, traj in enumerate(all_traj_data):
        traj_dir = os.path.join(base_dir, f'traj_{traj_idx}')
        os.makedirs(traj_dir, exist_ok=True)

        # Q03: TD-error over time (also returns TD data for Q08)
        try:
            td_errors = plot_q03_td_error_trajectory(
                traj, traj_idx, agent_internals, variant, traj_dir)
            all_td_errors_per_traj.append(td_errors)
        except Exception as e:
            print(f'[Diagnostics] Q03 failed for traj {traj_idx}: {e}')
            traceback.print_exc()
            all_td_errors_per_traj.append([])

        # Q04: Q(base) over time
        try:
            plot_q04_q_base_trajectory(traj, traj_idx, agent_internals, variant, traj_dir)
        except Exception as e:
            print(f'[Diagnostics] Q04 failed for traj {traj_idx}: {e}')
            traceback.print_exc()

        # Q05: Q(exec) over time
        try:
            plot_q05_q_exec_trajectory(traj, traj_idx, agent_internals, variant, traj_dir)
        except Exception as e:
            print(f'[Diagnostics] Q05 failed for traj {traj_idx}: {e}')
            traceback.print_exc()

        # Q06: Gradient norm at exec
        try:
            plot_q06_grad_norm_exec(traj, traj_idx, agent_internals, variant, traj_dir)
        except Exception as e:
            print(f'[Diagnostics] Q06 failed for traj {traj_idx}: {e}')
            traceback.print_exc()

        # Q07: Gradient norm at base
        try:
            plot_q07_grad_norm_base(traj, traj_idx, agent_internals, variant, traj_dir)
        except Exception as e:
            print(f'[Diagnostics] Q07 failed for traj {traj_idx}: {e}')
            traceback.print_exc()

        # Q09: Q variance across sampled base actions
        try:
            plot_q09_q_variance_base(traj, traj_idx, agent_internals, variant, traj_dir)
        except Exception as e:
            print(f'[Diagnostics] Q09 failed for traj {traj_idx}: {e}')
            traceback.print_exc()

        # Q10: Q variance across sampled exec actions
        try:
            plot_q10_q_variance_exec(traj, traj_idx, agent_internals, variant, traj_dir)
        except Exception as e:
            print(f'[Diagnostics] Q10 failed for traj {traj_idx}: {e}')
            traceback.print_exc()

        # E01: Delta-Q
        try:
            plot_e01_delta_q(traj, traj_idx, agent_internals, variant, traj_dir)
        except Exception as e:
            print(f'[Diagnostics] E01 failed for traj {traj_idx}: {e}')
            traceback.print_exc()

        # E02: Action traces
        try:
            plot_e02_action_traces(traj, traj_idx, agent_internals, variant, traj_dir)
        except Exception as e:
            print(f'[Diagnostics] E02 failed for traj {traj_idx}: {e}')
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Aggregate plots: Q01, Q02, Q08
    # ------------------------------------------------------------------

    # Q01: Multi-step consistency
    try:
        plot_q01_multistep_consistency(all_traj_data, agent_internals, variant, aggregate_dir)
    except Exception as e:
        print(f'[Diagnostics] Q01 failed: {e}')
        traceback.print_exc()

    # Q02: Q vs Q_targ scatter
    try:
        plot_q02_q_vs_qtarg(all_traj_data, agent_internals, variant, aggregate_dir)
    except Exception as e:
        print(f'[Diagnostics] Q02 failed: {e}')
        traceback.print_exc()

    # Q08: TD-error histogram
    try:
        plot_q08_td_histogram(all_td_errors_per_traj, all_traj_data, aggregate_dir)
    except Exception as e:
        print(f'[Diagnostics] Q08 failed: {e}')
        traceback.print_exc()

    # ------------------------------------------------------------------
    # UMAP landscape plots: L01-L05 (per trajectory)
    # ------------------------------------------------------------------
    try:
        print('[Diagnostics] Fitting UMAP model...')
        reducer, action_mean, action_std = fit_umap_model(all_traj_data, variant)
        print('[Diagnostics] UMAP model fitted.')

        for traj_idx, traj in enumerate(all_traj_data):
            traj_dir = os.path.join(base_dir, f'traj_{traj_idx}')

            try:
                plot_l01_umap_scatter(traj, traj_idx, agent_internals, variant,
                                      reducer, action_mean, action_std, traj_dir)
            except Exception as e:
                print(f'[Diagnostics] L01 failed for traj {traj_idx}: {e}')
                traceback.print_exc()

            try:
                plot_l02_grad_ascent_base(traj, traj_idx, agent_internals, variant,
                                          reducer, action_mean, action_std, traj_dir)
            except Exception as e:
                print(f'[Diagnostics] L02 failed for traj {traj_idx}: {e}')
                traceback.print_exc()

            try:
                plot_l03_grad_ascent_exec(traj, traj_idx, agent_internals, variant,
                                          reducer, action_mean, action_std, traj_dir)
            except Exception as e:
                print(f'[Diagnostics] L03 failed for traj {traj_idx}: {e}')
                traceback.print_exc()

            try:
                plot_l04_grad_line_base(traj, traj_idx, agent_internals, variant,
                                        reducer, action_mean, action_std, traj_dir)
            except Exception as e:
                print(f'[Diagnostics] L04 failed for traj {traj_idx}: {e}')
                traceback.print_exc()

            try:
                plot_l05_grad_line_exec(traj, traj_idx, agent_internals, variant,
                                        reducer, action_mean, action_std, traj_dir)
            except Exception as e:
                print(f'[Diagnostics] L05 failed for traj {traj_idx}: {e}')
                traceback.print_exc()

    except Exception as e:
        print(f'[Diagnostics] UMAP fitting/plots failed: {e}')
        traceback.print_exc()

    print(f'[Diagnostics] Done generating plots for step {step_i}.')
