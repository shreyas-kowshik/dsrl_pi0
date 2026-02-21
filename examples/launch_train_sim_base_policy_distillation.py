"""Launch script for Filtered Behavior Cloning / Base Policy Distillation.

This script provides CLI argument parsing for filtered BC training where:
1. A base policy (Pi-0.5) collects trajectories
2. Only successful trajectories are used for fine-tuning
3. The process repeats for multiple rounds
"""

import argparse
import sys
from examples.train_sim_base_policy_distillation import main_base_policy_distillation
from jaxrl2.utils.general_utils import AttrDict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filtered Behavior Cloning / Base Policy Distillation')

    # Basic training parameters
    parser.add_argument('--seed', default=42, help='Random seed.', type=int)
    parser.add_argument('--launch_group_id', default='', help='Group id used to group runs on wandb.')
    parser.add_argument('--eval_episodes', default=10, help='Number of episodes used for evaluation.', type=int)
    parser.add_argument('--env', default='libero', help='Name of environment (libero)')
    parser.add_argument('--log_interval', default=100, help='Logging interval (within each training round).', type=int)
    parser.add_argument('--wandb_project', default='filtered_bc_sim', help='WandB project name')
    parser.add_argument('--prefix', default='', help='Prefix to use for wandb')
    parser.add_argument('--suffix', default='', help='Suffix to use for wandb')

    # Base policy parameters
    parser.add_argument('--pi_05_config', default='', help='Config name for Pi-0.5 model', type=str)
    parser.add_argument('--pi_05_ckpt_dir', default='', help='Checkpoint dir for Pi-0.5 model', type=str)
    parser.add_argument('--libero_task', default='', help='LIBERO task name', type=str)
    parser.add_argument('--task_suite_name', default='libero_10', help='LIBERO task suite name', type=str)
    parser.add_argument('--task_id', default=8, help='Task ID within the suite (0-indexed)', type=int)

    # Action chunk parameters
    parser.add_argument('--chunk_len', default=10, help='Action chunk length (Pi-0.5 horizon used for execution)', type=int)
    parser.add_argument('--query_freq', default=10, help='Query frequency (env steps per agent action query)', type=int)

    # Filtered BC parameters
    parser.add_argument('--num_rounds', default=5, help='Number of collect-then-train rounds', type=int)
    parser.add_argument('--num_collect_trajectories', default=50, help='Number of trajectories to collect per round', type=int)
    parser.add_argument('--num_train_steps_per_round', default=1000, help='Number of gradient steps per round', type=int)
    parser.add_argument('--batch_size', default=8, help='Mini batch size for training', type=int)

    # Training parameters
    parser.add_argument('--checkpoint_interval', default=1, help='Save checkpoint every N rounds (0=disabled)', type=int)

    # Action sample filtering
    parser.add_argument('--drop_short_actions', default=1,
                        help='Drop training samples with fewer than action_horizon remaining steps (1=yes, 0=no). '
                             'If 0, pads with env-specific padding (Libero: [0]*6+[1]).', type=int)

    # Cumulative data
    parser.add_argument('--cumulative_data', default=0,
                        help='If 1, keep all successful trajectories from previous rounds and train on the '
                             'cumulative buffer. If 0, only train on the current round trajectories.', type=int)

    # Learning rate override
    parser.add_argument('--flat_lr', default=None, type=float,
                        help='If provided, override the config LR schedule with a flat (constant) learning rate. '
                             'Uses the same default AdamW optimizer from the pi0.5 config, but with this fixed LR.')
    parser.add_argument('--warmup_steps', default=None, type=int,
                        help='If provided, override the warmup_steps in the LR schedule from the config.')

    # Expert data
    parser.add_argument('--load_expert_data', default=0,
                        help='If 1, load expert trajectories from the path specified by --expert_data_path '
                             'and always include them in training as successful demonstrations.', type=int)
    parser.add_argument('--expert_data_path', default='/home/skowshik/vla/codebase/openpi/data_dumps',
                        help='Path to a JSON dump file or directory of JSON dumps (from '
                             'openpi/scripts/dump_filtered_data.py) containing expert episode indices.',
                        type=str)

    args = parser.parse_args()
    variant = AttrDict(vars(args))

    # Convert flags to booleans
    variant['drop_short_actions'] = bool(variant.get('drop_short_actions', 1))
    variant['cumulative_data'] = bool(variant.get('cumulative_data', 0))
    variant['load_expert_data'] = bool(variant.get('load_expert_data', 0))

    # Print configuration
    print("=" * 60)
    print("FILTERED BEHAVIOR CLONING CONFIGURATION")
    print("=" * 60)
    print(f"  env: {variant.env}")
    print(f"  pi_05_config: {variant.pi_05_config}")
    print(f"  pi_05_ckpt_dir: {variant.pi_05_ckpt_dir}")
    print(f"  libero_task: {variant.libero_task}")
    print(f"  task_suite_name: {variant.task_suite_name}")
    print(f"  task_id: {variant.task_id}")
    print(f"  chunk_len: {variant.chunk_len}")
    print(f"  query_freq: {variant.query_freq}")
    print(f"  --- Filtered BC ---")
    print(f"  num_rounds: {variant.num_rounds}")
    print(f"  num_collect_trajectories: {variant.num_collect_trajectories}")
    print(f"  num_train_steps_per_round: {variant.num_train_steps_per_round}")
    print(f"  batch_size: {variant.batch_size}")
    print(f"  drop_short_actions: {variant.drop_short_actions}")
    print(f"  cumulative_data: {variant.cumulative_data}")
    if variant.get('flat_lr') is not None:
        print(f"  flat_lr: {variant.flat_lr}")
    if variant.get('warmup_steps') is not None:
        print(f"  warmup_steps (override): {variant.warmup_steps}")
    print(f"  load_expert_data: {variant.load_expert_data}")
    if variant.load_expert_data:
        print(f"  expert_data_path: {variant.expert_data_path}")
    print(f"  --- Eval ---")
    print(f"  eval_episodes: {variant.eval_episodes}")
    print(f"  checkpoint_interval: {variant.checkpoint_interval}")
    print("=" * 60)
    print(variant)

    main_base_policy_distillation(variant)
    sys.exit()
