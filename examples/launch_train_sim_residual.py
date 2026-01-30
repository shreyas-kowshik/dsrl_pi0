"""Launch script for Residual SAC training in simulation.

This script provides CLI argument parsing for Residual SAC training.
Key addition: --residual_alpha for controlling the residual scaling factor.
"""

import argparse
import sys
from examples.train_sim_residual import main_residual
from jaxrl2.utils.launch_util import parse_training_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Residual SAC Training in Simulation')

    # Basic training parameters
    parser.add_argument('--seed', default=42, help='Random seed.', type=int)
    parser.add_argument('--launch_group_id', default='', help='Group id used to group runs on wandb.')
    parser.add_argument('--eval_episodes', default=10, help='Number of episodes used for evaluation.', type=int)
    parser.add_argument('--env', default='libero', help='Name of environment (libero, aloha_cube)')
    parser.add_argument('--log_interval', default=1000, help='Logging interval.', type=int)
    parser.add_argument('--eval_interval', default=5000, help='Eval interval.', type=int)
    parser.add_argument('--checkpoint_interval', default=-1, help='Checkpoint interval.', type=int)
    parser.add_argument('--batch_size', default=16, help='Mini batch size.', type=int)
    parser.add_argument('--max_steps', default=int(1e6), help='Number of training steps.', type=int)
    parser.add_argument('--add_states', default=1, help='Whether to add low-dim states to observations', type=int)
    parser.add_argument('--wandb_project', default='residual_sac_sim', help='WandB project name')
    parser.add_argument('--start_online_updates', default=1000, help='Steps to collect before starting updates', type=int)
    parser.add_argument('--algorithm', default='residual_sac', help='Type of algorithm')
    parser.add_argument('--prefix', default='', help='Prefix to use for wandb')
    parser.add_argument('--suffix', default='', help='Suffix to use for wandb')
    parser.add_argument('--multi_grad_step', default=1, help='Gradient steps per env step (UTD)', type=int)
    parser.add_argument('--resize_image', default=-1, help='Size of image if need resizing', type=int)
    parser.add_argument('--query_freq', default=-1, help='Query frequency', type=int)
    parser.add_argument('--pi_05_config', default='', help='Config name for Pi-0.5 model', type=str)
    parser.add_argument('--pi_05_ckpt_dir', default='', help='Checkpoint dir for Pi-0.5 model', type=str)
    
    # Residual SAC specific parameters
    parser.add_argument('--residual_alpha', default=0.1, help='Scaling factor for residual actions', type=float)
    parser.add_argument('--chunk_len', default=10, help='Action chunk length (Pi-0.5 horizon)', type=int)
    parser.add_argument('--use_zero_residual_initially', default=1, help='Use zero residual for first trajectory (1=yes, 0=no)', type=int)

    # Default training kwargs
    train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        hidden_dims=(256, 256, 256),
        cnn_features=(64, 64, 64, 64),
        cnn_strides=(2, 1, 1, 1),
        cnn_padding='VALID',
        latent_dim=200,
        discount=0.999,
        tau=0.005,
        critic_reduction='mean',
        dropout_rate=0.0,
        aug_next=1,
        use_bottleneck=True,
        encoder_type='small',
        encoder_norm='group',
        use_spatial_softmax=True,
        softmax_temperature=-1,
        target_entropy='auto', #maybe 7.0
        num_qs=10,
        action_magnitude=1.0,
        num_cameras=1,
        backup_entropy=False,
        critic_pop_base_actions=False,
        clip_temp=True,
        clip_min_temp=0.01,
        clip_max_temp=2.0,
    )

    variant, args = parse_training_args(train_args_dict, parser)
    
    # Convert flag to boolean
    variant['use_zero_residual_initially'] = bool(variant.get('use_zero_residual_initially', 1))
    variant['backup_entropy'] = bool(variant.get('backup_entropy', 0))
    variant['critic_pop_base_actions'] = bool(variant.get('critic_pop_base_actions', 0))
    variant['clip_temp'] = bool(variant.get('clip_temp', 1))
    print("=" * 60)
    print("RESIDUAL SAC CONFIGURATION")
    print("=" * 60)
    print(f"  residual_alpha: {variant.get('residual_alpha', 0.1)}")
    print(f"  chunk_len: {variant.get('chunk_len', 10)}")
    print(f"  use_zero_residual_initially: {variant.get('use_zero_residual_initially', True)}")
    print("=" * 60)
    print(variant)
    
    main_residual(variant)
    sys.exit()
