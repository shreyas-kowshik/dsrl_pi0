"""Launch script for Residual RL training in simulation.

This script provides CLI argument parsing for Residual SAC/PPO/GRPO training.
Key additions:
- --residual_alpha for controlling the residual scaling factor.
- --algo for selecting algorithm: 'residual_sac', 'residual_ppo', 'residual_grpo'
"""

import argparse
import sys
from examples.train_sim_residual import main_residual
from jaxrl2.utils.launch_util import parse_training_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Residual RL Training in Simulation')

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
    parser.add_argument('--algorithm', default='residual_sac', help='Type of algorithm (for wandb naming)')
    parser.add_argument('--prefix', default='', help='Prefix to use for wandb')
    parser.add_argument('--suffix', default='', help='Suffix to use for wandb')
    parser.add_argument('--multi_grad_step', default=1, help='Gradient steps per env step (UTD)', type=int)
    parser.add_argument('--resize_image', default=-1, help='Size of image if need resizing', type=int)
    parser.add_argument('--query_freq', default=-1, help='Query frequency', type=int)
    parser.add_argument('--pi_05_config', default='', help='Config name for Pi-0.5 model', type=str)
    parser.add_argument('--pi_05_ckpt_dir', default='', help='Checkpoint dir for Pi-0.5 model', type=str)
    
    # Residual-specific parameters
    parser.add_argument('--residual_alpha', default=0.1, help='Scaling factor for residual actions', type=float)
    parser.add_argument('--chunk_len', default=10, help='Action chunk length (Pi-0.5 horizon)', type=int)
    parser.add_argument('--use_zero_residual_initially', default=1, help='Use zero residual for first trajectory (1=yes, 0=no)', type=int)
    
    # Algorithm selection: 'residual_sac', 'q_weighted_pg', 'residual_grpo'
    parser.add_argument('--algo', default='residual_sac', help='Algorithm: residual_sac, q_weighted_pg, residual_grpo', type=str)
    
    # PPO/GRPO specific parameters (adv_clip_min/max for clipping advantages)
    parser.add_argument('--actor_tau', default=0.005, help='[DEPRECATED] Target actor soft update rate (not used anymore)', type=float)
    parser.add_argument('--grpo_num_samples', default=8, help='Number of action samples per state for GRPO', type=int)
    parser.add_argument('--clip_epsilon', default=0.2, help='PPO clip epsilon', type=float)
    parser.add_argument('--clip_min_epsilon_multiplier', default=1.0, help='Multiplier for lower bound of PPO clip', type=float)
    parser.add_argument('--clip_max_epsilon_multiplier', default=1.0, help='Multiplier for upper bound of PPO clip', type=float)
    parser.add_argument('--entropy_coeff', default=1e-3, help='Entropy bonus coefficient (for PPO/GRPO)', type=float)
    parser.add_argument('--advantage_critic_reduction', default='mean', help='How to reduce Q ensemble for advantages (min, mean)', type=str)
    parser.add_argument('--adv_clip_min', default=None, help='Optional lower bound for advantage clipping', type=float)
    parser.add_argument('--adv_clip_max', default=None, help='Optional upper bound for advantage clipping', type=float)
    
    # Stability parameters (new)
    parser.add_argument('--log_ratio_clip', default=20.0, help='Clamp log_ratio to [-clip, clip] before exp', type=float)
    parser.add_argument('--log_prob_clip', default=50.0, help='Clamp log_probs to [-clip, clip]', type=float)
    parser.add_argument('--max_grad_norm', default=1.0, help='Max gradient norm for clipping', type=float)
    parser.add_argument('--use_huber_loss', default=0, help='Use Huber loss for critic (1=yes, 0=no)', type=int)
    parser.add_argument('--huber_delta', default=1.0, help='Delta parameter for Huber loss', type=float)
    
    # Update ratio control (new)
    parser.add_argument('--num_critic_updates', default=2, help='Number of critic updates per batch', type=int)
    parser.add_argument('--num_actor_updates', default=4, help='Number of actor updates per batch', type=int)

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
    variant['use_huber_loss'] = bool(variant.get('use_huber_loss', 0))
    
    algo = variant.get('algo', 'residual_sac')
    print("=" * 60)
    print(f"RESIDUAL RL CONFIGURATION ({algo.upper()})")
    print("=" * 60)
    print(f"  algo: {algo}")
    print(f"  residual_alpha: {variant.get('residual_alpha', 0.1)}")
    print(f"  chunk_len: {variant.get('chunk_len', 10)}")
    print(f"  use_zero_residual_initially: {variant.get('use_zero_residual_initially', True)}")
    if algo in ['q_weighted_pg', 'residual_grpo']:
        print(f"  grpo_num_samples: {variant.get('grpo_num_samples', 8)}")
        print(f"  clip_epsilon: {variant.get('clip_epsilon', 0.2)}")
        print(f"  clip_min_epsilon_multiplier: {variant.get('clip_min_epsilon_multiplier', 1.0)}")
        print(f"  clip_max_epsilon_multiplier: {variant.get('clip_max_epsilon_multiplier', 1.0)}")
        print(f"  entropy_coeff: {variant.get('entropy_coeff', 1e-3)}")
        print(f"  advantage_critic_reduction: {variant.get('advantage_critic_reduction', 'mean')}")
        print(f"  adv_clip_min: {variant.get('adv_clip_min', None)}")
        print(f"  adv_clip_max: {variant.get('adv_clip_max', None)}")
        print("  --- Stability parameters ---")
        print(f"  log_ratio_clip: {variant.get('log_ratio_clip', 20.0)}")
        print(f"  log_prob_clip: {variant.get('log_prob_clip', 50.0)}")
        print(f"  max_grad_norm: {variant.get('max_grad_norm', 1.0)}")
        print(f"  use_huber_loss: {variant.get('use_huber_loss', False)}")
        print(f"  huber_delta: {variant.get('huber_delta', 1.0)}")
        print("  --- Update ratio ---")
        print(f"  num_critic_updates: {variant.get('num_critic_updates', 2)}")
        print(f"  num_actor_updates: {variant.get('num_actor_updates', 4)}")
    print("=" * 60)
    print(variant)
    
    main_residual(variant)
    sys.exit()
