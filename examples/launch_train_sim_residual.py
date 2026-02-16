"""Launch script for Residual RL training in simulation.

This script provides CLI argument parsing for Residual SAC/PPO/GRPO training.
Key additions:
- --residual_alpha for controlling the residual scaling factor.
- --algo for selecting algorithm: 'sac', 'residual_sac', 'q_weighted_pg', 'residual_grpo'
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
    parser.add_argument('--env', default='libero', help='Name of environment (libero, aloha_cube, cartpole)')
    parser.add_argument('--log_interval', default=1000, help='Logging interval.', type=int)
    parser.add_argument('--eval_interval', default=5000, help='Eval interval.', type=int)
    parser.add_argument('--diagnostic_freq', default=1, help='Generate diagnostics for every Nth eval trajectory (1=all, 2=half, etc). 0 disables diagnostics.', type=int)
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
    parser.add_argument('--libero_task', default='', help='LIBERO task name (e.g. KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it)', type=str)
    
    # CartPole-specific parameters
    parser.add_argument('--cartpole_horizon', default=100, help='Episode horizon for CartPole env', type=int)
    
    # Reward shaping
    parser.add_argument('--reward_type', default='sparse', help='Reward type: sparse (-1/0 binary) or dense (raw env reward)', type=str, choices=['sparse', 'dense'])
    
    # Residual-specific parameters
    parser.add_argument('--residual_alpha', default=0.1, help='Scaling factor for residual actions', type=float)
    parser.add_argument('--chunk_len', default=10, help='Action chunk length (Pi-0.5 horizon)', type=int)
    parser.add_argument('--use_zero_residual_initially', default=1, help='Use zero residual for first trajectory (1=yes, 0=no)', type=int)
    parser.add_argument('--predict_a_exec', default=0, help='Actor predicts a_exec directly instead of delta (1=yes, 0=no)', type=int)
    parser.add_argument('--learn_std', default=1, help='Whether the policy learns a state-dependent std (1=yes, 0=no)', type=int)
    
    # Algorithm selection: 'sac', 'residual_sac', 'q_weighted_pg', 'residual_grpo', 'residual_parl', 'residual_gradq'
    parser.add_argument('--algo', default='residual_sac', help='Algorithm: sac, residual_sac, q_weighted_pg, residual_grpo, residual_parl, residual_gradq', type=str)
    
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
    
    # BC regularization parameters
    parser.add_argument('--bc_reg_coeff', default=0.0, help='BC regularization coefficient (0 = disabled)', type=float)
    parser.add_argument('--bc_on_success_only', default=0, help='BC loss only on success transitions (1=yes, 0=no)', type=int)
    
    # Success buffer parameters
    parser.add_argument('--success_buffer_ratio', default=0.0, help='Fraction of actor batch from success buffer (0 = disabled, e.g. 0.2 for 20%%)', type=float)
    parser.add_argument('--success_buffer_min_size', default=100, help='Min samples in success buffer before using it', type=int)
    
    # PARL (Policy-Agnostic RL) parameters
    parser.add_argument('--parl_num_samples', default=16, help='N: number of action candidates sampled from actor', type=int)
    parser.add_argument('--parl_num_elites', default=4, help='K: number of top-Q actions kept for gradient refinement', type=int)
    parser.add_argument('--parl_num_grad_steps', default=5, help='Number of gradient ascent steps on Q w.r.t. action', type=int)
    parser.add_argument('--parl_step_size', default=0.01, help='Step size (learning rate) for gradient ascent on actions', type=float)

    # On-policy PPO parameters
    parser.add_argument('--on_policy_ppo', default=0, help='Use on-policy PPO with stored log_probs (1=yes, 0=no)', type=int)
    parser.add_argument('--normalize_advantages', default=0, help='Normalize advantages (1=yes, 0=no)', type=int)
    
    # Actor architecture / optimizer
    parser.add_argument('--actor_hidden_dims', default='', help='Actor MLP hidden dims (comma-separated, e.g. "256,256"). Empty=use --hidden_dims.', type=str)
    parser.add_argument('--actor_optimizer', default='adam', help='Optimizer for actor: adam or sgd', type=str)

    # Policy std bounds (NaN stability)
    parser.add_argument('--log_std_min', default=-5.0, help='Min log_std for policy (NaN stability)', type=float)
    parser.add_argument('--log_std_max', default=2.0, help='Max log_std for policy', type=float)
    
    # BC warmup parameters
    parser.add_argument('--bc_warmup_steps', default=0, help='Number of gradient steps for BC warmup (0=disabled)', type=int)
    parser.add_argument('--bc_warmup_num_critic_updates', default=10, help='Critic updates per grad step during BC warmup (aggressive)', type=int)
    parser.add_argument('--bc_warmup_num_actor_updates', default=1, help='Actor BC updates per grad step during BC warmup (light)', type=int)

    # VLM embedding parameters
    parser.add_argument('--use_vlm_embedding', default=0, help='Use VLM embeddings instead of raw pixels (1=yes, 0=no)', type=int)
    parser.add_argument('--vlm_embedding_dim', default=2048, help='Hidden dim of VLM embedding (W in [B,S,W])', type=int)
    parser.add_argument('--vlm_seq_len', default=16, help='Sequence length of VLM embedding (S in [B,S,W])', type=int)

    # Checkpoint resume
    parser.add_argument('--restore_path', default='', help='Path to checkpoint dir to resume training from', type=str)

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
        critic_pop_base_actions=True,
        clip_temp=True,
        clip_min_temp=0.01,
        clip_max_temp=2.0,
    )

    variant, args = parse_training_args(train_args_dict, parser)
    
    # Convert flag to boolean
    variant['use_zero_residual_initially'] = bool(variant.get('use_zero_residual_initially', 1))
    variant['predict_a_exec'] = bool(variant.get('predict_a_exec', 0))
    variant['backup_entropy'] = bool(variant.get('backup_entropy', 0))
    variant['critic_pop_base_actions'] = bool(variant.get('critic_pop_base_actions', 0))
    variant['clip_temp'] = bool(variant.get('clip_temp', 1))
    variant['use_huber_loss'] = bool(variant.get('use_huber_loss', 0))
    variant['bc_on_success_only'] = bool(variant.get('bc_on_success_only', 0))
    variant['on_policy_ppo'] = bool(variant.get('on_policy_ppo', 0))
    variant['normalize_advantages'] = bool(variant.get('normalize_advantages', 0))
    variant['learn_std'] = bool(variant.get('learn_std', 1))
    variant['use_vlm_embedding'] = bool(variant.get('use_vlm_embedding', 0))

    # Parse actor_hidden_dims: comma-separated string -> tuple of ints, or None
    actor_hd_str = variant.get('actor_hidden_dims', '')
    if actor_hd_str and str(actor_hd_str).strip():
        variant['actor_hidden_dims'] = tuple(int(x) for x in str(actor_hd_str).split(','))
    else:
        variant['actor_hidden_dims'] = None  # will default to hidden_dims in learner

    algo = variant.get('algo', 'residual_sac')
    print("=" * 60)
    print(f"RESIDUAL RL CONFIGURATION ({algo.upper()})")
    print("=" * 60)
    print(f"  algo: {algo}")
    print(f"  reward_type: {variant.get('reward_type', 'sparse')}")
    print(f"  residual_alpha: {variant.get('residual_alpha', 0.1)}")
    print(f"  chunk_len: {variant.get('chunk_len', 10)}")
    print(f"  use_zero_residual_initially: {variant.get('use_zero_residual_initially', True)}")
    print(f"  predict_a_exec: {variant.get('predict_a_exec', False)}")
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
    print(f"  learn_std: {variant.get('learn_std', True)}")
    print("  --- BC Regularization ---")
    print(f"  bc_reg_coeff: {variant.get('bc_reg_coeff', 0.0)}")
    print(f"  bc_on_success_only: {variant.get('bc_on_success_only', False)}")
    print(f"  success_buffer_ratio: {variant.get('success_buffer_ratio', 0.0)}")
    print(f"  success_buffer_min_size: {variant.get('success_buffer_min_size', 100)}")
    print("  --- On-policy PPO ---")
    print(f"  on_policy_ppo: {variant.get('on_policy_ppo', False)}")
    print(f"  normalize_advantages: {variant.get('normalize_advantages', False)}")
    print(f"  log_std_min: {variant.get('log_std_min', -5.0)}")
    print(f"  log_std_max: {variant.get('log_std_max', 2.0)}")
    print("  --- BC Warmup ---")
    print(f"  bc_warmup_steps: {variant.get('bc_warmup_steps', 0)}")
    print(f"  bc_warmup_num_critic_updates: {variant.get('bc_warmup_num_critic_updates', 10)}")
    print(f"  bc_warmup_num_actor_updates: {variant.get('bc_warmup_num_actor_updates', 1)}")
    print("  --- VLM Embedding ---")
    print(f"  use_vlm_embedding: {variant.get('use_vlm_embedding', False)}")
    print(f"  vlm_embedding_dim: {variant.get('vlm_embedding_dim', 2048)}")
    print(f"  vlm_seq_len: {variant.get('vlm_seq_len', 16)}")
    print("=" * 60)
    print(variant)
    
    main_residual(variant)
    sys.exit()
