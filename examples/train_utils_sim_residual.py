"""Training utilities for Residual SAC in simulation.

This module implements the residual SAC training loop where:
- A frozen base policy (Pi-0.5) produces base action chunks
- SAC predicts residual actions in environment action space
- Executed actions are: a_exec = clip(base_action + alpha * delta, -1, 1)

The key difference from DSRL-pi is that SAC operates in env action space,
not diffusion noise space.
"""

from tqdm import tqdm
import numpy as np
import wandb
import jax
import jax.numpy as jnp
from openpi_client import image_tools
import math
import PIL
from jaxrl2.data.dataset import concat_recursive
from flax.core import frozen_dict


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def obs_to_img(obs, variant):
    """Convert raw observation to resized image for DSRL actor/critic."""
    if variant.env == 'libero':
        curr_image = obs["agentview_image"][::-1, ::-1]
    elif variant.env == 'aloha_cube':
        curr_image = obs["pixels"]["top"]
    elif variant.env == 'cartpole':
        curr_image = obs["image"]
    else:
        raise NotImplementedError(f"Unknown env: {variant.env}")
    if variant.resize_image > 0: 
        curr_image = np.array(PIL.Image.fromarray(curr_image).resize((variant.resize_image, variant.resize_image)))
    return curr_image


def obs_to_pi_zero_input(obs, variant):
    """Convert raw observation to Pi-0/Pi-0.5 input format."""
    if variant.env == 'libero':
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        )
        
        obs_pi_zero = {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            ),
            "prompt": str(variant.task_description),
        }
    elif variant.env == 'aloha_cube':
        img = np.ascontiguousarray(obs["pixels"]["top"])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        obs_pi_zero = {
            "state": obs["agent_pos"],
            "images": {"cam_high": np.transpose(img, (2, 0, 1))}
        }
    elif variant.env == 'cartpole':
        # CartPole uses ZeroBasePolicy, so this is never actually called
        # for inference, but we define it for consistency
        obs_pi_zero = {
            "state": obs["state"],
            "image": obs["image"],
        }
    else:
        raise NotImplementedError(f"Unknown env: {variant.env}")
    return obs_pi_zero


def obs_to_qpos(obs, variant):
    """Extract qpos (proprioceptive state) from observation."""
    if variant.env == 'libero':
        qpos = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )
    elif variant.env == 'aloha_cube':
        qpos = obs["agent_pos"]
    elif variant.env == 'cartpole':
        qpos = obs["state"]
    else:
        raise NotImplementedError(f"Unknown env: {variant.env}")
    return qpos


def _sample_actor_batch(
    replay_buffer, success_replay_buffer,
    batch_size, success_buffer_ratio,
    success_buffer_min_size, use_success_buffer,
    shard_fn=None,
):
    """Sample a batch for actor updates, optionally mixing in success buffer data.
    
    If success buffer is enabled and has enough data, composes the batch as:
        - (1 - success_buffer_ratio) * batch_size samples from main buffer
        - success_buffer_ratio * batch_size samples from success buffer
    Otherwise, samples entirely from the main buffer.
    
    Args:
        replay_buffer: Main replay buffer (all data).
        success_replay_buffer: Success-only replay buffer (may be None).
        batch_size: Total batch size.
        success_buffer_ratio: Fraction of batch from success buffer (e.g., 0.2).
        success_buffer_min_size: Minimum samples in success buffer before using it.
        use_success_buffer: Whether success buffer mixing is enabled.
        shard_fn: Optional function to shard batch across devices.
        
    Returns:
        FrozenDict batch for actor update.
    """
    # Check if success buffer is ready
    success_buffer_ready = (
        use_success_buffer
        and success_replay_buffer is not None
        and len(success_replay_buffer) >= success_buffer_min_size
    )
    
    if success_buffer_ready:
        success_batch_size = max(1, int(batch_size * success_buffer_ratio))
        main_batch_size = batch_size - success_batch_size
        
        main_batch = replay_buffer.sample(main_batch_size)
        success_batch = success_replay_buffer.sample(success_batch_size)
        
        # Concatenate: concat_recursive handles nested dicts
        mixed = concat_recursive([main_batch, success_batch])
        batch = frozen_dict.freeze(mixed)
    else:
        batch = replay_buffer.sample(batch_size)
    
    if shard_fn is not None:
        batch = shard_fn(batch)
    
    return batch


def trajwise_alternating_training_loop_residual(
    variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
    perform_control_evals=True, shard_fn=None, agent_dp=None, success_replay_buffer=None
):
    """Main training loop for Residual SAC.
    
    Args:
        variant: Training configuration.
        agent: Residual SAC agent (PixelSACResidualLearner).
        env: Training environment.
        eval_env: Evaluation environment.
        online_replay_buffer: Replay buffer for online data.
        replay_buffer: Main replay buffer (same as online for pure online RL).
        wandb_logger: WandB logger.
        perform_control_evals: Whether to run policy evaluations.
        shard_fn: Function to shard batches across devices.
        agent_dp: Frozen base policy (Pi-0.5 / Pi-0).
        success_replay_buffer: Optional separate buffer for successful trajectories.
    """
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    # Success buffer configuration
    success_buffer_ratio = variant.get('success_buffer_ratio', 0.0)
    success_buffer_min_size = variant.get('success_buffer_min_size', 100)
    use_success_buffer = (success_replay_buffer is not None and success_buffer_ratio > 0.0)
    
    if use_success_buffer:
        # Compute split sizes for mixed actor batches
        success_batch_size = max(1, int(variant.batch_size * success_buffer_ratio))
        main_batch_size = variant.batch_size - success_batch_size
        print(f'[Success Buffer] Enabled: ratio={success_buffer_ratio}, '
              f'main_batch={main_batch_size}, success_batch={success_batch_size}, '
              f'min_size={success_buffer_min_size}')

    total_env_steps = 0
    i = 0
    on_policy_ppo = variant.get('on_policy_ppo', False)
    
    # BC warmup configuration
    bc_warmup_steps = variant.get('bc_warmup_steps', 0)
    bc_warmup_num_critic_updates = variant.get('bc_warmup_num_critic_updates', 10)
    bc_warmup_num_actor_updates = variant.get('bc_warmup_num_actor_updates', 1)
    if bc_warmup_steps > 0:
        print(f'[BC Warmup] Enabled: warmup_steps={bc_warmup_steps}, '
              f'critic_updates={bc_warmup_num_critic_updates}, '
              f'actor_updates={bc_warmup_num_actor_updates}')
    
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)
    
    with tqdm(total=variant.max_steps, initial=0) as pbar:
        while i <= variant.max_steps:
            traj = collect_traj_residual(variant, agent, env, i, agent_dp)
            traj_id = online_replay_buffer._traj_counter
            add_online_data_to_buffer_residual(variant, traj, online_replay_buffer, success_replay_buffer)
            total_env_steps += traj['env_steps']
            print('online buffer timesteps length:', len(online_replay_buffer))
            print('online buffer num traj:', traj_id + 1)
            if success_replay_buffer is not None:
                print('success buffer timesteps length:', len(success_replay_buffer))
            print('total env steps:', total_env_steps)
            
            if variant.get("num_online_gradsteps_batch", -1) > 0:
                num_gradsteps = variant.num_online_gradsteps_batch
            else:
                num_gradsteps = len(traj["rewards"]) * variant.multi_grad_step

            # UTD ratios: num_critic_updates and num_actor_updates per collected step
            num_critic_updates = getattr(variant, 'num_critic_updates', 1)
            num_actor_updates = getattr(variant, 'num_actor_updates', 1)

            if len(online_replay_buffer) > variant.start_online_updates:
                # Perform first visualization before updating
                if i == 0:
                    print('Performing evaluation for initial checkpoint (residual PPO)')
                    if perform_control_evals:
                        perform_control_eval_residual(agent, eval_env, i, variant, wandb_logger, agent_dp)
                    if hasattr(agent, 'perform_eval'):
                        agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                for _ in tqdm(range(num_gradsteps), desc='gradsteps', leave=False):
                    # Determine if we're in BC warmup phase
                    in_bc_warmup = (bc_warmup_steps > 0 and i < bc_warmup_steps)
                    
                    # Select update ratios based on phase
                    if in_bc_warmup:
                        curr_num_critic_updates = bc_warmup_num_critic_updates
                        curr_num_actor_updates = bc_warmup_num_actor_updates
                    else:
                        curr_num_critic_updates = num_critic_updates
                        curr_num_actor_updates = num_actor_updates
                    
                    # Critic updates: always TD learning (aggressive during warmup)
                    critic_info = {}
                    for _ in range(curr_num_critic_updates):
                        batch = next(replay_buffer_iterator)
                        critic_info = agent.update_critic(batch)

                    # Actor updates: BC during warmup, RL after
                    actor_info = {}
                    if in_bc_warmup:
                        # BC warmup: distill base policy actions via MSE
                        for _ in range(curr_num_actor_updates):
                            actor_batch = next(replay_buffer_iterator)
                            if shard_fn is not None:
                                actor_batch = shard_fn(actor_batch)
                            actor_info = agent.update_actor_bc(actor_batch)
                    elif on_policy_ppo:
                        # On-policy PPO: sample from last trajectory with stored log_probs
                        for _ in range(curr_num_actor_updates):
                            actor_batch = online_replay_buffer.sample_from_last_traj(variant.batch_size)
                            actor_batch = jax.device_put(actor_batch)
                            if shard_fn is not None:
                                actor_batch = shard_fn(actor_batch)
                            actor_info = agent.update_actor_onpolicy(actor_batch)
                    else:
                        # Off-policy: sample from full buffer (original GRPO/QPG path)
                        for _ in range(curr_num_actor_updates):
                            actor_batch = _sample_actor_batch(
                                replay_buffer, success_replay_buffer,
                                variant.batch_size, success_buffer_ratio,
                                success_buffer_min_size, use_success_buffer,
                                shard_fn,
                            )
                            actor_info = agent.update_actor(actor_batch)

                    # Combine info for logging
                    update_info = {**critic_info, **actor_info}
                    update_info['residual/alpha'] = float(agent._residual_alpha)
                    update_info['algo'] = agent.algo
                    update_info['bc_warmup/is_warmup'] = 1.0 if in_bc_warmup else 0.0
                    
                    # Log phase transition
                    if bc_warmup_steps > 0 and i == bc_warmup_steps:
                        print(f'\n{"="*60}')
                        print(f'[BC Warmup -> RL] Transitioning at step {i}')
                        print(f'{"="*60}\n')

                    pbar.update()
                    i += 1

                    if i % variant.log_interval == 0:
                        update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                        for k, v in update_info.items():
                            if hasattr(v, 'ndim'):
                                if v.ndim == 0:
                                    wandb_logger.log({f'training/{k}': v}, step=i)
                                elif v.ndim <= 2:
                                    wandb_logger.log_histogram(f'training/{k}', v, i)
                            else:
                                # Scalar value (e.g., residual_alpha)
                                wandb_logger.log({f'training/{k}': v}, step=i)
                        
                        wandb_logger.log({
                            'replay_buffer_size': len(online_replay_buffer),
                            'success_buffer_size': len(success_replay_buffer) if success_replay_buffer is not None else 0,
                            'success_buffer_active': float(
                                use_success_buffer 
                                and success_replay_buffer is not None 
                                and len(success_replay_buffer) >= success_buffer_min_size
                            ),
                            'episode_return (exploration)': traj['episode_return'],
                            'is_success (exploration)': int(traj['is_success']),
                        }, i)

                    if i % variant.eval_interval == 0:
                        wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                        wandb_logger.log({'num_online_trajs': traj_id + 1}, step=i)
                        wandb_logger.log({'env_steps': total_env_steps}, step=i)
                        if perform_control_evals:
                            perform_control_eval_residual(agent, eval_env, i, variant, wandb_logger, agent_dp)
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    if variant.checkpoint_interval != -1 and i % variant.checkpoint_interval == 0:
                        agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)


def add_online_data_to_buffer_residual(variant, traj, online_replay_buffer, success_replay_buffer=None):
    """Add collected trajectory to replay buffer for Residual SAC.
    
    Stores:
        - observations with 'base_action' (chunk from frozen policy)
        - actions = delta_actions (residual) or a_exec (if predict_a_exec=True)
        - rewards, masks, discount
        - success_flag: 1.0 if this episode was successful, 0.0 otherwise
        - old_log_probs: log probability of the action under the behavior policy
    
    If success_replay_buffer is provided and the trajectory was successful,
    transitions are also added to the success buffer.
    """
    discount_horizon = variant.query_freq
    actions = np.array(traj['actions'])  # (B, query_freq, action_dim) - delta or a_exec depending on predict_a_exec
    base_actions = np.array(traj['base_actions'])  # (B, chunk_len, action_dim)
    episode_len = len(actions)
    rewards = np.array(traj['rewards'])
    masks = np.array(traj['masks'])
    is_success = float(traj['is_success'])
    old_log_probs = np.array(traj.get('old_log_probs', np.zeros(episode_len)))

    for t in range(episode_len):
        obs = traj['observations'][t]
        next_obs = traj['observations'][t + 1]
        
        # Remove batch dimension
        obs = {k: v[0] for k, v in obs.items()}
        next_obs = {k: v[0] for k, v in next_obs.items()}
        
        # Add base_action to observations (with trailing dimension)
        obs['base_action'] = base_actions[t][..., np.newaxis]  # (chunk_len, action_dim, 1)
        if t < episode_len - 1:
            next_obs['base_action'] = base_actions[t + 1][..., np.newaxis]
        else:
            # Last step: use same base_action (will be masked anyway)
            next_obs['base_action'] = base_actions[t][..., np.newaxis]
        
        if not variant.add_states:
            obs.pop('state', None)
            next_obs.pop('state', None)
        
        insert_dict = dict(
            observations=obs,
            next_observations=next_obs,
            actions=actions[t],  # delta_action or a_exec (depending on predict_a_exec)
            next_actions=actions[t + 1] if t < episode_len - 1 else actions[t],
            rewards=rewards[t],
            masks=masks[t],
            discount=variant.discount ** discount_horizon,
            success_flag=is_success,
            old_log_probs=old_log_probs[t],
        )
        online_replay_buffer.insert(insert_dict)
        
        # Also insert into success buffer if trajectory was successful
        if success_replay_buffer is not None and is_success > 0.5:
            success_replay_buffer.insert(insert_dict)
    
    online_replay_buffer.increment_traj_counter()
    if success_replay_buffer is not None and is_success > 0.5:
        success_replay_buffer.increment_traj_counter()


def collect_traj_residual(variant, agent, env, i, agent_dp=None):
    """Collect a trajectory using Residual SAC.
    
    At each query step:
    1. Query frozen Pi-0.5 (with internal random noise) -> base_actions
    2. Build SAC observation with base_action
    3. SAC samples delta_actions (residual)
    4. Execute: a_exec = clip(base_actions + alpha * delta_actions, -1, 1)
    
    Args:
        variant: Training configuration (must have residual_alpha).
        agent: Residual SAC agent.
        env: Environment.
        i: Current training step (for exploration strategy).
        agent_dp: Frozen base policy (Pi-0.5).
        
    Returns:
        Trajectory dict with observations, base_actions, actions (delta), rewards, etc.
        If on_policy_ppo=True, also includes old_log_probs.
    """
    query_frequency = variant.query_freq
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    residual_alpha =  float(agent._residual_alpha)
    chunk_len = variant.chunk_len  # e.g., 10 for Pi-0.5
    on_policy_ppo = variant.get('on_policy_ppo', False)
    predict_a_exec = variant.get('predict_a_exec', False)
    
    # Flag to control initial exploration behavior
    use_zero_residual_initially = variant.get('use_zero_residual_initially', True)
    
    # BC warmup: force zero residual for ALL trajectories during warmup
    bc_warmup_steps = variant.get('bc_warmup_steps', 0)
    in_bc_warmup = (bc_warmup_steps > 0 and i < bc_warmup_steps)
    force_zero_residual = (i == 0 and use_zero_residual_initially) or in_bc_warmup

    agent._rng, rng = jax.random.split(agent._rng)
    
    if 'libero' in variant.env:
        obs = env.reset()
    elif 'aloha' in variant.env:
        obs, _ = env.reset()
    elif variant.env == 'cartpole':
        obs = env.reset()
    
    image_list = []  # for visualization
    rewards = []
    action_list = []  # delta actions
    base_action_list = []  # base actions from Pi-0.5
    obs_list = []
    old_log_probs_list = []  # log probs from behavior policy (for on-policy PPO)

    for t in tqdm(range(max_timesteps)):
        curr_image = obs_to_img(obs, variant)
        qpos = obs_to_qpos(obs, variant)

        if t % query_frequency == 0:
            assert agent_dp is not None, "Frozen base policy (agent_dp) is required for residual SAC"
            
            # 1. Query frozen Pi-0.5 (uses internal random noise)
            obs_pi_zero = obs_to_pi_zero_input(obs, variant)
            base_actions = agent_dp.infer(obs_pi_zero)["actions"][:chunk_len]  # (chunk_len, action_dim)
            
            # 2. Build SAC observation with base_action
            if variant.add_states:
                obs_dict = {
                    'pixels': curr_image[np.newaxis, ..., np.newaxis],
                    'state': qpos[np.newaxis, ..., np.newaxis],
                    'base_action': base_actions[np.newaxis, ..., np.newaxis],  # (1, chunk_len, action_dim, 1)
                }
            else:
                obs_dict = {
                    'pixels': curr_image[np.newaxis, ..., np.newaxis],
                    'base_action': base_actions[np.newaxis, ..., np.newaxis],
                }
            
            # 3. Sample actions from SAC
            rng, key = jax.random.split(rng)
            if force_zero_residual:
                # Zero residual: evaluate base policy (used for first traj or BC warmup)
                delta_actions = np.zeros((query_frequency, variant.action_dim))
                
                if in_bc_warmup:
                    # During BC warmup: compute log_prob of the zero-residual action
                    # so that old_log_probs are stored consistently for all transitions.
                    if predict_a_exec:
                        eval_actions_flat = np.clip(base_actions[:query_frequency], -1.0, 1.0).reshape(1, -1)
                    else:
                        eval_actions_flat = delta_actions.reshape(1, -1)
                    log_prob = agent.compute_log_prob(obs_dict, eval_actions_flat)
                    log_prob = float(np.squeeze(log_prob))
                    if t == 0:
                        print(f"[BC warmup] Forcing zero residual (log_prob={log_prob:.4f})")
                else:
                    # First trajectory (initial eval): compute log_prob for PPO correctness
                    if predict_a_exec:
                        eval_actions_flat = np.clip(base_actions[:query_frequency], -1.0, 1.0).reshape(1, -1)
                    else:
                        eval_actions_flat = delta_actions.reshape(1, -1)
                    # Use compute_log_prob which safely clamps to avoid atanh(±1)
                    log_prob = agent.compute_log_prob(obs_dict, eval_actions_flat)
                    log_prob = float(np.squeeze(log_prob))
                    if t == 0:
                        print(f"[t={t}] Using zero residual (initial eval) (log_prob={log_prob:.4f})")
                
                # Compose executed action (base only)
                actions = np.clip(base_actions[:query_frequency], -1.0, 1.0)
            else:
                # Use deterministic (mode) actions for rollouts.
                # Exploration comes from the stochastic base policy, not from
                # sampling noise in the residual — sampling adds jitter.
                if on_policy_ppo:
                    # On-policy PPO: use mode action but compute its log_prob
                    # under current policy for importance weighting.
                    actions_flat = agent.eval_actions(obs_dict)  # mode (deterministic)
                    # Compute log_prob of the mode action
                    log_prob = agent.compute_log_prob(obs_dict, actions_flat)
                    log_prob = float(np.squeeze(log_prob))
                else:
                    actions_flat = agent.eval_actions(obs_dict)  # mode (deterministic)
                    log_prob = 0.0  # not needed for off-policy
                raw_actions = np.reshape(actions_flat, (query_frequency, variant.action_dim))
                
                # NaN guard: if action contains NaN/Inf, replace with zeros
                if not np.all(np.isfinite(raw_actions)):
                    print(f"[WARNING] NaN/Inf detected in actions at t={t}, replacing with zeros")
                    raw_actions = np.nan_to_num(raw_actions, nan=0.0, posinf=0.0, neginf=0.0)
                    log_prob = 0.0  # invalidated by NaN replacement
                
                if predict_a_exec:
                    # Actor predicts a_exec directly
                    actions = np.clip(raw_actions, -1.0, 1.0)
                    delta_actions = raw_actions  # store raw actor output (which IS a_exec)
                else:
                    # Original: actor predicts delta, compose a_exec
                    delta_actions = raw_actions
                    actions = np.clip(base_actions[:query_frequency] + residual_alpha * delta_actions, -1.0, 1.0)
            
            # Store for replay buffer
            if predict_a_exec:
                action_list.append(actions)  # Store a_exec
            else:
                action_list.append(delta_actions)  # Store residual
            base_action_list.append(base_actions)  # Store base action
            obs_list.append(obs_dict)
            old_log_probs_list.append(log_prob)  # Store log_prob from behavior policy
            
            # Log residual stats occasionally
            if t == 0:
                if predict_a_exec:
                    delta_from_base = actions - base_actions[:query_frequency]
                    delta_norm = np.linalg.norm(delta_from_base)
                else:
                    delta_norm = np.linalg.norm(delta_actions)
                base_norm = np.linalg.norm(base_actions)
                print(f"[t={t}] base_norm={base_norm:.4f}, delta_norm={delta_norm:.4f}, alpha={residual_alpha}, predict_a_exec={predict_a_exec}")
     
        action_t = actions[t % query_frequency]
        
        if 'libero' in variant.env:
            obs, reward, done, _ = env.step(action_t)
        elif 'aloha' in variant.env:
            obs, reward, terminated, truncated, _ = env.step(action_t)
            done = terminated or truncated
        elif variant.env == 'cartpole':
            obs, reward, done, info = env.step(action_t)
            
        rewards.append(reward)
        image_list.append(curr_image)
        if done:
            break

    # Add last observation
    curr_image = obs_to_img(obs, variant)
    qpos = obs_to_qpos(obs, variant)
    
    # For last obs, we need a base_action - use the last one (will be masked)
    last_base_action = base_action_list[-1] if base_action_list else np.zeros((chunk_len, variant.action_dim))
    obs_dict = {
        'pixels': curr_image[np.newaxis, ..., np.newaxis],
        'state': qpos[np.newaxis, ..., np.newaxis],
        'base_action': last_base_action[np.newaxis, ..., np.newaxis],
    }
    if not variant.add_states:
        obs_dict.pop('state', None)
    obs_list.append(obs_dict)
    image_list.append(curr_image)
    
    # Per episode stats
    rewards = np.array(rewards)
    episode_return = np.sum(rewards[rewards != None])
    if variant.env == 'cartpole':
        # CartPole: success if most recent info shows success (pole was upright)
        is_success = bool(info.get('success', 0))
    else:
        is_success = (reward == env_max_reward)
    print(f'Rollout Done: {episode_return=}, Success: {is_success}')
    
    reward_type = variant.get('reward_type', 'sparse')
    query_steps = len(action_list)
    if reward_type == 'dense':
        # Keep raw environment rewards; set terminal mask on done
        rewards = np.array(rewards, dtype=np.float32)
        if is_success:
            masks = np.concatenate([np.ones(query_steps - 1), [0]])
        else:
            masks = np.ones(query_steps)
    else:
        # Sparse -1/0 reward for SAC training
        if is_success:
            rewards = np.concatenate([-np.ones(query_steps - 1), [0]])
            masks = np.concatenate([np.ones(query_steps - 1), [0]])
        else:
            rewards = -np.ones(query_steps)
            masks = np.ones(query_steps)

    return {
        'observations': obs_list,
        'actions': action_list,  # delta actions (if predict_a_exec=False) or a_exec (if predict_a_exec=True)
        'base_actions': base_action_list,  # base actions from Pi-0.5
        'rewards': rewards,
        'masks': masks,
        'is_success': is_success,
        'episode_return': episode_return,
        'images': image_list,
        'env_steps': t + 1,
        'old_log_probs': old_log_probs_list,  # log probs from behavior policy
    }


def perform_control_eval_residual(agent, env, i, variant, wandb_logger, agent_dp=None):
    """Evaluate Residual SAC policy.
    
    Uses the same logic as collect_traj_residual but without exploration noise.
    """
    query_frequency = variant.query_freq
    print(f'[Eval] query frequency: {query_frequency}')
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    residual_alpha =  float(agent._residual_alpha)
    chunk_len = variant.chunk_len
    predict_a_exec = variant.get('predict_a_exec', False)
    
    episode_returns = []
    highest_rewards = []
    success_rates = []
    episode_lens = []
    
    # Track residual statistics across evaluation
    all_delta_norms = []
    all_base_norms = []
    all_clipping_rates = []

    rng = jax.random.PRNGKey(variant.seed + 456)

    for rollout_id in range(variant.eval_episodes):
        if 'libero' in variant.env:
            obs = env.reset()
        elif 'aloha' in variant.env:
            obs, _ = env.reset()
        elif variant.env == 'cartpole':
            obs = env.reset()
            
        image_list = []
        rewards = []
        
        for t in tqdm(range(max_timesteps)):
            curr_image = obs_to_img(obs, variant)

            if t % query_frequency == 0:
                qpos = obs_to_qpos(obs, variant)
                
                assert agent_dp is not None
                
                # 1. Query frozen Pi-0.5
                obs_pi_zero = obs_to_pi_zero_input(obs, variant)
                base_actions = agent_dp.infer(obs_pi_zero)["actions"][:chunk_len]
                
                # 2. Build SAC observation
                if variant.add_states:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                        'state': qpos[np.newaxis, ..., np.newaxis],
                        'base_action': base_actions[np.newaxis, ..., np.newaxis],
                    }
                else:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                        'base_action': base_actions[np.newaxis, ..., np.newaxis],
                    }
                
                rng, key = jax.random.split(rng)
                
                if i == 0:
                    # Initial evaluation: zero residual to test base policy
                    delta_actions = np.zeros((query_frequency, variant.action_dim))
                    actions = np.clip(base_actions[:query_frequency], -1.0, 1.0)
                else:
                    # SAC samples residual (deterministic: use mean)
                    actions_flat = agent.eval_actions(obs_dict)  # Use eval_actions for deterministic
                    raw_actions = np.reshape(actions_flat, (query_frequency, variant.action_dim))
                    
                    # NaN guard: if action contains NaN/Inf, replace with zeros
                    if not np.all(np.isfinite(raw_actions)):
                        print(f"[WARNING] NaN/Inf detected in eval actions at t={t}, replacing with zeros")
                        raw_actions = np.nan_to_num(raw_actions, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    if predict_a_exec:
                        # Actor predicts a_exec directly
                        actions = np.clip(raw_actions, -1.0, 1.0)
                        delta_actions = actions - base_actions[:query_frequency]
                    else:
                        # Original: actor predicts delta, compose a_exec
                        delta_actions = raw_actions
                        actions = np.clip(base_actions[:query_frequency] + residual_alpha * delta_actions, -1.0, 1.0)
                
                # 3. Compose executed action (already done above)
                
                # Track statistics
                delta_norm = np.linalg.norm(delta_actions.flatten())
                base_norm = np.linalg.norm(base_actions.flatten())
                clipping_rate = np.mean(np.abs(actions) > 0.999)
                all_delta_norms.append(delta_norm)
                all_base_norms.append(base_norm)
                all_clipping_rates.append(clipping_rate)
              
            action_t = actions[t % query_frequency]
            
            if 'libero' in variant.env:
                obs, reward, done, _ = env.step(action_t)
            elif 'aloha' in variant.env:
                obs, reward, terminated, truncated, _ = env.step(action_t)
                done = terminated or truncated
            elif variant.env == 'cartpole':
                obs, reward, done, eval_info = env.step(action_t)
                
            rewards.append(reward)
            image_list.append(curr_image)
            if done:
                break

        # Per episode stats
        episode_lens.append(t + 1)
        rewards = np.array(rewards)
        episode_return = np.sum(rewards)
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        if variant.env == 'cartpole':
            is_success = bool(eval_info.get('success', 0))
        else:
            is_success = (reward == env_max_reward)
        success_rates.append(is_success)
                
        print(f'Rollout {rollout_id}: {episode_return=}, Success: {is_success}')
        video = np.stack(image_list).transpose(0, 3, 1, 2)
        wandb_logger.log({f'eval_video/{rollout_id}': wandb.Video(video, fps=50)}, step=i)

    # Log aggregate statistics
    success_rate = np.mean(np.array(success_rates))
    avg_return = np.mean(episode_returns)
    avg_episode_len = np.mean(episode_lens)
    
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    wandb_logger.log({'evaluation/avg_return': avg_return}, step=i)
    wandb_logger.log({'evaluation/success_rate': success_rate}, step=i)
    wandb_logger.log({'evaluation/avg_episode_len': avg_episode_len}, step=i)
    
    # Log residual-specific evaluation metrics
    wandb_logger.log({'evaluation/delta_norm_mean': np.mean(all_delta_norms)}, step=i)
    wandb_logger.log({'evaluation/delta_norm_std': np.std(all_delta_norms)}, step=i)
    wandb_logger.log({'evaluation/base_norm_mean': np.mean(all_base_norms)}, step=i)
    wandb_logger.log({'evaluation/clipping_rate_mean': np.mean(all_clipping_rates)}, step=i)
    
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / variant.eval_episodes
        wandb_logger.log({f'evaluation/Reward >= {r}': more_or_equal_r_rate}, step=i)
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{variant.eval_episodes} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)


def make_multiple_value_reward_visulizations_residual(agent, variant, i, replay_buffer, wandb_logger):
    """Create value/reward visualizations for random trajectories."""
    trajs = replay_buffer.get_random_trajs(3)
    images = agent.make_value_reward_visulization(variant, trajs)
    wandb_logger.log({'reward_value_images': wandb.Image(images)}, step=i)
