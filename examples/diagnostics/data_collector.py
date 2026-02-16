"""Data structures for evaluation trajectory data used by diagnostics."""

import dataclasses
from typing import List, Dict, Any
import numpy as np


@dataclasses.dataclass
class EvalTrajectoryData:
    """All per-step data from one evaluation rollout, needed for diagnostics.

    Attributes:
        obs_dicts: Per query-step observation dicts (each with batch dim 1).
            Length is T_query + 1 (includes final obs for bootstrapping).
            Keys: 'pixels', 'base_action', optionally 'state', 'vlm_embedding'.
        base_actions: (T_query, chunk_len, action_dim) base actions from frozen policy.
        delta_actions: (T_query, query_freq, action_dim) raw actor output.
        a_exec: (T_query, query_freq, action_dim) composed+clipped executed actions.
        rewards: (T_steps,) per env-step rewards.
        terminated: Whether the episode ended with a true terminal state.
        truncated: Whether the episode was cut off by time limit.
        is_success: Whether the episode was successful.
        images: Camera frames per env step, list of np.ndarray.
        episode_return: Sum of rewards.
        query_frequency: Number of env steps per query.
    """
    obs_dicts: List[Dict[str, Any]]
    base_actions: np.ndarray
    delta_actions: np.ndarray
    a_exec: np.ndarray
    rewards: np.ndarray
    terminated: bool
    truncated: bool
    is_success: bool
    images: List[np.ndarray]
    episode_return: float
    query_frequency: int
