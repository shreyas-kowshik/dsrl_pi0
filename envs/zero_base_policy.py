"""Zero-action base policy for testing the residual RL pipeline.

This replaces Pi-0.5 for simple test environments (e.g., CartPole).
It returns zero actions in the same format that the residual pipeline expects:
    {"actions": np.zeros((chunk_len, action_dim))}
"""
import numpy as np


class ZeroBasePolicy:
    """A dummy base policy that always returns zero actions.
    
    Mimics the interface of the Pi-0.5 trained policy so it can be
    used as a drop-in replacement for `agent_dp` in the residual pipeline.
    
    Args:
        action_dim: Dimensionality of the action space.
        chunk_len: Length of the action chunk (default: 10, matching Pi-0.5).
    """

    def __init__(self, action_dim: int, chunk_len: int = 10):
        self.action_dim = action_dim
        self.chunk_len = chunk_len

    def infer(self, obs: dict) -> dict:
        """Return zero actions in the same format as Pi-0.5.
        
        Args:
            obs: Observation dict (ignored).
            
        Returns:
            Dict with key "actions" -> np.ndarray of shape (chunk_len, action_dim).
        """
        return {
            "actions": np.zeros((self.chunk_len, self.action_dim), dtype=np.float32),
        }
