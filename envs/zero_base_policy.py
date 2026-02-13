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

    def __init__(self, action_dim: int, chunk_len: int = 10, vlm_embedding_dim: int = 2048, vlm_seq_len: int = 16):
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.vlm_embedding_dim = vlm_embedding_dim
        self.vlm_seq_len = vlm_seq_len

    def infer(self, obs: dict, return_vlm_embedding: bool = False) -> dict:
        """Return zero actions in the same format as Pi-0.5.
        
        Args:
            obs: Observation dict (ignored).
            return_vlm_embedding: If True, also return a dummy VLM embedding
                matching the interface of Pi-0.5's ``(hidden_state, kv_cache)`` tuple.
            
        Returns:
            Dict with key "actions" -> np.ndarray of shape (chunk_len, action_dim).
            If return_vlm_embedding, also includes "vlm_embedding" -> (hidden_state, None).
        """
        result = {
            "actions": np.zeros((self.chunk_len, self.action_dim), dtype=np.float32),
        }
        if return_vlm_embedding:
            # Dummy VLM embedding: (hidden_state, kv_cache)
            # hidden_state shape: (1, vlm_seq_len, vlm_embedding_dim)  — batch dim included
            dummy_hidden_state = np.zeros(
                (1, self.vlm_seq_len, self.vlm_embedding_dim), dtype=np.float32
            )
            result["vlm_embedding"] = (dummy_hidden_state, None)
        return result
