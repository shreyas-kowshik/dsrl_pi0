from typing import Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks import MLP
from jaxrl2.networks.constants import default_init

class LearnedStdNormalPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)

        log_stds = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))
        return distribution

class TanhMultivariateNormalDiag(distrax.Transformed):
    def __init__(self,
                 loc: jnp.ndarray,
                 scale_diag: jnp.ndarray,
                 low: Optional[jnp.ndarray] = None,
                 high: Optional[jnp.ndarray] = None):
        
        # Base distribution: Gaussian
        distribution = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        layers = []

        # 2. Rescale: [-1, 1] -> [low, high]
        # This is mathematically identical to your manual rescale_from_tanh
        if (low is not None) and (high is not None):
            # Transformation: y = x * scale + shift
            # Equivalent to: (x + 1) / 2 * (high - low) + low
            scale = (high - low) / 2.0
            shift = (high + low) / 2.0
            
            # ScalarAffine provides the inverse and log_det_jacobian automatically
            rescale_bijector = distrax.ScalarAffine(shift=shift, scale=scale)
            layers.append(distrax.Block(rescale_bijector, 1))

        # 1. Squash: Gaussian -> [-1, 1]
        layers.append(distrax.Block(distrax.Tanh(), 1))

        # distrax.Chain applies the list in REVERSE order (Last element first).
        # Execution flow: Sample -> Tanh (index 1) -> Rescale (index 0) -> Action
        bijector = distrax.Chain(layers)

        super().__init__(distribution=distribution, bijector=bijector)

    def mode(self) -> jnp.ndarray:
        # Calculates the greedy action: Rescale(Tanh(Mean))
        return self.bijector.forward(self.distribution.mode())

class LearnedStdTanhNormalPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    low: Optional[float] = None
    high: Optional[float] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)

        log_stds = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = TanhMultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds), low=self.low, high=self.high)
        return distribution