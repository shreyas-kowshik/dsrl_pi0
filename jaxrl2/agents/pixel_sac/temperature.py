import flax.linen as nn
import jax.numpy as jnp


class Temperature(nn.Module):
    initial_temperature: float = 1.0
    min_temp: float = 0.01
    max_temp: float = 2.00
    clip_temp: bool = True

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), jnp.log(self.initial_temperature)))
        temp = jnp.exp(log_temp)
        if self.clip_temp:
            temp = jnp.clip(temp, self.min_temp, self.max_temp)
        return temp