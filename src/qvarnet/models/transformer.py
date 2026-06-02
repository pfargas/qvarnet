from collections.abc import Callable

import jax.numpy as jnp
from flax import linen as nn

from .base import BaseModel
from .registry import register_model


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: self-attention + position-wise MLP."""

    embed_dim: int
    num_heads: int
    mlp_hidden: tuple
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x):
        # x: (..., n_particles, embed_dim)

        # Self-attention sub-layer
        h = nn.LayerNorm()(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
        )(h, h)
        x = x + h

        # MLP sub-layer
        h = nn.LayerNorm()(x)
        for features in self.mlp_hidden:
            h = nn.Dense(features)(h)
            h = self.activation(h)
        h = nn.Dense(self.embed_dim)(h)
        x = x + h

        return x


@register_model("transformer")
class TransformerWavefunction(BaseModel):
    """Permutation-invariant log-wavefunction via stacked self-attention.

    Particles attend to one another through ``num_layers`` transformer blocks.
    No positional encoding — identical particles are exchangeable.

    Dimension contract (same as DeepSet — set ``n_particles`` in LogWavefunction):
        Input:  (..., n_particles, per_particle_dim)
        Output: (..., 1)   — log|ψ|, no envelope

    Typical config:
        embed_dim=32, num_heads=4, num_layers=2,
        block_mlp_hidden=(64,), output_hidden=(32,)
    """

    embed_dim: int = 32
    num_heads: int = 4
    num_layers: int = 2
    block_mlp_hidden: tuple = (64,)
    output_hidden: tuple = (32,)
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x):
        # x: (..., n_particles, per_particle_dim)

        # Project particles into embedding space
        x = nn.Dense(self.embed_dim)(x)  # (..., n_particles, embed_dim)

        # Stacked transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_hidden=self.block_mlp_hidden,
                activation=self.activation,
            )(x)

        # Final layer norm then mean-pool over particles → permutation invariant
        x = nn.LayerNorm()(x)
        x = jnp.mean(x, axis=-2)  # (..., embed_dim)

        # Output MLP
        for features in self.output_hidden:
            x = nn.Dense(features)(x)
            x = self.activation(x)
        x = nn.Dense(1)(x)  # (..., 1)

        return x

    @classmethod
    def from_config(cls, model_args: dict):
        return cls(
            embed_dim=model_args.get("embed_dim", 32),
            num_heads=model_args.get("num_heads", 4),
            num_layers=model_args.get("num_layers", 2),
            block_mlp_hidden=tuple(model_args.get("block_mlp_hidden", (64,))),
            output_hidden=tuple(model_args.get("output_hidden", (32,))),
        )

    @classmethod
    def get_input_shape(cls, model_args: dict, batch_size: int) -> tuple:
        raise NotImplementedError("TransformerWavefunction input shape depends on LogWavefunction geometry.")
