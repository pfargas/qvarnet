from collections.abc import Callable

import jax
from flax import linen as nn
from jax import numpy as jnp

from qvarnet.ansatz.mlp import MLP


class MessagePassingGNN(nn.Module):
    """Permutation-invariant log-wavefunction from message passing on the complete graph.

    Every particle i is a node, every pair (i, j) an edge with feature
    e_ij = (r_i - r_j, |r_i - r_j|²). Embed, then ``n_layers`` rounds, then read out:

        h_i = E(r_i)
        m_i = mean_{j≠i} M([h_i, h_j, e_ij]),    h_i ← h_i + U([h_i, m_i])
        log|ψ| = R(mean_i h_i)

    Each round has its own M and U. ``*_hidden`` are the hidden widths of each network.

    Dimension contract:
        Input:  (..., n_particles, per_particle_dim)  — pre-reshaped by LogWavefunction
        Output: (..., 1)  — log|ψ|, no envelope
    """

    hidden_dim: int = 16
    n_layers: int = 2
    embed_hidden: list = (32,)
    message_hidden: list = (32,)
    update_hidden: list = (32,)
    readout_hidden: list = (32,)
    activation: Callable = nn.tanh
    kernel_init: Callable = nn.initializers.lecun_normal()

    def _mlp(self, hidden, out):
        return MLP(hidden=list(hidden), output_dim=out, hidden_activation=self.activation,
                   kernel_init=self.kernel_init)

    def setup(self):
        self.embed = self._mlp(self.embed_hidden, self.hidden_dim)
        self.messages = [self._mlp(self.message_hidden, self.hidden_dim) for _ in range(self.n_layers)]
        self.node_updates = [self._mlp(self.update_hidden, self.hidden_dim) for _ in range(self.n_layers)]
        self.readout = self._mlp(self.readout_hidden, 1)

    def edges(self, x):
        """(..., N, d) → edge features (..., N, N, d+1)."""
        diff = x[..., :, None, :] - x[..., None, :, :]  # (..., N, N, d)
        return jnp.concatenate([diff, jnp.sum(diff**2, axis=-1, keepdims=True)], axis=-1)

    def message_pass(self, h, e, layer):
        """One round: h (..., N, H), e (..., N, N, d+1) → new h (..., N, H)."""
        n = h.shape[-2]
        h_i = jnp.broadcast_to(h[..., :, None, :], (*h.shape[:-1], n, self.hidden_dim))
        h_j = jnp.broadcast_to(h[..., None, :, :], h_i.shape)
        msg = self.messages[layer](jnp.concatenate([h_i, h_j, e], axis=-1))  # (..., N, N, H)
        not_self = jnp.broadcast_to((1.0 - jnp.eye(n))[:, :, None], msg.shape)
        m = jnp.sum(msg * not_self, axis=-2) / (n - 1)  # (..., N, H)
        return h + self.node_updates[layer](jnp.concatenate([h, m], axis=-1))

    def __call__(self, x):
        e = self.edges(x)
        h = self.embed(x)  # (..., N, H)
        for layer in range(self.n_layers):
            h = self.message_pass(h, e, layer)
        return self.readout(jnp.mean(h, axis=-2))  # (..., 1)


if __name__ == "__main__":
    # self-check: output shape, and invariance under swapping particles
    jax.config.update("jax_numpy_rank_promotion", "raise")
    x = jax.random.normal(jax.random.PRNGKey(0), (7, 5, 1))  # batch 7, N=5, 1D
    net = MessagePassingGNN()
    params = net.init(jax.random.PRNGKey(1), x)
    out = net.apply(params, x)
    assert out.shape == (7, 1), out.shape
    assert jnp.allclose(out, net.apply(params, x[:, ::-1, :]), atol=1e-6)
    print("ok", sorted(params["params"]))
