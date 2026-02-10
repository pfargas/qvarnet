import jax.numpy as jnp
import jax.nn
from .layers import CustomDense
from flax import linen as nn
from typing import Callable
from .base import BaseModel

from .registry import register_model


@register_model("fermionic-mlp")
class FermionicMLP(BaseModel):
    architecture: list
    n_fermions: int
    hidden_activation: Callable = nn.tanh
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()
    n_dim: int = 1

    def setup(self):
        # 1. DEFINITION PHASE
        # We create the layers here. They are assigned to 'self',
        # so Flax knows these are the parameters of the model.

        self.hidden_layers = [
            CustomDense(
                features=feat,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name=f"hidden_{i}",  # Optional, but good for debugging
            )
            for i, feat in enumerate(self.architecture)
        ]

        self.output_layer = CustomDense(
            features=self.n_fermions,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="orbital_output",
        )

    def __call__(self, x):
        # x input shape is either:
        #   (Batch, N*Dim) -> e.g., (1000, 3) # in the energy evaluation
        #   (N*Dim,)       -> e.g., (3,) # in the sampler

        # 1. ROBUST RESHAPE
        # We take all leading dimensions (*x.shape[:-1]) as the batch/grouping dims
        # and explicitly split ONLY the last dimension into (N, Dim).
        # This works for shape (1000, 3) -> (1000, N, Dim)
        # AND for shape (3,) -> (N, Dim)

        n_dim = getattr(self, "n_dim", 1)  # Default to 1D if not specified
        h = x.reshape(*x.shape[:-1], self.n_fermions, n_dim)
        # *shape[:-1] captures all leading dimensions,
        # then we split the last dim into (N, Dim) where N is n_fermions and Dim is n_dim.

        # 2. RUN NETWORK
        # CustomDense automatically broadcasts over the extra dimensions.
        for layer in self.hidden_layers:
            h = layer(h)
            h = self.hidden_activation(h)

        # Output shape: (..., N, N)
        # (Batch, Particle i, Orbital j)
        orbitals = self.output_layer(h)

        # 3. DETERMINANT
        # jnp.linalg.det handles the last two dimensions (NxN) correctly
        # regardless of how many batch dimensions precede them.
        psi = jnp.linalg.det(orbitals)

        return psi


@register_model("half-spin-non-interacting-fermion")
class HalfSpinNonInteractingFermion(BaseModel):
    """
    A neural network ansatz for non-interacting fermions with spin.
    It includes an exponential envelope to satisfy boundary conditions
    (essential for atomic systems like Hydrogen).
    """

    architecture: list[int]
    hidden_activation: Callable = nn.tanh
    output_activation: Callable = jax.nn.identity
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()

    # Physics parameters
    n_dim: int = 3  # Default to 3D for atoms
    n_up: int = 1
    n_down: int = 1

    # Initial value for the exponential decay (can be learned)
    init_alpha: float = 1.0

    def setup(self):
        # 1. Total Fermions
        self.total_fermions = self.n_up + self.n_down

        # 2. Trainable Decay Parameter (Alpha)
        # By defining it with self.param, Flax adds it to the parameter collection.
        # We initialize it to 'init_alpha'.
        self.alpha = self.param(
            "alpha", nn.initializers.constant(self.init_alpha), (1,)
        )

        # 3. Hidden Layers (The "Backflow" / Correlation part)
        self.hidden_layers = [
            CustomDense(
                features=feat,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name=f"hidden_{i}",
            )
            for i, feat in enumerate(self.architecture)
        ]

        # 4. Orbital Output Layer
        # We generate enough orbitals for all particles
        self.output_layer = CustomDense(
            features=self.total_fermions,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="orbital_output",
        )

    def __call__(self, x):
        """
        Forward pass of the wavefunction.
        Args:
            x: Input coordinates. Shape (Batch, N_particles * N_dim) or (N_particles * N_dim,)
        Returns:
            psi: The value of the wavefunction (scalar).
        """
        n_dim = getattr(self, "n_dim", 3)

        # -----------------------------------------------------------
        # 1. ROBUST RESHAPE & DISTANCE CALCULATION
        # -----------------------------------------------------------
        # Reshape to (Batch, N_particles, N_dim)
        # This handles both batched (Batch, N*D) and unbatched (N*D) inputs
        h = x.reshape(*x.shape[:-1], self.total_fermions, n_dim)

        # Calculate distance from origin r_i = |x_i| for every particle
        # Shape: (Batch, N_particles)
        r = jnp.linalg.norm(h, axis=-1)

        # Sum distances for the envelope: R = sum(r_i)
        total_r = jnp.sum(r, axis=-1)

        # -----------------------------------------------------------
        # 2. ENVELOPE (The Physics Fix)
        # -----------------------------------------------------------
        # Psi_envelope = exp(-alpha * sum(r_i))
        # We use softplus to ensure alpha stays positive during training
        # (Negative alpha would make the wavefunction explode at infinity)
        envelope = jnp.exp(-nn.softplus(self.alpha) * total_r)

        # -----------------------------------------------------------
        # 3. SPLIT STREAMS (Spin Up / Spin Down)
        # -----------------------------------------------------------
        # Spin UP gets the first n_up particles
        h_up = h[..., : self.n_up, :]
        # Spin DOWN gets the remaining particles
        h_down = h[..., self.n_up :, :]

        # -----------------------------------------------------------
        # 4. NEURAL NETWORK PASS
        # -----------------------------------------------------------
        # Apply layers. Note: h_up and h_down pass through the SAME weights.
        for layer in self.hidden_layers:
            h_up = self.hidden_activation(layer(h_up))
            h_down = self.hidden_activation(layer(h_down))

        # Project to Orbitals
        # We slice columns [..., :n_up] to form a square matrix (N_up x N_up)
        # We slice columns [..., :n_down] because down spins also occupy lowest orbitals
        orbitals_up = self.output_layer(h_up)[..., : self.n_up]
        orbitals_down = self.output_layer(h_down)[..., : self.n_down]

        # -----------------------------------------------------------
        # 5. DETERMINANTS (Slater)
        # -----------------------------------------------------------
        # Handle cases where a spin sector is empty (e.g. Hydrogen n_down=0)
        if self.n_up > 0:
            psi_up = jnp.linalg.det(orbitals_up)
        else:
            psi_up = 1.0

        if self.n_down > 0:
            psi_down = jnp.linalg.det(orbitals_down)
        else:
            psi_down = 1.0

        # -----------------------------------------------------------
        # 6. COMBINE
        # -----------------------------------------------------------
        # Psi_total = Det(Up) * Det(Down) * Envelope
        # Squeeze ensures we return a scalar per batch element
        return (psi_up * psi_down * envelope).squeeze()


@register_model("fermionic-mlp-2")
class FermionicMLP2ferms(BaseModel):
    architecture: list
    n_fermions: int = 2
    hidden_activation: Callable = nn.tanh
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()

    def setup(self):
        # 1. DEFINITION PHASE
        # We create the layers here. They are assigned to 'self',
        # so Flax knows these are the parameters of the model.

        self.hidden_layers = [
            CustomDense(
                features=feat,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name=f"hidden_{i}",  # Optional, but good for debugging
            )
            for i, feat in enumerate(self.architecture)
        ]

        self.output_layer = CustomDense(
            features=self.n_fermions,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="orbital_output",
        )

    def __call__(self, x):
        # 2. COMPUTATION PHASE
        # We just use the layers we defined above.

        def orbital_net(pos):
            y = pos
            for layer in self.hidden_layers:
                y = layer(y)
                y = self.hidden_activation(y)
            y = self.output_layer(y)
            return y

        # Apply the exact same network functions to both inputs
        # (This is mathematically guaranteed to use shared weights)
        orb_1 = orbital_net(x[..., :1])
        orb_2 = orbital_net(x[..., 1:])

        # Determinant Logic
        phi1_A = orb_1[..., 0]
        phi1_B = orb_1[..., 1]

        phi2_A = orb_2[..., 0]
        phi2_B = orb_2[..., 1]

        return phi1_A * phi2_B - phi1_B * phi2_A
