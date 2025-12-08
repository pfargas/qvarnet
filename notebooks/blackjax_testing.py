import time
import numpy as np
import jax
import jax.numpy as jnp
import blackjax

import jax
print("JIT disabled? ", jax.config.jax_disable_jit)


# synthetic data
observed = jnp.asarray(np.random.normal(10.0, 20.0, size=100))  # jnp array

def logdensity_fn(x):
    # explicit, jax-native normal logpdf (avoids any non-jax objects)
    loc = x["loc"]
    scale = x["scale"]
    # -0.5 * ((obs - loc)/scale)**2 - log(scale) - 0.5*log(2*pi)
    n = observed.size
    sq = ((observed - loc) / scale) ** 2
    return -0.5 * jnp.sum(sq) - n * jnp.log(scale) - 0.5 * n * jnp.log(2 * jnp.pi)

# build kernel (nuts.step is a plain JAX function)
step_size = 1e-3
inverse_mass_matrix = jnp.ones(2)
nuts = blackjax.nuts(logdensity_fn, step_size, inverse_mass_matrix)

initial_position = {"loc": 1.0, "scale": 2.0}
state = nuts.init(initial_position)

# kernel for lax.scan: carry is state, xs is a single PRNG key
def kernel(carry, key):
    next_state, info = nuts.step(key, carry)   # NOTE: nuts.step NOT jitted here
    return next_state, info

# create keys (jnp array)
keys = jax.random.split(jax.random.PRNGKey(0), 1000)

# JIT the entire scan so it becomes one XLA computation (fast)
scan = jax.jit(lambda s, ks: jax.lax.scan(kernel, s, ks))

# First call will compile (may take some seconds), subsequent calls are fast
# run twice
print("Compiling (slow)...")
t0 = time.perf_counter()
state, infos = scan(state, keys)
t1 = time.perf_counter()
print("Took", t1-t0)

print("Running again (fast)...")
t0 = time.perf_counter()
state, infos = scan(state, keys)
t1 = time.perf_counter()
print("Took", t1-t0)

print("Final position:", state.position)
