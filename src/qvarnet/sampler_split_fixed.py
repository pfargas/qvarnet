"""
Fixed version of sampler_split.py that addresses the numerical issues
"""

from functools import partial
import jax
from jax import random
from jax import numpy as jnp
from matplotlib.pyplot import hist


@partial(jax.jit, static_argnames=("prob_fn"))
def mh_kernel(key, prob_fn, prob_params, position, prob, step_size, PBC=10.0):
    uniform_random_numbers = random.uniform(key, shape=(position.shape[0] + 1,))
    proposal = position + step_size * (2 * uniform_random_numbers[:-1] - 1)
    proposal = ((proposal + 0.5 * PBC) % PBC) - 0.5 * PBC
    proposal_prob = prob_fn(proposal, prob_params)
    accept_prob = jnp.minimum(1.0, proposal_prob / prob)
    accept = uniform_random_numbers[-1] < accept_prob
    new_position = jnp.where(accept, proposal, position)
    new_prob = jnp.where(accept, proposal_prob, prob)
    
    # Fix: Actually track acceptance rate
    acceptance_rate = accept.astype(jnp.float32)
    return new_position, new_prob, acceptance_rate


@partial(jax.jit, static_argnames=("prob_fn", "n_steps"))
def mh_chain_fixed(key, PBC, prob_fn, prob_params, init_position, step_size, n_steps):
    """
    Fixed MH chain using proper key management.
    
    Key fix: Generate subkeys upfront rather than during scan
    """
    
    init_prob = prob_fn(init_position, prob_params)
    
    # FIX: Generate all subkeys upfront - creates independent streams
    subkeys = random.split(key, n_steps)
    
    def body_fn(carry, subkey):
        position, prob, acceptance_sum = carry
        new_position, new_prob, acceptance = mh_kernel(
            key=subkey,
            prob_fn=prob_fn,
            prob_params=prob_params,
            position=position,
            prob=prob,
            step_size=step_size,
            PBC=PBC,
        )
        return (new_position, new_prob, acceptance_sum + acceptance), new_position

    carry0 = (init_position, init_prob, jnp.array(0.0))
    (_, _, total_acceptance), positions = jax.lax.scan(body_fn, carry0, subkeys)
    
    # Calculate actual acceptance rate
    acceptance_rate = total_acceptance / n_steps
    return positions, acceptance_rate


@partial(jax.jit, static_argnames=("prob_fn", "n_steps"))
def mh_chain_chunked(key, PBC, prob_fn, prob_params, init_position, step_size, n_steps, chunk_size=1000):
    """
    Hybrid approach: generate random numbers in chunks to balance memory usage
    with numerical consistency similar to pre-generated approach.
    """
    
    init_prob = prob_fn(init_position, prob_params)
    current_position = init_position
    current_prob = init_prob
    
    all_positions = []
    total_acceptance = jnp.array(0.0)
    
    for i in range(0, n_steps, chunk_size):
        current_chunk_size = min(chunk_size, n_steps - i)
        
        # Generate chunk of random numbers (like pre-generated but in smaller pieces)
        key, subkey = random.split(key)
        chunk_rand_nums = random.uniform(
            subkey, 
            shape=(current_chunk_size, init_position.shape[0] + 1)
        )
        
        # Process chunk with pre-generated-style kernel
        @partial(jax.jit, static_argnames=("prob_fn"))
        def mh_kernel_precomputed(uniform_random_numbers, prob_fn, prob_params, position, prob, step_size, PBC=10.0):
            proposal = position + step_size * (2 * uniform_random_numbers[:-1] - 1)
            proposal = ((proposal + 0.5 * PBC) % PBC) - 0.5 * PBC
            proposal_prob = prob_fn(proposal, prob_params)
            accept_prob = jnp.minimum(1.0, proposal_prob / prob)
            accept = uniform_random_numbers[-1] < accept_prob
            new_position = jnp.where(accept, proposal, position)
            new_prob = jnp.where(accept, proposal_prob, prob)
            acceptance = accept.astype(jnp.float32)
            return new_position, new_prob, acceptance
        
        # Process the chunk
        def process_chunk(carry, rand_vals):
            pos, prob, acc_sum = carry
            new_pos, new_prob, acc = mh_kernel_precomputed(
                rand_vals, prob_fn, prob_params, pos, prob, step_size, PBC
            )
            return (new_pos, new_prob, acc_sum + acc), new_pos
        
        carry0 = (current_position, current_prob, jnp.array(0.0))
        (_, new_prob, chunk_acceptance), chunk_positions = jax.lax.scan(
            process_chunk, carry0, chunk_rand_nums
        )
        
        all_positions.append(chunk_positions)
        current_position = chunk_positions[-1]  # Use last position as seed for next chunk
        current_prob = new_prob
        total_acceptance = total_acceptance + chunk_acceptance
    
    # Combine all chunks
    positions = jnp.concatenate(all_positions, axis=0)
    acceptance_rate = total_acceptance / n_steps
    
    return positions, acceptance_rate


if __name__ == "__main__":
    print("This is the FIXED module for Metropolis-Hastings sampling.")
    
    # Test the fixed implementation
    sampler = jax.vmap(
        mh_chain_fixed,
        in_axes=(
            0,  # key
            None,  # PBC
            None,  # prob_fn
            None,  # prob_params
            0,  # init_position
            None,  # step_size
            None,  # n_steps
        ),
        out_axes=0,
    )
    
    n_chains = 4
    DoF = 2
    n_steps = 10000
    key = random.PRNGKey(0)
    keys = random.split(key, n_chains)
    init_positions = jnp.zeros((n_chains, DoF))
    step_size = 1.0
    PBC = 10.0
    prob_fn = lambda x, params: jnp.exp(-jnp.sum(x**2))  # Example: Gaussian
    prob_params = None
    
    samples, acceptance_rates = sampler(
        keys,
        PBC,
        prob_fn,
        prob_params,
        init_positions,
        step_size,
        n_steps,
    )
    
    print("Samples shape:", samples.shape)  # (n_chains, n_steps, DoF)
    print("Acceptance rates per chain:", acceptance_rates)
    
    hist(samples.reshape(-1, DoF)[:, 0], bins=30)
    import matplotlib.pyplot as plt
    plt.show()
