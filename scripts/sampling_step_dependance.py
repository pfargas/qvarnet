import os
import sys
import argparse
import jax

parser = argparse.ArgumentParser(description="Run MCMC sampling")
parser.add_argument(
    "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use"
)
parser.add_argument("--split", action="store_true", help="Use split RNG")
parser.add_argument(
    "--n-chains", type=int, default=1000, help="Number of chains to run"
)
parser.add_argument(
    "--loop-step", type=int, default=200, help="Step in steps per chain"
)
parser.add_argument(
    "--n-trials", type=int, default=2, help="Number of trials for statistics in timing"
)
parser.add_argument(
    "--n-steps", type=int, default=10_000, help="Number of max steps per chain"
)

args = parser.parse_args()

device = args.device
split = args.split if args.split else False
jax.config.update("jax_platform_name", device)
print("Default device:", jax.devices())
N_CHAINS = args.n_chains
SPACING = args.loop_step
N_STEPS = args.n_steps
N_TRIALS = args.n_trials

if SPACING > N_STEPS:
    raise ValueError("Step spacing cannot be larger than the number of steps.")
if N_STEPS < 10:
    raise ValueError("Number of steps must be at least 10.")

from jax import random
from jax import numpy as jnp

import time
from functools import partial

times = []
std_times = []


number_of_steps = range(10, N_STEPS + 1, SPACING)


def erase_block(n_lines):
    # Move cursor up n_lines
    for _ in range(n_lines):
        sys.stdout.write("\033[F")  # Move cursor up one line
        sys.stdout.write("\033[K")  # Clear that line
    sys.stdout.flush()


@partial(jax.jit, static_argnames=("prob_fn",))
def mh_chain_with_all_random_nums(random_values, prob_fn, init_pos):
    # Placeholder implementation of mh_chain

    def mh_kernel(carry, random_values):
        position, old_prob = carry
        proposal = position + (2 * random_values[0] - 1)
        proposal_prob = prob_fn(proposal)
        accept_prob = jnp.minimum(1.0, proposal_prob / old_prob)
        accept = random_values[-1] < accept_prob
        new_position = jnp.where(accept, proposal, position)
        new_prob = jnp.where(accept, proposal_prob, old_prob)
        carry = (new_position, new_prob)
        return carry, new_position

    initial_prob = prob_fn(init_pos)
    carry = (init_pos, initial_prob)
    positions, _ = jax.lax.scan(mh_kernel, carry, random_values)
    return positions

@partial(jax.jit, static_argnames=("n_steps", "prob_fn"))
def mh_chain_with_split(init_key, n_steps, prob_fn, init_pos):
    # Placeholder implementation of mh_chain

    def mh_kernel(carry, key):
        position, old_prob, key = carry
        key, subkey_proposal, subkey_acceptance = random.split(key, 3)
        proposal = position + (2 * random.uniform(subkey_proposal) - 1)
        proposal_prob = prob_fn(proposal)
        accept_prob = jnp.minimum(1.0, proposal_prob / old_prob)
        accept = random.uniform(subkey_acceptance) < accept_prob
        new_position = jnp.where(accept, proposal, position)
        new_prob = jnp.where(accept, proposal_prob, old_prob)
        carry = (new_position, new_prob, key)
        return carry, new_position

    initial_prob = prob_fn(init_pos)
    carry = (init_pos, initial_prob, init_key)
    positions, _ = jax.lax.scan(mh_kernel, carry, None, length=n_steps)
    return positions


sampler = jax.vmap(
    mh_chain_with_all_random_nums,
    in_axes=(
        0,
        None,
        0,
    ),  # random_values, prob_fn, init_position
    out_axes=0,
)

sampler_split = jax.vmap(
    mh_chain_with_split,
    in_axes=(0, None, None, 0),  # init_key, n_steps, prob_fn, init_position
    out_axes=0,
)


@jax.jit
def prob_fn(positions):
    """A simple probability function: a Gaussian centered at the origin."""
    return jnp.exp(-0.5 * jnp.sum(positions**2, axis=-1))


def time_run_sampling(n_chains, n_steps, DoF=1, n_trials=10):

    start_time_prep = time.perf_counter()
    init_keys = jax.random.split(random.PRNGKey(42), n_chains)

    init_positions = jnp.zeros((n_chains, DoF))

    random_values = jax.random.uniform(random.PRNGKey(0), (n_chains, n_steps, DoF + 1))

    # Generate JAX array to append times of execution to have statistics
    execution_times = jnp.zeros(n_trials)
    end_time_prep = time.perf_counter()
    print(f"\tPreparation completed in {end_time_prep - start_time_prep:.2e} seconds.")

    start_time = time.perf_counter()
    _ = sampler(random_values, prob_fn, init_positions)
    end_time = time.perf_counter()
    print(f"\tWarm-up sampling completed in {end_time - start_time:.2e} seconds.")

    for i in range(n_trials):
        start_time = time.perf_counter()
        result = sampler(random_values, prob_fn, init_positions)
        jax.tree.map(lambda x: x.block_until_ready(), result)

        end_time = time.perf_counter()
        print(f"\t\tSampling completed in {end_time - start_time:.2e} seconds.")

        start_time_append = time.perf_counter()
        execution_times = execution_times.at[i].set(end_time - start_time)
        end_time_append = time.perf_counter()
        print(
            f"\tTiming recorded in {end_time_append - start_time_append:.2e} seconds."
        )

    erase_block(n_trials + 1)

    print(f"\tAll {n_trials} trials completed.")
    start_mean_std = time.perf_counter()
    mean_time = execution_times.mean().item()
    std_time = execution_times.std().item()
    end_mean_std = time.perf_counter()
    print(f"\tMean and std computed in {end_mean_std - start_mean_std:.2e} seconds.")

    return mean_time, std_time


def time_run_sampling_split(n_chains, n_steps, DoF=1, n_trials=10):

    start_time_prep = time.perf_counter()
    init_keys = jax.random.split(random.PRNGKey(42), n_chains)

    init_positions = jnp.zeros((n_chains, DoF))

    # Generate JAX array to append times of execution to have statistics
    execution_times = jnp.zeros(n_trials)
    end_time_prep = time.perf_counter()
    print(f"\tPreparation completed in {end_time_prep - start_time_prep:.2e} seconds.")

    start_time = time.perf_counter()
    _ = sampler_split(init_keys, n_steps, prob_fn, init_positions)
    end_time = time.perf_counter()
    print(f"\tWarm-up sampling completed in {end_time - start_time:.2e} seconds.")

    for i in range(n_trials):
        start_time = time.perf_counter()
        result = sampler_split(init_keys, n_steps, prob_fn, init_positions)
        jax.tree.map(lambda x: x.block_until_ready(), result)

        end_time = time.perf_counter()
        print(f"\t\tSampling completed in {end_time - start_time:.2e} seconds.")

        start_time_append = time.perf_counter()
        execution_times = execution_times.at[i].set(end_time - start_time)
        end_time_append = time.perf_counter()
        print(
            f"\tTiming recorded in {end_time_append - start_time_append:.2e} seconds."
        )

    erase_block(n_trials + 1)

    print(f"\tAll {n_trials} trials completed.")
    start_mean_std = time.perf_counter()
    mean_time = execution_times.mean().item()
    std_time = execution_times.std().item()
    end_mean_std = time.perf_counter()
    print(f"\tMean and std computed in {end_mean_std - start_mean_std:.2e} seconds.")

    return mean_time, std_time


for n_step in number_of_steps:
    print(f"\n\nRunning sampling with {n_step} steps per chain...\n")

    start_time_sampler = time.perf_counter()
    sampler = sampler_split if split else sampler
    print(f"\tUsing sampler: {'split RNG' if split else 'all random numbers'}")
    end_time_sampler = time.perf_counter()
    print(
        f"\tSampler selection completed in {end_time_sampler - start_time_sampler:.2e} seconds.\n"
    )

    start_global_timer = time.perf_counter()
    if split:
        mean_time, std_time = time_run_sampling_split(
            N_CHAINS,
            n_step,
            n_trials=N_TRIALS,
        )
    else:
        mean_time, std_time = time_run_sampling(
            N_CHAINS,
            n_step,
            n_trials=N_TRIALS,
        )
    end_global_timer = time.perf_counter()
    print(
        f"\n\t**Total time for sampling with {n_step} steps per chain: {end_global_timer - start_global_timer:.2f} seconds.**"
    )

    start_time_appending = time.perf_counter()

    times.append(mean_time)
    std_times.append(std_time)

    end_time_appending = time.perf_counter()
    print(
        f"\n\tAppending times completed in {end_time_appending - start_time_appending:.2e} seconds."
    )


import csv
import os

os.makedirs(f"results/steps/{device}", exist_ok=True)

with open(
    f"results/steps/{device}/times_with_{N_CHAINS}_split_{split}.csv",
    mode="w",
    newline="",
) as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Number of Steps", "Time (seconds)", "Standard Deviation (seconds)"]
    )
    for n_step, t, std in zip(number_of_steps, times, std_times):
        writer.writerow([n_step, t, std])