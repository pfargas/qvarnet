from functools import partial
import jax
from jax import random
from jax import numpy as jnp
from matplotlib.pyplot import hist


@partial(jax.jit, static_argnames=("prob_fn"))
def mh_kernel(
    uniform_random_numbers, prob_fn, prob_params, position, prob, step_size, PBC=10.0
):
    proposal = position + step_size * (
        2 * uniform_random_numbers[: position.shape[0]] - 1
    )
    proposal = ((proposal + 0.5 * PBC) % PBC) - 0.5 * PBC
    proposal_prob = prob_fn(proposal, prob_params)
    accept_prob = jnp.minimum(1.0, proposal_prob / prob)
    accept = uniform_random_numbers[-1] < accept_prob
    new_position = jnp.where(accept, proposal, position)
    new_prob = jnp.where(accept, proposal_prob, prob)
    # acceptance_rate = jnp.mean(accept.astype(jnp.float32))
    # acceptance_rate = accept.astype(jnp.float32)
    acceptance_rate = 0.0
    return new_position, new_prob, acceptance_rate


@jax.jit
def adapt_step_size(step_size, accept, target=0.5, lr=0.01):
    # accept is a float 0.0 or 1.0
    # return step_size * jnp.exp(lr * (accept - target))
    return step_size


@partial(jax.jit, static_argnames=("prob_fn"))
def mh_chain(random_values, PBC, prob_fn, prob_params, init_position, step_size=1.0):
    """
    Single MH chain using pre-generated step keys.
    random_values: shape (n_steps, DoF + 1)
    init_position: shape (DoF,)
    """

    init_prob = prob_fn(init_position, prob_params)
    carry0 = (init_position, init_prob, step_size)

    def body_fn(carry, random_values):
        position, prob, step = carry
        new_position, new_prob, _ = mh_kernel(
            random_values, prob_fn, prob_params, position, prob, step, PBC
        )
        return (new_position, new_prob, step), new_position

    (_, _, _), positions = jax.lax.scan(body_fn, carry0, random_values)
    return positions


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    import matplotlib.pyplot as plt
    import numpy as np
    import time

    # set CPU as device
    jax.config.update("jax_platform_name", "cuda")

    print(jax.devices())
    print("This is the qvarnet.sampler module.")
    print("TESTING PLAYGROUND FOR THE SAMPLER MODULE")

    master_key = random.PRNGKey(0)

    @jax.jit
    def test_2d_prob_fn(x, params):
        return jnp.exp(-0.5 * jnp.sum((x) ** 2, axis=-1))
        # return jnp.exp(-0.5 * x**2)

    sampler = jax.vmap(
        mh_chain,
        in_axes=(
            0,
            None,
            None,
            None,
            0,
        ),  # random_values, PBC, prob_fn, prob_params, init_position
        out_axes=0,
    )

    def sampler_timer(n_steps):
        n_chains = 1
        DoF = 1
        PBC = 20.0
        key_prop, key_acc = random.split(master_key)

        keys_prop = random.split(key_prop, n_chains * n_steps).reshape(
            n_chains, n_steps, 2
        )
        keys_acc = random.split(key_acc, n_chains * n_steps).reshape(
            n_chains, n_steps, 2
        )

        init_positions = jax.random.normal(random.PRNGKey(0), (n_chains, DoF))

        import time

        start_time = time.perf_counter()
        samples = sampler(
            keys_prop, keys_acc, PBC, test_2d_prob_fn, None, init_positions
        )
        end_time = time.perf_counter()
        print(
            f"Sampling with {n_steps} steps completed in {end_time - start_time:.2f} seconds."
        )
        return samples

    # Actual test
    times = []
    print("Starting actual test...")
    # for n_steps in np.logspace(3, 6, num=30, dtype=int):
    # sampler_timer(n_steps)  # Warm-up for this step count
    # start = time.perf_counter()
    # sampler_timer(n_steps)
    # end = time.perf_counter()
    # times.append(end - start)

    # plt.plot(np.log10(np.logspace(3, 6, num=30)), times, marker='o', label='Sampling Time')
    # plt.xlabel('log10(Number of Steps)')
    # plt.ylabel('Time (seconds)')
    # plt.title('Sampling Time vs Number of Steps')
    # plt.legend()
    # plt.grid()
    # plt.show()

    n_chains = 1
    DoF = 1
    n_steps = 10_000_000
    PBC = 5.0

    rand_nums = jax.random.uniform(random.PRNGKey(42), (n_chains, n_steps, DoF + 1))

    print("CONFIG USED:")
    print(f"n_chains: {n_chains}, DoF: {DoF}, n_steps: {n_steps}")

    init_positions = jax.random.normal(random.PRNGKey(0), (n_chains, DoF)) * 2.0

    import time

    start_time = time.perf_counter()
    samples = sampler(rand_nums, PBC, test_2d_prob_fn, None, init_positions)
    end_time = time.perf_counter()
    print(f"Sampling completed in {end_time - start_time:.2f} seconds.")
    start_time = time.perf_counter()
    samples = sampler(rand_nums, PBC, test_2d_prob_fn, None, init_positions)
    end_time = time.perf_counter()
    print(f"Sampling completed in {end_time - start_time:.2e} seconds.")

    print("Samples shape:", samples.shape)  # (n_chains, DoF)
    average_position = jnp.mean(samples.reshape(-1, DoF), axis=0)
    print("Shape after reshaping:", samples.reshape(-1, DoF).shape)
    print("Average position over all samples:", average_position)
    std_position = jnp.std(samples.reshape(-1, DoF), axis=0)
    print("Std position over all samples:", std_position)

    print("THEORETICAL VALUES:")
    print("Average position: 0.0, 0.0")
    print("Std position: 1.0, 1.0")

    plt.figure(figsize=(8, 6))
    plt.hist(
        samples.reshape(-1, DoF)[:, 0],
        bins=100,
        density=True,
        alpha=0.7,
        label="Sampled",
    )
    x = jnp.linspace(-PBC / 2, PBC / 2, 1000)
    plt.plot(
        x,
        (1 / jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * x**2),
        label="Theoretical",
        color="red",
    )
    plt.title("1D Marginal Distribution")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    # histo_data = jnp.histogram2d(
    #     samples.reshape(-1, DoF)[:, 0],
    #     samples.reshape(-1, DoF)[:, 1],
    #     # bins=jnp.ceil(jnp.sqrt(n_chains)).astype(int),
    #     bins=100,
    #     range=[[-PBC / 2, PBC / 2], [-PBC / 2, PBC / 2]],
    #     density=False,
    # )
    # # plt.figure(figsize=(8, 6))

    # # plt.pcolor(histo_data[1], histo_data[2], histo_data[0].T, shading="auto")

    # # 3d plot
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection="3d")
    # X, Y = jnp.meshgrid(histo_data[1][:-1], histo_data[2][:-1])
    # Z = histo_data[0].T

    # # set z axis to something small
    # ax.set_zlim(0, jnp.max(Z) * 1.1)
    # ax.set_xlim(-PBC / 2, PBC / 2)
    # ax.set_ylim(-PBC / 2, PBC / 2)
    # ax.plot_surface(X, Y, Z, cmap="viridis")

    # # set axes squared
    # plt.axis("equal")
    # plt.show()
