"""The VMC driver: the public entry point for ground-state optimisation.

A run is four phases, and this file is just their order::

    build_context  ->  warm_up  ->  make_update_fn  ->  run_loop

Each lives in its own module (``setup``, ``warmup``, ``step``, ``loop``), so the
thing you have to read to understand the flow is short, and the thing you have to
read to change one phase is only that phase.
"""

from qvarnet.vmc.loop import run_loop
from qvarnet.vmc.result import TrainResult
from qvarnet.vmc.setup import build_context
from qvarnet.vmc.step import make_update_fn
from qvarnet.vmc.warmup import warm_up


class VMC:
    """A variational Monte Carlo ground-state optimisation.

    Minimises E[theta] = <psi|H|psi> / <psi|psi> over the parameters of a
    log-amplitude ansatz, sampling |psi|^2 with the given ``sampler``.

        result = VMC(
            shape=(n_chains, n_particles * n_dim),
            model=psi,
            optimizer=optax.adam(1e-3),
            hamiltonian=ham,
            training_config=TrainingConfig(n_epochs=5_000),
            sampler=OrderedMetropolis(),
        ).run()

    Every pluggable part is an object you construct and pass -- the ansatz, the
    Hamiltonian, the optimizer, the sampler, the callbacks. There is no registry
    and no name to look up.

    Args:
        shape: ``(n_chains, dof)`` of the sampler.
        model: the ansatz. Must return **log|psi(x)|**, shape ``(..., 1)``.
        optimizer: any optax GradientTransformation.
        hamiltonian: a ContinuousHamiltonian.
        training_config: epochs, seed, checkpointing, step-size adaptation.
        initial_chain_config: how walkers start and warm up.
        sampler_params: a SamplingConfig, or a dict of its kwargs.
        coord_mode: LabCoords (default) or JacobiCoords.
        sampler: a Sampler; defaults to Metropolis().
        model_name, model_args: provenance recorded in run_config.json.
        qgt_config: QGTConfig (or a dict) for the stochastic-reconfiguration path.
        auxiliary_losses: extra loss terms added to the VMC energy.
        callbacks: Callback instances; a NaN guard is always prepended.
        select: metric ranking retained snapshots, lower being better. A key name
            ("std" default, "energy", ...) or a callable ``(metrics) -> float``.
        k_best: how many best-by-``select`` parameter sets to retain; 0 keeps none.
        init_params: warm-start parameters. A resumable checkpoint takes precedence.
    """

    def __init__(
        self,
        shape,
        model,
        optimizer,
        hamiltonian,
        training_config,
        *,
        initial_chain_config=None,
        sampler_params=None,
        coord_mode=None,
        sampler=None,
        model_name=None,
        model_args=None,
        qgt_config=None,
        auxiliary_losses=(),
        callbacks=None,
        select="std",
        k_best=3,
        init_params=None,
    ):
        self._kwargs = dict(
            shape=shape,
            model=model,
            optimizer=optimizer,
            hamiltonian=hamiltonian,
            training_config=training_config,
            initial_chain_config=initial_chain_config,
            sampler_params=sampler_params,
            coord_mode=coord_mode,
            sampler=sampler,
            model_name=model_name,
            model_args=model_args,
            qgt_config=qgt_config,
            auxiliary_losses=auxiliary_losses,
            callbacks=callbacks,
            select=select,
            k_best=k_best,
            init_params=init_params,
        )

    def run(self) -> TrainResult:
        """Train, and return the run's history, parameters and final sampler state."""
        ctx = build_context(**self._kwargs)
        warm_up(ctx)
        result = run_loop(ctx, make_update_fn(ctx))
        if ctx.training_config.print_summary:
            result.summary()
        return result


def train(
    shape,
    model,
    optimizer,
    hamiltonian,
    training_config,
    initial_chain_config=None,
    sampler_params=None,
    coord_mode=None,
    sampler=None,
    model_name=None,
    model_args=None,
    qgt_config=None,
    auxiliary_losses=(),
    callbacks=None,
    select="std",
    k_best=3,
    init_params=None,
) -> TrainResult:
    """Function form of :class:`VMC` -- ``VMC(...).run()``. See VMC for the arguments."""
    return VMC(
        shape,
        model,
        optimizer,
        hamiltonian,
        training_config,
        initial_chain_config=initial_chain_config,
        sampler_params=sampler_params,
        coord_mode=coord_mode,
        sampler=sampler,
        model_name=model_name,
        model_args=model_args,
        qgt_config=qgt_config,
        auxiliary_losses=auxiliary_losses,
        callbacks=callbacks,
        select=select,
        k_best=k_best,
        init_params=init_params,
    ).run()
