# Contributing to qvarnet

## Extending it

You should not need to edit the library to add a Hamiltonian, an ansatz, a sampler,
a proposal family, a loss or a callback. Write a class in your own file and pass the
instance — the same way you pass an optax optimizer. There is no registry and no name
to invent.

The three worked examples in [`examples/`](examples/) are the whole story:

```python
@dataclass(frozen=True)
class MySampler(Metropolis):
    def constrain(self, x):
        return jnp.sort(x)

result = VMC(..., sampler=MySampler()).run()
```

If something genuinely cannot be done from outside, that is a bug in the seam — open
an issue for the seam rather than a patch that special-cases your use.

Anything jit-static (samplers, proposals, configs) must be a **frozen dataclass**, so
it is hashable and JAX can treat it as a compile-time constant.

## The layering

The package is a stack, and imports only ever point downward:

    core < physics < ansatz < sampling < optim < analysis < callbacks < vmc

```bash
uv run python scripts/check_layers.py
```

This is enforced because it was not always true: `callbacks` and `vmc` used to import
each other, `callbacks` and `diagnostics` likewise, and a `config` package imported
the samplers it configured. Two of those cycles were hidden by function-local
imports, so **all internal imports are absolute** — you cannot see what
`from ..config import X` crosses without counting dots.

## Where prose goes

Documentation is not the problem; documentation in the wrong place is.

| where | answers | ships? |
|---|---|---|
| docstrings | what this callable is and what it takes | yes |
| `docs/adr/` | *why* a design choice was made | yes |
| `docs/explainers/` | what a method is, how to read its output | yes |
| `docs/notes/` | working notes, open questions | **no** |

```bash
uv run python scripts/check_docs.py            # enforce
uv run python scripts/check_docs.py --report   # see the ranking
```

A docstring is a summary line plus `Args:`/`Returns:` where the types do not already
say it. No rationale, no changelog, no roadmap pointers. When one outgrows the cap
(20 lines, 24 for a module) the surplus is nearly always rationale or a tutorial:
move it, do not delete it. `docs/notes/` exists so nothing has to be polished before
it can be written down.

## Numerics are a contract

This is numerical code, so a refactor that changes results is a bug even if every
other test passes.

```bash
uv run pytest tests/test_golden_trace.py    # bit-identical energy traces
uv run pytest tests/test_no_retrace.py      # the epoch update compiles once
uv run python scripts/bench_epoch.py        # wall-time per epoch vs baseline
```

`test_golden_trace.py` pins fixed-seed traces exactly, across the plain sampler, a
constrained sampler and the block-adaptive warmup. If your change moves a trace,
either it should not have — or it is a deliberate fix, in which case regenerate with
`--update` **and say why in the commit message**.

Preserve RNG consumption order when touching the sampler:
`split(key, n_chains)` → per-chain `split(key, n_steps)` → in-kernel `split(key)`.

## Before opening a PR

```bash
uv sync
uv run pytest -q
uv run python scripts/check_layers.py
uv run python scripts/check_docs.py
uv run ruff check src tests examples && uv run ruff format --check src
```

## The runq sweep targets

`soft_sphere_gas/point.py` and `calogero-sutherland/cs_sweep/point.py` are targets for
the `runq` job scheduler, which keys completed runs by the canonical JSON of the
resolved parameter dict. **Adding, removing, renaming or re-defaulting a `run_point`
parameter re-keys every run in that sweep** and orphans finished work.
`tests/test_runq_targets.py` pins those signatures against a snapshot. The library may
change freely beneath them; `point.py` absorbs it.
