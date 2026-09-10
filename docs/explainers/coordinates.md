# Coordinates: what the sampler moves vs what the ansatz sees

A `CoordMode` defines the relationship between **sampler space** (where walkers
live and Metropolis moves happen) and **model input space** (what the ansatz is
evaluated on). Hamiltonian potentials always receive **lab coordinates**, whatever
the mode — `VMC` injects the conversion, so a Hamiltonian never handles it.

## LabCoords — the default

Sampler space is lab space. Nothing is transformed. Sampler shape and model input
shape are both `(n_chains, N*d)`.

## JacobiCoords — centre of mass removed

For a translationally invariant system the centre of mass is a free particle: it
diffuses without bound and contributes nothing to the internal energy, while adding
a slow, badly-mixing direction to every chain.

`JacobiCoords(n_particles_physical, n_dim)` samples in Jacobi coordinates, which
carry only the N−1 *relative* degrees of freedom. The CM mode is gone from the
sampler entirely rather than being fought with a longer chain.

The shapes then differ, which is the thing to remember:

    sampler shape       (n_chains, (N-1)*d)     relative coordinates
    model input shape   (n_chains,  N   *d)     lab coordinates

The ansatz still sees lab coordinates: `wrap_model_apply` applies the inverse
transform on the way in. So the same ansatz works under either mode, unchanged.

`result.cm_mean` / `result.cm_std` report the centre-of-mass drift, which is a useful
sanity check under `LabCoords` and identically zero under `JacobiCoords`.

## Adding a coordinate system

One subclass, three methods:

| method | job |
|---|---|
| `model_input_shape(sample_shape)` | shape used to initialise the ansatz parameters |
| `wrap_model_apply(model_apply)` | wrap so it accepts sampler-space input |
| `samples_to_lab(samples)` | sampler space → lab coordinates, for potentials and estimators |

There is nothing to register: pass your instance as `VMC(coord_mode=...)`.
