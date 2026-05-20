# Writing a qvarnet experiment

## 1. Define the model

Pick a model from `MODEL_REGISTRY` and build its constructor args.

```python
from qvarnet.models import get_model

model_name = "deep-set"
model_args = {
    "n_particles": 5,
    "phi_hidden_architecture": [64, 64],
    "F_hidden_architecture": [64, 64],
    "hidden_internal_dimension": 16,
}
model = get_model(model_name, **model_args)
```

Available models: `"mlp"`, `"deep-set"`, `"mlp-fourth-decay"`, `"exponential-deep-set"`, ...
Check `MODEL_REGISTRY` for the full list.

---

## 2. Define the Hamiltonian

```python
from qvarnet.hamiltonian import get_hamiltonian

hamiltonian = get_hamiltonian("harmonic-oscillator")
```

Available Hamiltonians: `"harmonic-oscillator"`, `"nn-oscillator"`, `"calogero-sutherland"`.

---

## 3. Choose a coordinate mode

```python
from qvarnet.config.coord_mode import LabCoords, JacobiCoords

coord_mode = LabCoords()
# or, to remove the centre-of-mass degree of freedom:
# coord_mode = JacobiCoords(n_particles_physical=5, n_dim=1)
```

---

## 4. Get the sample shape from the model

```python
n_chains = 2000
shape = model.get_input_shape(model_args, n_chains)  # (n_chains, dof)
```

---

## 5. Define the optimizer

```python
import optax

optimizer = optax.adam(learning_rate=1e-3)
```

---

## 6. Configure training and sampling

```python
from qvarnet.config.training_setup import TrainingConfig, SamplingConfig

output_path = "./outputs/my_experiment/"

training_config = TrainingConfig(
    n_epochs=5000,
    checkpoint_path=output_path,
    is_update_step_size=True,
    rng_seed=42,
)

sampling_config = SamplingConfig(
    step_size=0.5,
    chain_length=200,
    thermalization_steps=50,
    thinning_factor=1,
    PBC=40.0,
)
```

> **Note:** Both configs are frozen dataclasses and are passed as `static_argnames`
> to `jax.jit`. Changing any field triggers a JIT retrace.

---

## 7. Save the run config to disk (before training)

This writes `checkpoints/run_config.json`, which is required for `load_run()` to work later.
Call it before `train()` so the config is on disk even if training crashes.

```python
from qvarnet.utils import save_run_config

save_run_config(
    path=output_path,
    model_name=model_name,
    model_args=model_args,
    sample_shape=shape,
    coord_mode=coord_mode,
    training_config=training_config,
)
```

---

## 8. Set up callbacks

`RunOutputCallback` is added automatically by `train()` with `n=1, metric=["energy"]`.
Pass your own instance to override the defaults.

```python
from qvarnet.callbacks import RunOutputCallback

callbacks = [
    RunOutputCallback(
        n=5,
        path=output_path,
        metric=["energy", "std"],
    )
]
```

---

## 9. Train

```python
from qvarnet import train

result = train(
    shape=shape,
    model=model,
    optimizer=optimizer,
    hamiltonian=hamiltonian,
    training_config=training_config,
    sampler_params=sampling_config,
    coord_mode=coord_mode,
    callbacks=callbacks,
)
```

---

## 10. What you have on disk after training

```
outputs/my_experiment/
├── history.csv                    # scalar diagnostics, one row per epoch
│                                  # columns: step, energy, std, acceptance_rate,
│                                  #          step_size, cm_mean, cm_std
└── checkpoints/
    ├── run_config.json            # model + training config (written in step 7)
    ├── best_energy_0.msgpack      # full params of the lowest-energy epoch
    ├── best_energy_1.msgpack      # 2nd lowest, etc.
    ├── best_std_0.msgpack         # full params of the lowest-std epoch
    └── ...
```

---

## 11. Reload for inference

```python
from qvarnet.utils import load_run

run = load_run(output_path, checkpoint_filename="best_energy_0.msgpack")
# run.model          — reconstructed Flax model
# run.params         — parameters from the best-energy epoch
# run.training_config
# run.coord_mode
```

---

## 12. Load history for analysis

```python
import csv

with open(f"{output_path}/history.csv") as f:
    rows = list(csv.DictReader(f))

energies = [float(r["energy"]) for r in rows]
stds     = [float(r["std"])    for r in rows]
steps    = [int(r["step"])     for r in rows]
```
