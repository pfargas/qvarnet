import json
from computations_JAX import run_experiment   # <- your script function

with open("params.json") as f:
    configs = json.load(f)

for i, cfg in enumerate(configs):
    print(f"Running config {i+1}/{len(configs)}: {cfg}")
    run_experiment(**cfg)
