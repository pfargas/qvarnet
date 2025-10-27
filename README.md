# qvarnet: A package to simulate quantum systems via VMC with Artificial Neural Networks ansatzë

### Installation

To install the conda environment, from the root directory of the project run:

```bash
conda env create -f environment_config.yaml
```

and activate with

```bash
conda activate jax
```

### Execution

From the root folder, run

```bash
qvarnet run
```

The parameters are found inside `./src/qvarnet/cli/parameters`