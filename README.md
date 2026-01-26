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

Finally, from the root directory of the project, the package must be installed in edit mode:

```bash
pip install -e .
```

### Execution

From the root folder, run

```bash
qvarnet run [>output.txt]
```

The parameters are found inside `./src/qvarnet/cli/parameters`

The [] option is recommended if there are prints in the code. The progressbar is still displayed in the terminal

# WIP: Diagrams

## Class Diagram

```mermaid
classDiagram
    class BaseConfig {
        <<abstract>>
        +config_path: Path
        +data: Dict[str, Any]
        +__init__(config_path: Path)
        +_load_config()* Dict[str, Any]
        +_validate_config()*
        +get(key: str, default=None) Any
        +merge_with(other_config: Dict[str, Any]) Dict[str, Any]
        +save(output_path: Optional[Path]) void
        +to_dict() Dict[str, Any]
    }
    
    class ExperimentConfig {
        +__init__(config_path: Path)
        +_load_config() Dict[str, Any]
        +_get_default_config() Dict[str, Any]
        +_validate_config() void
    }
    
    class CLI {
        -parser: ArgumentParser
        -args: Namespace
        -config: ExperimentConfig
        +__init__()
        +_setup_parser() ArgumentParser
        +parse_args(argv=None) Namespace
        +_parse_overrides() Dict[str, Any]
        +_list_presets_and_exit() void
        +get_args() Dict[str, Any]
        +get_config() ExperimentConfig
        +get_optimizer_args() Dict[str, Any]
        +get_training_args() Dict[str, Any]
        +get_model_args() Dict[str, Any]
        +get_sampler_args() Dict[str, Any]
        +get_hamiltonian_args() Dict[str, Any]
        +get_output_args() Dict[str, Any]
        +get_seed() int
    }
    
    class ConfigurationFunctions {
        <<utility>>
        +load_config(config_path: str) ExperimentConfig
        +create_preset(preset_name: str, **overrides) ExperimentConfig
        +list_presets() List[Dict]
        +validate_config(config_dict: dict) bool
        +get_default_config() ExperimentConfig
        +create_config_from_dict(config_dict: Dict[str, Any]) ExperimentConfig
    }
    
    BaseConfig <|-- ExperimentConfig : inherits
    CLI --> ExperimentConfig : uses
    CLI --> ConfigurationFunctions : uses
    ExperimentConfig --> ConfigurationFunctions : uses
    
    note for CLI "Main CLI interface with\n argument parsing and\n configuration management"
    note for BaseConfig "Abstract base class\nfor configuration handling"
    note for ExperimentConfig "Complete experiment\nconfiguration with\n validation and defaults"
    note for ConfigurationFunctions "Utility functions for\n preset management and\n configuration operations"
```



Flowchart Diagram

```mermaid
flowchart TD
    %% Nodes
    A([Start])
    B[/Read Input File/]
    C{Is Data Clean?}
    D[Process Data]
    E[Log Error]
    F[(Save to DB)]
    G([End])

    %% Edge Connections
    A --> B
    B --> C
    C -- Yes --> D
    C -- No --> E
    E --> G
    D --> F
    F --> G
```