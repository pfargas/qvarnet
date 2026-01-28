import json
import sys
import os
import argparse
from pathlib import Path

# Import new configuration system
from .parameters import load_config, create_preset, list_presets, validate_config


class CLI:
    """CLI with configuration system support.

    When initialized, this class sets up an argument parser. The parser is executed by
    the `parse_args` method, which processes command line arguments, loads the configuration
    (from file, preset, or defaults), applies overrides, and validates the configuration.
    The loaded configuration and parsed arguments can be accessed via getter methods.

    Attributes:
        parser (argparse.ArgumentParser): The argument parser instance.
        args (argparse.Namespace): Parsed command line arguments.
        config (ExperimentConfig): Loaded configuration object.
    Methods:
        parse_args(argv=None): Parse command line arguments and load configuration.
        get_args(): Get all parsed arguments as a dictionary.
        get_config(): Get the loaded configuration object.
        get_optimizer_args(): Get optimizer configuration.
        get_training_args(): Get training configuration.
        get_model_args(): Get model configuration.
        get_sampler_args(): Get sampler configuration.
        get_hamiltonian_args(): Get hamiltonian configuration.
        get_output_args(): Get output configuration.
        get_seed(): Get experiment seed.

    """

    def __init__(self):
        self.parser = self._setup_parser()
        self.args = None
        self.config = None

    def _setup_parser(self):
        """Setup enhanced argument parser."""
        parser = argparse.ArgumentParser(
            "QVarNet CLI",
            description="Quantum Variational Monte Carlo with advanced configuration management",
        )

        # Commands
        parser.add_argument(
            "command", choices=["run", "list-presets"], help="Command to execute"
        )

        # Configuration options
        parser.add_argument(
            "--config", "-c", type=str, help="Path to configuration file"
        )
        parser.add_argument("--preset", "-p", type=str, help="Use preset configuration")

        parser.add_argument(
            "--preset-list",
            action="store_true",
            help="List all available preset configurations",
        )

        # Configuration overrides
        parser.add_argument(
            "--override",
            "-o",
            type=str,
            action="append",
            help="Override configuration values (key=value)",
        )

        # Output options
        parser.add_argument(
            "--config-dump",
            action="store_true",
            help="Print loaded configuration and exit",
        )

        parser.add_argument(
            "--split",
            "-s",
            action="store_true",
            help="Use the sampler_split module for sampling",
        )

        return parser

    def parse_args(self, argv=None):
        """Parse command line arguments."""
        self.args = self.parser.parse_args(argv)

        # self.args is a NameSpace with the following possible attributes:
        # - command: "run" or "list-presets"
        # - config: path to config file (str) or None
        # - preset: name of preset configuration (str) or None
        # - override: list of override strings (key=value) or None
        # - config_dump: bool, whether to print config and exit

        # Load configuration based on arguments
        if self.args.command == "list-presets":
            self._list_presets_and_exit()
        elif self.args.config:
            self.config = load_config(self.args.config)
        elif self.args.preset:
            overrides = self._parse_overrides() if self.args.override else {}
            self.config = create_preset(self.args.preset, **overrides)
        else:
            # Default configuration
            from .parameters import get_default_config

            self.config = get_default_config()

        # Apply any additional overrides
        if self.args.override:
            overrides = self._parse_overrides()
            self.config.data = self.config.merge_with(overrides)

        # Validate configuration
        if not validate_config(self.config.data):
            raise ValueError("Configuration validation failed")

        # Print configuration if requested
        # if self.args.config_dump:
        #     import json

        #     print(json.dumps(self.config.data, indent=2))
        #     sys.exit(0)

        return self.args

    def _parse_overrides(self):
        """Parse override arguments into dictionary."""
        overrides = {}
        for override in self.args.override:
            if "=" in override:
                key, value = override.split("=", 1)
                # Try to parse as JSON, fallback to string
                try:
                    overrides[key] = json.loads(value)
                except json.JSONDecodeError:
                    overrides[key] = value
            else:
                raise ValueError(f"Invalid override format: {override}")
        return overrides

    def _list_presets_and_exit(self):
        """List available presets and exit."""
        presets = list_presets()
        print("Available preset configurations:")
        for preset in presets:
            print(f"  {preset['name']}: {preset['description']}")
            if preset["tags"]:
                print(f"    Tags: {', '.join(preset['tags'])}")
        sys.exit(0)

    def get_args(self):
        """Get all arguments as dictionary."""
        return self.args.__dict__ if self.args else {}

    def get_config(self):
        """Get loaded configuration."""
        return self.config

    def get_optimizer_args(self):
        """Get optimizer configuration."""
        return self.config.get("optimizer", {}) if self.config else {}

    def get_training_args(self):
        """Get training configuration."""
        return self.config.get("training", {}) if self.config else {}

    def get_model_args(self):
        """Get model configuration."""
        return self.config.get("model", {}) if self.config else {}

    def get_sampler_args(self):
        """Get sampler configuration."""
        return self.config.get("sampler", {}) if self.config else {}

    def get_hamiltonian_args(self):
        """Get hamiltonian configuration."""
        return self.config.get("hamiltonian", {}) if self.config else {}

    def get_output_args(self):
        """Get output configuration."""
        return self.config.get("output", {}) if self.config else {}

    def get_seed(self):
        """Get experiment seed."""
        experiment_config = self.config.get("experiment", {}) if self.config else {}
        seed = experiment_config.get("seed", None)

        return seed if seed is not None else 42  # Default seed

    def get_info_experiment(self):
        """Get experiment information."""
        return self.config.get("experiment", {}) if self.config else {}


def main():
    """Main entry point with enhanced CLI support."""
    # Use new enhanced CLI by default
    cli = CLI()

    try:
        args = cli.parse_args()
    except Exception as e:
        print(f"Error parsing arguments: {e}")
        sys.exit(1)

    if args.command == "run":
        run_experiment(cli)
    elif args.command == "list-presets":
        # Already handled in parse_args()
        pass
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


def run_experiment(cli):
    """Run experiment using enhanced configuration system."""
    if not hasattr(cli, "config") or cli.config is None:
        print("Error: No configuration loaded")
        sys.exit(1)

    welcome_string = """
    *************************************
    *       QVarNet CLI interface      *
    *************************************
    """
    print(welcome_string)

    experiment_description = cli.config.data.get("experiment", {}).get(
        "description", "No description provided."
    )

    experiment_name = cli.config.data.get("experiment", {}).get(
        "name", "Untitled Experiment"
    )

    exp_info_log = f"""
    *************************************
    *       Experiment Information     *
    *************************************
    Experiment Name: {experiment_name}
    Description: {experiment_description}
    *************************************
    """

    print(exp_info_log)

    # config_dump
    if cli.args.config_dump:
        print("Printing all arguments provided:")
        print("\tConfig data:")
        for key, value in cli.get_config().data.items():
            print(f"\t- {key}: {value}")

    device = cli.config.data.get("device", {"type": "cpu", "idx": 0})
    import jax

    # print("=" * 20)
    # print("=" * 20)
    # print(jax.devices(device["type"])[device["idx"]])
    # print("=" * 20)
    # print("=" * 20)

    # jax.config.update("jax_platform_name", device["type"])
    jax.config.update("jax_default_device", jax.devices(device["type"])[device["idx"]])

    # print("Starting QVarNet...")
    # print("*" * 20)
    # print("Using device:", jax.devices())
    # print("*" * 20)

    # Import here to avoid circular dependency
    from qvarnet.main import run_experiment as run_experiment_main

    # Use the old FileParser interface for compatibility with existing main.py

    print("Running experiment...")
    run_experiment_main(cli, profile=cli.config.data.get("profile", False))
