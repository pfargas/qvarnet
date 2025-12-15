import json

import jax
import qvarnet
import sys
import os
import argparse


class FileParser:
    def __init__(self, filename):
        self.filename = filename
        self.args = None

    def parse(self):
        with open(self.filename, "r") as file:
            self.args = json.load(file)

        return self.args

    @property
    def get_args(self):
        return self.args

    @property
    def get_optimizer_args(self):
        return self.args.get("optimizer", {})

    @property
    def get_training_args(self):
        return self.args.get("training", {})

    @property
    def get_model_args(self):
        return self.args.get("model", {})

    @property
    def get_sampler_args(self):
        return self.args.get("sampler", {})


def main():
    argument_parser = argparse.ArgumentParser("QVarNet CLI")
    argument_parser.add_argument(
        "command", type=str, help="Command to execute", choices=["run"], default="run"
    )
    argument_parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to use",
    )
    argument_parser.add_argument(
        "--filepath",
        "-f",
        type=str,
        default=f"{os.path.dirname(qvarnet.__file__)}/cli/parameters/hyperparams.json",
    )
    argument_parser.add_argument("--profile", "-p", action="store_true")
    args = argument_parser.parse_args()

    if args.command == "run":
        run(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


def run(args):
    jax.config.update("jax_platform_name", args.device)
    print("Starting QVarNet...")
    print("*" * 20)
    print("Using device:", jax.devices())
    print("*" * 20)
    print("Loading parameters from:", args.filepath)

    path = args.filepath
    file_parser = FileParser(path)
    file_parser.parse()
    print("Parameters loaded")

    print("Running experiment...")

    from qvarnet.main import run_experiment

    run_experiment(file_parser, profile=args.profile)
