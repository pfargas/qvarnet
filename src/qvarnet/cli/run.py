import json
import qvarnet
import sys
import os


class ArgumentParser:
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
    command = sys.argv[1]
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    print(arg)
    if command == "run":
        run(arg)


def run(arg):
    if arg is not None:
        print(f"Using argument file: {arg}")
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        f"{arg}"
        if "CUDA_VISIBLE_DEVICES" not in os.environ
        else os.environ["CUDA_VISIBLE_DEVICES"]
    )
    path = os.path.dirname(qvarnet.__file__) + "/cli/parameters/hyperparams.json"
    parser = ArgumentParser(path)
    parser.parse()
    optimizer_args = parser.get_optimizer_args
    print(optimizer_args)
    print(parser.get_training_args)

    from qvarnet.main import run_experiment

    run_experiment(parser)


if __name__ == "__main__":
    parser = ArgumentParser("./cli/parameters/hyperparams.json")
    parser.parse()
    optimizer_args = parser.get_optimizer_args
    print(optimizer_args)
    print(parser.get_training_args)

    from qvarnet.train import run_experiment

    run_experiment(parser)
