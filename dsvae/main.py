from argparse import ArgumentParser
from functools import cached_property
import logging
import os
import wandb

from dsvae.utils import init_logger, HParams
from dsvae.train import train
from dsvae.evaluate import evaluate


class Client:
    def __init__(self, run_name: str = None, debug: bool = False):

        self.debug = debug
        if debug:
            os.environ["WANDB_MODE"] = "dryrun"
        if run_name is None:
            self.run = wandb.init(dir="outputs")
        else:
            raise NotImplementedError("Training run resuming is not yet implemented")
            # TODO: Implement restoring an existing run
            # self.run = wandb.restore(name, dir="outputs")

        if self.debug:
            self.logger = init_logger(logging.DEBUG)
            self.logger.warning("Running client in DEBUG mode")
        else:
            self.logger = init_logger(logging.INFO)

    @cached_property
    def hparams(self):
        if self.debug:
            hparams = HParams.from_yaml("config/debug.yml")
        else:
            hparams = HParams.from_yaml("config/default.yml")
        wandb.config.update(hparams, allow_val_change=True)
        return hparams

    @property
    def name(self):
        return self.run.name

    @property
    def savepath(self):
        path = Path(f"outputs/models/{self.run.name}")
        if path.is_dir():
            return path
        else:
            raise OSError(f"invalid model output directory {path}")

    def train(self):
        return train(self.name, self.hparams, self.logger)

    def resume(self):
        raise NotImplementedError

    def evaluate(self, run_name):
        return evaluate(self.name, self.logger)


if __name__ == "__main__":
    # initialize
    parser = ArgumentParser()
    parser.add_argument(
        "--train", default=True, action="store_true", help="Start a training run"
    )
    parser.add_argument("--resume", action="store_true", help="resume a training run")
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate a training run"
    )
    parser.add_argument(
        "--run_name",
        default=None,
        type=str,
        help="Name of training run if restoring or evaluating",
    )
    parser.add_argument("--debug", action="store_true", help="Run client in debug mode")
    args = parser.parse_args()

    client = Client(debug=args.debug)
    if args.debug == True:
        client.logger.warning("Setting DEBUG to True")
        os.environ["DEBUG"] = "1"
    else:
        os.environ["DEBUG"] = "0"

    if args.resume:
        if args.run_name is None:
            exit("You must pass the argument run_name to restore a training run")
        client.logger(f"Resuming run: {args.run_name}")
        client.resume(args.run_name)
    elif args.train:
        client.train()
    elif args.evaluate:
        if args.run_name is None:
            exit("You must pass the argument run_name to evaluate a training run")
        client.logger(f"Evaluating run: {args.run_name}")
        client.evaluate(run_name)
    else:
        exit("You must specify a job type")
