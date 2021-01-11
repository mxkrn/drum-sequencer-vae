from argparse import ArgumentParser
from IPython import get_ipython
import logging
import os
from pathlib import Path
from typing import Dict, Any
import yaml


class HParams(dict):
    def __init__(self, *args, **kwargs):
        super(HParams, self).__init__(*args, **kwargs)
        self.__dict__ = self
        self._parse()
        self._check()

    @classmethod
    def from_yaml(cls, filepath) -> Dict[str, Any]:
        """Construct HParams from hparams.yml."""
        with open(filepath, "r", encoding="utf-8") as f:
            hparams = yaml.load(f, Loader=yaml.FullLoader)
            return cls(hparams)

    def _parse(self):
        """Parse arguments if run in python shell."""
        if (get_ipython().__class__.__name__ == "NoneType") & (
            not bool(int(os.environ["DEBUG"]))
        ):
            parser = ArgumentParser()
            for key, value in self.items():
                parser.add_argument(
                    f"--{key}", type=type(value), default=value, required=False
                )
            args = parser.parse_args()
            for key, value in args.__dict__.items():
                self[key] = value
        else:
            logging.getLogger(__name__).info("Using default hyperparameters")

    def _check(self):
        """Check hyperparameters and generate dependent variables."""
        self["hidden_factor"] = (
            2 * self.n_layers if self.bidirectional else 1 * self.n_layers
        )
        self["input_size"] = self.input_size
        self["input_shape"] = (self.sequence_length, self.input_size)
        if self.max_anneal > self.epochs:
            self.max_anneal = self.epochs
            logging.getLogger(__name__).warning(
                "max_anneal is greater than epochs - forcing max_anneal=epochs"
            )
        if self.warm_latent > self.epochs:
            self.warm_latent = self.epochs
            logging.getLogger(__name__).warning(
                "warm_latent is greater than epochs - forcing warm_latent =epochs"
            )
