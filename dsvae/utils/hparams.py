from argparse import ArgumentParser
import logging
import os
from yaml import load, Loader
from typing import Dict, Any, Optional

from dsvae.utils.debug import Debug


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_yaml(filename: str):
        with open(filename, "r") as f:
            data = load(f.read(), Loader=Loader)
        data_dict = AttrDict(**data)
        return data_dict


def parse(hparams):
    """
    Parse arguments
    """
    parser = ArgumentParser()
    if not parser.prog == "pytest":
        for key, value in hparams.items():
            if key == "debug":
                parser.add_argument("--debug", action="store_true", required=False)
            else:
                parser.add_argument(
                    f"--{key}", type=type(value), default=value, required=False
                )
        args = parser.parse_args()
        for key, value in args.__dict__.items():
            hparams[key] = value
    return hparams


def process(hparams):
    """Generate dependent variables in hyperparameters."""
    hparams["hidden_factor"] = (
        2 * hparams.n_layers if hparams.bidirectional else 1 * hparams.n_layers
    )
    hparams["input_size"] = hparams.channels * 3
    hparams["input_shape"] = (hparams.sequence_length, hparams.input_size)
    if hparams.max_anneal > hparams.epochs:
        hparams.max_anneal = hparams.epochs
        logging.getLogger(__name__).warning(
            "max_anneal is greater than epochs - forcing max_anneal=epochs"
        )
    if hparams.warm_latent > hparams.epochs:
        hparams.warm_latent = hparams.epochs
        logging.getLogger(__name__).warning(
            "warm_latent is greater than epochs - forcing warm_latent =epochs"
        )
    return hparams


def get_hparams(filename: Optional[str] = None) -> Dict[str, Any]:
    if filename is not None:
        logging.getLogger(__name__).info(f"Loading hparams from {filename}")
        hparams = AttrDict.from_yaml(filename)
    else:
        hparams = AttrDict(
            dataset="gmd",
            num_workers=0,
            batch_size=4,
            channels=9,  # number of instruments
            sequence_length=16,
            file_shuffle=True,  # shuffles data loading across different MIDI patterns
            pattern_shuffle=True,  # shuffle sub-patterns within a MIDI pattern
            scale_factor=2,
            model="vae",
            bidirectional=False,
            n_layers=2,
            hidden_size=512,
            latent_size=8,
            lstm_dropout=0.1,
            # teacher_force_ratio=0.0,
            beta=1e4,
            max_anneal=200,
            attention=False,
            disentangle=False,
            epochs=300,
            lr=1e-4,
            warm_latent=100,
            early_stop=30,
            device="",
            debug=False,
            task="groove",
        )
    hparams = parse(hparams)
    hparams = process(hparams)
    logging.getLogger(__name__).debug("HParams:")
    for k, v in hparams.items():
        logging.getLogger(__name__).debug(f"{k}: {v}")
    return hparams
