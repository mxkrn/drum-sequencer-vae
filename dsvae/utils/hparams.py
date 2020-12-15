from argparse import ArgumentParser
import logging
from typing import Dict, Any


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def parse(hparams):
    """
    Parse arguments
    """
    parser = ArgumentParser()
    for key, value in hparams.items():
        parser.add_argument(f"--{key}", type=type(value), default=value, required=False)
    args = parser.parse_args()
    for key, value in args.__dict__.items():
        hparams.add_argument(key, value)
    return hparams


def process(hparams):
    """Generate dependent variables in hyperparameters."""
    hparams["hidden_factor"] = (
        2 * hparams.n_layers if hparams.bidirectional else 1 * hparams.n_layers
    )
    return hparams


def get_hparams(logger: logging.Logger, **kwargs) -> Dict[str, Any]:
    hparams = AttrDict(
        dataset="gmd",
        nbworkers=0,
        batch_size=128,
        model="vae",
        bidirectional=True,
        n_layers=2,
        hidden_size=512,
        latent_size=2,
        lstm_dropout=0.1,
        beta_factor=1e4,
        gamma_factor=1,
        attention=False,
        disentangle=False,
        epochs=100,
        lr=1e-4,
        warm_latent=50,
        early_stop=50,
        device="",
    )
    # hparams = parse(hparams)
    hparams = process(hparams)
    return hparams
