from argparse import ArgumentParser
import os
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
    return hparams


def get_hparams(**kwargs) -> Dict[str, Any]:
    hparams = AttrDict(
        dataset="gmd",
        num_workers=0,
        batch_size=4,
        channels=9,  # number of instruments
        sequence_length=16,
        file_shuffle=True,  # shuffles data loading across different MIDI patterns
        pattern_shuffle=True,  # shuffle sub-patterns within a MIDI pattern
        scale_factor=10,
        model="vae",
        bidirectional=False,
        n_layers=2,
        hidden_size=512,
        latent_size=8,
        lstm_dropout=0.1,
        # teacher_force_ratio=0.0,
        beta=1e4,
        max_anneal=100,
        attention=False,
        disentangle=False,
        epochs=250,
        lr=1e-4,
        warm_latent=50,
        early_stop=30,
        device="",
    )
    debug = bool(int(os.environ["_PYTEST_RAISE"]))
    if not debug:
        hparams = parse(hparams)
    hparams = process(hparams)
    return hparams
