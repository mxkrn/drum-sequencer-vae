import pytest
from pathlib import Path

from dsvae.utils.hparams import HParams


def test_construct_hparams():
    filepath = Path("config/debug.yml")
    hparams = HParams.from_yaml(filepath)
    assert len(hparams.keys()) > 0
    for k, v in hparams.items():
        assert hparams[k] == v