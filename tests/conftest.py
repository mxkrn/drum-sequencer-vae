import json
import pytest
from pathlib import Path

from dsvae.utils.hparams import get_hparams
from dsvae.utils.ops import init_logger
from dsvae.data.loader import NoteSequenceDataLoader


@pytest.fixture
def path_to_data() -> Path:
    return Path.cwd() / Path("tests/fixtures/gmd")


@pytest.fixture
def files(path_to_data):
    files = []
    for f in path_to_data.glob("**/*.mid"):
        files.append(f)
    return files


@pytest.fixture
def pitch_mapping(path_to_data):
    for filepath in path_to_data.glob("**/pitch_mapping.json"):
        with open(filepath, "r") as f:
            return json.load(f)


@pytest.fixture
def hparams():
    hparams = get_hparams()
    return hparams


@pytest.fixture
def sample(path_to_data):
    batch_size = 2
    loader = NoteSequenceDataLoader(path_to_data, batch_size, "train")
    batch = next(iter(loader))
    return batch


@pytest.fixture
def channels(path_to_data):
    batch_size = 1
    loader = NoteSequenceDataLoader(path_to_data, batch_size, "train")
    return loader.channels
