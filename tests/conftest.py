import json
import pytest
from pathlib import Path


@pytest.fixture
def path_to_data() -> Path:
    return Path.cwd() / Path("tests/fixtures")


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
