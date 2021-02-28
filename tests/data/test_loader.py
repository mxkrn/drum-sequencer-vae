import pytest
from pathlib import Path
import torch

from dsvae.data.loader import NoteSequenceDataLoader, train_test_split


# TODO: More tests for:
# - file shuffling
# - worker loading
# - performance


def test_gmd_data_loader(path_to_data):
    batch_size = 4
    for split in ["train", "valid", "test"]:
        loader = NoteSequenceDataLoader(path_to_data, batch_size, split, False, False)
        length = len([x for x in loader]) * batch_size
        for batch in loader:
            assert batch[0].shape == torch.Size([batch_size, 16, 27])
            assert batch[1].shape == torch.Size([batch_size, 16, 27])


def test_train_test_split_unique(path_to_data):
    ds_per_key = 5
    files = []
    for i in range(ds_per_key):
        files.append(Path(f"/some/path/{i}.mid"))

    data_splits = {"train": 0.4, "valid": 0.3, "test": 0.3}

    split_files = train_test_split(files, splits=data_splits)

    train_names = set([ds.name for ds in split_files["train"]])
    valid_names = set([ds.name for ds in split_files["valid"]])
    test_names = set([ds.name for ds in split_files["test"]])
    assert len(train_names.intersection(valid_names)) == 0
    assert len(train_names.intersection(test_names)) == 0
    assert len(valid_names.intersection(test_names)) == 0


def test_train_test_split_invalid_splits():
    files = []
    for i in range(50):
        ds = Path(f"/some/path/{i}.mid")
        files.append(ds)
    # these splits are invalid
    data_splits_list = [
        {"train": 0.7, "valid": 0.1, "test": 0.1},  # AssertionError
        {"train": 0.8, "valid": 0.1, "test": 0.0},  # AssertionError
        {"train": 0.0, "valid": 1.1, "test": 0.0},  # AssertionError
    ]
    for data_splits in data_splits_list:
        with pytest.raises(AssertionError):
            train_test_split(files, data_splits)


def test_pattern_shuffle(path_to_data):
    batch_size = 1
    loader = NoteSequenceDataLoader(path_to_data, batch_size, "train", False, False)
    for input, target, _, _ in loader:
        assert torch.all(torch.eq(input[:, :, :9], target[:, :, :9]))
        assert torch.all(torch.eq(input[:, :, 9:], torch.zeros(input[:, :, 9:].size())))
        assert torch.any(torch.ne(target[:, :, 9:], input[:, :, 9:]))
