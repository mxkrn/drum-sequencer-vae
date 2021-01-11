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


# def test_gmd_dataset_constructor(path_to_data: Path, files: List[Path]):
#     ds = GMDMidiStreamDataset(path_to_data)

#     for fname in path_to_data.glob("**/*.json"):
#         with open(fname, "r") as f:
#             pitch_mapping = json.loads(f.read())
#     assert ds.pitch_mapping == pitch_mapping

#     for i, filepath in enumerate(islice(ds.files, 8)):
#         assert filepath.is_file()
#         assert filepath == files[i]

#     batch_size = 4
#     loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
#     frame_indices = torch.tensor([0, 0, 1, 2])
#     file1 = "182_afrocuban_105_fill_4-4.mid"
#     file2 = "183_afrocuban_105_beat_4-4.mid"
#     sample_names = (file1, file2, file2, file2)

#     for f in loader:
#         torch.all(torch.eq(f[2], frame_indices))
#         assert f[3] == sample_names
#         break
