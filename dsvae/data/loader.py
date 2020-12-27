import json
from functools import cached_property
import logging
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from typing import List, Dict

from dsvae.data.dataset import NoteSequenceDataset


logger = logging.getLogger(__name__)


class NoteSequenceDataLoader:
    def __init__(
        self,
        path_to_data: Path,
        batch_size: int,
        split=None,
        file_shuffle: bool = False,
        pattern_shuffle: bool = True,
        scale_factor: int = 1,
        num_workers: int = 0,
    ):
        """Helper class that returns a DataLoader instance of a concatenated
        dataset of NoteSequenceDataset instances.

        Args:
            path_to_data: Path to a directory of datasets
            dataset_name: Name of dataset directory
            batch_size: Batch size
            file_shuffle: Shuffle files in loader
            num_workers: Number of concurrent data loader workers
        Returns:
            Instance of DataLoader.
        """
        self.path_to_data = path_to_data  # / Path(dataset_name)
        assert self.path_to_data.is_dir(), f"not a valid directory: {path_to_data}"

        # self.midi_files = [x for x in path_to_data.glob("**/*.mid") if x.is_file()]
        # assert len(self.midi_files) > 0

        self.batch_size = batch_size
        self.file_shuffle = file_shuffle
        self.pattern_shuffle = pattern_shuffle
        self.scale_factor = scale_factor
        self.num_workers = num_workers
        self.invalid_files = []
        self.channels = None  # will be set in _build
        self.sequence_length = None  # will be set in _build

        self.splits = {"train": 0.8, "valid": 0.1, "test": 0.1}
        if split not in self.splits.keys():
            raise Exception(
                f"Invalid split {split} - must be one of {list(self.splits.keys())}"
            )
        else:
            self.split = split

        self._build()

    @cached_property
    def files(self):
        files = [x for x in self.path_to_data.glob("**/*.mid") if x.is_file()]
        if len(files) > 0:
            return files
        else:
            raise FileNotFoundError("Invalid path to MIDI data {self.path_to_data}")

    @cached_property
    def split_files(self):
        return train_test_split(self.files, self.splits)

    def _build(self):
        datasets = []
        for midi_file in self.split_files[self.split]:
            if midi_file in self.invalid_files:
                continue
            ds = NoteSequenceDataset.from_midi(midi_file, self.pitch_mapping["pitches"], self.pattern_shuffle, self.scale_factor)
            if self.channels is None:
                self.channels = ds.channels
            if self.sequence_length is None:
                self.sequence_length = ds.sequence_length
            if ds.valid_meter:
                datasets.append(ds)
            else:
                self.invalid_files.append(midi_file)
                continue

        self.datasets = datasets
        logger.info(
            f"Loaded {self.split} dataset with {len(self.dataset)} samples (~{self.splits[self.split]}%)"
        )

    def __iter__(
        self,
    ) -> DataLoader:
        loader = DataLoader(
            self.dataset,
            shuffle=self.file_shuffle,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            drop_last=True,
        )
        yield from loader

    @cached_property
    def dataset(self):
        # datasets_dict = train_test_split(self.datasets, self.splits)
        return ConcatDataset(self.datasets)

    @cached_property
    def pitch_mapping(self) -> dict:
        for fname in self.path_to_data.glob("**/*.json"):
            with open(fname, "r") as f:
                return json.load(f)


def train_test_split(
    files: List[Path], splits: Dict[str, float], seed=None
) -> Dict[str, List[Dataset]]:
    """Gets train, valid, and test splits of a list of paths

    Parameters
    ----------
    files
        List of paths to MIDI files. Each MIDI file will correspond to a single dataset.
    splits
        Dictionary of split ratio for each split type
    seed
        Set a fixed seed for reproducible data loading

    Returns
    ----------
    Dict of MIDI files split according to ratios in splits
    """
    assert sum(splits.values()) == 1.0, "Split ratio do not sum up to 1.0"

    # split_datasets = dict(train=[], valid=[], test=[])
    split_files_dict = dict(train=[], valid=[], test=[])
    # dataset_indices = np.arange(len(datasets))
    total_length = len(files)

    if seed is not None:
        np.random.seed(seed)
    # np.random.shuffle(dataset_indices)
    np.random.shuffle(files)

    for i, (key, split_ratio) in enumerate(splits.items()):
        if i < len(splits) - 1:  # sample w/o replacement
            split_length = round(split_ratio * total_length)
            # split_indices, dataset_indices = (
            #     dataset_indices[:split_length],
            #     dataset_indices[split_length:],
            # )
            split_files, files = (
                files[:split_length],
                files[split_length:]
            )
        else:  # take remainder
            # split_indices = dataset_indices
            split_files = files

        split_files_dict[key] = split_files
        # for idx in split_indices:
        #     split_datasets[key].append(datasets[idx])
        #     split_datasets[key].append()

    return split_files_dict


# def check_splits(split_datasets: Dict[str, List[Dataset]], total_length: int):
#     """Check and log some information on the dataset splits."""
#     train_length = len(split_datasets["train"])
#     valid_length = len(split_datasets["valid"])
#     test_length = len(split_datasets["test"])

#     logger.info(
#         f"Train split contains {train_length} files which is "
#         f"{train_length / total_length}% of the dataset"
#     )
#     logger.info(
#         f"Valid split contains {valid_length} files which is :"
#         f"{valid_length / total_length}% of the dataset"
#     )
#     logger.info(
#         f"Test split contains {test_length} files which is "
#         f"{test_length / total_length}% of the dataset"
#     )
