import json
from functools import cached_property
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset

from dsvae.data.dataset import NoteSequenceDataset


class Loader:

    def __init__(
        self,
        path_to_data: Path,
        dataset_name: str,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        """Helper class that returns a DataLoader instance of a concatenated
        dataset of NoteSequenceDataset instances.

        Args:
            path_to_data: Path to a directory of datasets
            dataset_name: Name of dataset directory
            batch_size: Batch size
            shuffle: Shuffle data in loader
            num_workers: Number of concurrent data loader workers
        Returns:
            Instance of DataLoader.
        """
        self.path_to_data = path_to_data / Path(dataset_name)
        assert self.path_to_data.is_dir(), f"not a valid directory: {path_to_data}"

        self.midi_files = [x for x in path_to_data.glob("**/*.mid") if x.is_file()]
        assert len(self.midi_files) > 0

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.invalid_files = []
        self.channels = None  # will be set in _build
        self.sequence_length = None

        self._build()

    def _build(self):
        datasets = []
        for midi_file in self.midi_files:
            if midi_file in self.invalid_files:
                continue
            ds = NoteSequenceDataset.from_midi(midi_file, self.pitch_mapping["pitches"])
            if self.channels is None:
                self.channels = ds.channels
            if self.sequence_length is None:
                self.sequence_length = ds.sequence_length
            if ds.valid_meter:
                datasets.append(ds)
            else:
                self.invalid_files.append(midi_file)
                continue

        self.dataset = ConcatDataset(datasets)

    def __iter__(
        self,
    ) -> DataLoader:
        loader = DataLoader(
            self.dataset,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            drop_last=True,
        )
        yield from loader

    @cached_property
    def pitch_mapping(self) -> dict:
        for fname in self.path_to_data.glob("**/*.json"):
            with open(fname, "r") as f:
                return json.load(f)
