from functools import cached_property
import logging
import math
import numpy as np
from note_seq import sequences_lib, midi_io
from note_seq.protobuf.music_pb2 import NoteSequence
from pathlib import Path
import torch
from torch.utils.data import Dataset
from typing import List, Tuple

from dsvae.data.converter.groove import GrooveConverter


def note_sequence_to_tensor(
    note_sequence: NoteSequence,
    pitch_mapping: dict,
    meter: Tuple[int, int] = (4, 4),
):
    """Helper function to convert a NoteSequence protobuf to a numpy array."""
    quantized_sequence = sequences_lib.quantize_note_sequence(
        note_sequence, steps_per_quarter=meter[0]
    )
    converter = GrooveConverter(
        steps_per_quarter=meter[0],
        quarters_per_bar=meter[1],
        pitch_classes=pitch_mapping,
        humanize=True,
    )
    tensor = converter.to_tensors(quantized_sequence)
    return tensor


class NoteSequenceDataset(Dataset):
    minute = 60
    second = 60
    meter = (4, 4)
    bars_per_frame = 1
    sequence_length = bars_per_frame * 16

    def __init__(
        self,
        note_sequence: NoteSequence,
        filepath: Path,
        pitch_mapping: dict,
        pattern_shuffle: bool = True,
        scale_factor: int = 1,
    ):
        super(NoteSequenceDataset, self).__init__()

        self.filepath = filepath
        self.name = filepath.stem
        self.base_note_sequence = note_sequence
        self.qpm = note_sequence.tempos[0].qpm
        self.pitch_mapping = pitch_mapping
        self.channels = len(pitch_mapping)

        self.pattern_shuffle = pattern_shuffle
        self.scale_factor = scale_factor

        if ((not self.pattern_shuffle) & (self.scale_factor > 1)):
            logging.getLogger(__name__).warning(
                "Scaling the dataset without shuffling is the same as duplicating the dataset - "
                "forcing scale_factor to 1."
            )
            self.scale_factor = 1
        elif self.scale_factor < 1:
            logging.getLogger(__name__).warning(
                "Scaling the dataset by less than 1 is not allowed - "
                "forcing scale_factor to 1."
            )
            self.scale_factor = 1

        try:
            self.meter = note_sequence.meter
        except AttributeError:
            pass

        self._meiosis()

    @classmethod
    def from_midi(cls, filepath: Path, pitch_mapping: dict, pattern_shuffle: bool, scale_factor: int):
        """
        Construct NoteSequence class directly from a MIDI file.
        """
        with open(filepath, "rb") as f:
            note_sequence = midi_io.midi_to_note_sequence(f.read())
            return cls(note_sequence, filepath, pitch_mapping, pattern_shuffle, scale_factor)

    def _meiosis(self):
        """We want to replicate using an evolutionary algorithm which uses
        all original 1-bar sequences as parents.

        *WARNING* In some cultures this could be considered incestuous.
        """
        self.data = []
        tensor = note_sequence_to_tensor(self.base_note_sequence, self.pitch_mapping)

        if len(tensor.inputs) == 0:
            logging.getLogger(__name__).warning(f"Failed loading {self.filepath}")
            pass
        else:
            inputs = tensor.inputs[0].reshape(
                (-1, self.sequence_length, self.channels * 3)
            )
            targets = tensor.outputs[0].reshape(
                (-1, self.sequence_length, self.channels * 3)
            )

            for i, input_frame in enumerate(inputs):
                for gen in range(self.scale_factor):
                    if self.pattern_shuffle:
                        np.random.shuffle(targets)
                    target_frame = targets[i]
                    sample = (
                        torch.tensor(input_frame, dtype=torch.float),
                        torch.tensor(target_frame, dtype=torch.float),
                    )
                    self.data.append(sample)

    def __getitem__(self, idx):
        inputs, targets = self.data[idx]
        return inputs, targets, self.name, idx

    def __len__(self):
        return len(self.data)

    @cached_property
    def valid_meter(self) -> bool:
        if (self.meter[0] == 4) & (self.meter[0] == 4):
            return True
        else:
            return False

    @cached_property
    def frame_lengths(self) -> List[int]:
        """
        Returns:
            List of start and end times of N frames of length
            self.bars_per_frame.
        """
        # frame_size = self.bars_per_frame * self.duration_bars
        num_sequences = int(self.duration_bars)

        frame_lengths = [0.0]
        for i in range(1, num_sequences + 1):
            frame_lengths.append(i * self.seconds_per_bar)
        return frame_lengths

    @cached_property
    def seconds_per_bar(self) -> float:
        """Time in seconds of one bar"""
        bars_per_minute = self.qpm / self.meter[0]
        return 1 / (bars_per_minute / self.minute)

    @cached_property
    def duration_bars(self) -> float:
        """Length of `note_sequence` in seconds or bars
        Args:
            bars: Whether to return the duration in seconds or bars.
        """
        return math.ceil(self.base_note_sequence.total_time / self.seconds_per_bar)

    @cached_property
    def duration_seconds(self) -> float:
        """Length of `note_sequence` in seconds or bars
        Args:
            bars: Whether to return the duration in seconds or bars.
        """
        return float(self.base_note_sequence.total_time)
