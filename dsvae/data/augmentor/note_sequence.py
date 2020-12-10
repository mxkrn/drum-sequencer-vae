from functools import cached_property
from note_seq import sequences_lib, midi_io
import math
from pathlib import Path
from typing import Any, List, Iterable, Tuple

from dsvae.data.converter.groove import GrooveConverter


def sequence_to_tensor(sequence, pitch_classes: dict, meter: Tuple[int, int] = (4, 4)):

    quantized_sequence = sequences_lib.quantize_note_sequence(
        sequence, steps_per_quarter=meter[0]
    )
    converter = GrooveConverter(
        steps_per_quarter=meter[0],
        quarters_per_bar=meter[1],
        pitch_classes=pitch_classes,
        humanize=True
    )
    tensor = converter.to_tensors(quantized_sequence)
    return tensor


class NoteSequenceHandler:
    minute = 60
    second = 60
    meter = (4, 4)
    bars_per_frame = 1

    def __init__(self, note_sequence):
        self.base_note_sequence = note_sequence
        self.qpm = note_sequence.tempos[0].qpm

    @classmethod
    def from_midi(cls, file: str):
        """
        Construct NoteSequence class directly from a MIDI file.
        """
        with open(file, "rb") as f:
            note_sequence = midi_io.midi_to_note_sequence(f.read())
            return cls(note_sequence)

    @property
    def frames(self) -> Iterable[Any]:
        """
        Returns:
            note_sequence split into bars of length self.bars_split
        """
        for i in range(1, len(self.frame_lengths)):
            sequence = sequences_lib.extract_subsequence(
                self.base_note_sequence,
                self.frame_lengths[i - 1],
                self.frame_lengths[i]
                )
            yield sequence

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
    def duration_bars(self, unit="seconds") -> float:
        """Length of `note_sequence` in seconds or bars
        Args:
            bars: Whether to return the duration in seconds or bars.
        """
        return math.ceil(self.base_note_sequence.total_time / self.seconds_per_bar)

    @cached_property
    def duration_seconds(self, unit="seconds") -> float:
        """Length of `note_sequence` in seconds or bars
        Args:
            bars: Whether to return the duration in seconds or bars.
        """
        return float(self.base_note_sequence.total_time)

    def save(self, save_path: Path):
        midi_io.note_sequence_to_midi_file(self.base_note_sequence, save_path)
