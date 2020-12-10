from functools import cached_property
from itertools import chain, cycle
import json
from pathlib import Path
from typing import Iterable, List, Dict, Tuple
from torch.utils.data import IterableDataset

from dsvae.utils.ops import init_logger
from dsvae.data.augmentor.note_sequence import NoteSequenceHandler, sequence_to_tensor


class MidiStreamDataset(IterableDataset):
    def __init__(self, name: str, load_path: Path, pitch_map_filename: str = "pitch_mapping.json"):
        """This class is an iterable dataset to handle streaming MIDI data from disk

        Args:
            name: Name of MIDI dataset
            load_path: Top-level directory of unaugmented MIDI files
            pitch_map_fname: File name of JSON file containing MIDI pitch to index mapping
        """
        super(MidiStreamDataset, self).__init__()
        self.logger = init_logger()

        self.load_path = load_path / Path(name)
        assert self.load_path.is_dir(), f"invalid directory {load_path}"

        pitch_map_abspath = self.load_path / Path(pitch_map_filename)
        assert (
            pitch_map_abspath.is_file()
        ), f"invalid file to pitch_map_filename {pitch_map_abspath}"

        self.ignore_files = []
        self._ops = []

    @cached_property
    def files(self) -> Iterable[List[str]]:
        for f in self.load_path.glob("**/*.mid"):
            yield f

    @cached_property
    def pitch_mapping(self) -> Dict[str, int]:
        """Load the MIDI pitches to index mapping, you must pass a path
        Args:
            filepath: Absolute path to JSON containing the mapping from MIDI pitch to index.
        """
        try:
            for fname in self.load_path.glob("**/*.json"):
                with open(fname, "r") as f:
                    return json.load(f)
        except FileNotFoundError:
            self.logger.error("please specify a valid path to pitch_mapping.json")

    def valid_meter(self, note_sequence, fname: str) -> bool:
        """
        Currently only supports 4/4 meter
        """
        if fname in self.ignore_files:  # use cache for speed-ups
            return False
        try:
            ts = note_sequence.meter
        except AttributeError:
            self.logger.warning(f"{fname} does not contain time signature information.")
            return False
        if (ts[0] == 4) & (ts[0] == 4):
            return True
        else:
            self.ignore_files.append(fname)
            return False

    def __iter__(self):
        return self._stream(self.files)

    def _stream(self, files):
        return chain.from_iterable(map(self._parse_midi, cycle(files)))

    def _parse_midi(self, midi_file: Path) -> Tuple[NoteSequenceHandler, Path]:
        """Parse data from a MIDI-encoded file
        Args:
            midi_file: Path to MIDI file
        Returns:
            Tuple of NoteSequence and the corresponding midi_file_path
        """
        note_sequence = NoteSequenceHandler.from_midi(midi_file)
        if self.valid_meter(note_sequence, midi_file):
            for frame_idx, seq in enumerate(note_sequence.frames):
                tensor = sequence_to_tensor(seq, self.pitch_mapping["pitches"])
                inputs = tensor.inputs[0]
                targets = tensor.outputs[0]
                yield (inputs, targets, frame_idx, midi_file.name)

    @cached_property
    def __len__(self):
        count = 0
        for f in self.load_path.glob("**/*.mid"):
            count += 1
        return count


class GMDMidiStreamDataset(MidiStreamDataset):
    def __init__(self, load_path: Path):
        super(GMDMidiStreamDataset, self).__init__(
            name="gmd",
            load_path=load_path
        )
