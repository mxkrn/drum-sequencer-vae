import numpy as np
from numpy.testing import assert_almost_equal

from dsvae.data.augmentor.note_sequence import NoteSequenceHandler, sequence_to_tensor


def test_sequence_handler_properties(files):

    for midi_file in files:
        sequence_handler = NoteSequenceHandler.from_midi(midi_file)
        assert sequence_handler.minute == 60
        assert sequence_handler.second == 60
        assert sequence_handler.meter == (4, 4)
        assert sequence_handler.bars_per_frame == 1
        assert sequence_handler.qpm > 0
        assert type(sequence_handler.qpm) == float
        assert sequence_handler.qpm == 120
        assert sequence_handler.seconds_per_bar == 2.0


def test_monophonic_1bar(files):
    for midi_file in files:
        if midi_file.stem == "M-1":  # monophonic 1-bar
            sequence_handler = NoteSequenceHandler.from_midi(midi_file)
            assert sequence_handler.frame_lengths == [0.0, 2.0]
            assert round(sequence_handler.duration_seconds, 5) == 1.55729

            notes = sequence_handler.base_note_sequence.notes
            assert len(notes) == 4
            for i, note in enumerate(notes):
                if i == 0:
                    duration = note.end_time
                assert note.pitch == 36
                assert note.velocity == 100
                delta = note.end_time - note.start_time
                assert_almost_equal(delta, duration)


def test_polyphonic_1bar(files):
    for midi_file in files:
        if midi_file.stem == "P-1":  # polyphonic 1-bar
            sequence_handler = NoteSequenceHandler.from_midi(midi_file)
            # test length
            assert sequence_handler.frame_lengths == [0.0, 2.0]
            assert round(sequence_handler.duration_seconds, 5) == 1.93229

            # test notes
            notes = sequence_handler.base_note_sequence.notes
            assert len(notes) == 22
            pitches = []
            for i, note in enumerate(notes):
                # test correct pitches in sequence
                if note.pitch not in pitches:
                    pitches.append(note.pitch)

                if i == 0:
                    duration = note.end_time
                assert note.velocity == 100
                delta = note.end_time - note.start_time
                assert_almost_equal(delta, duration)
            assert pitches == [36, 44, 39]


def test_polyphonic_multiple_bars(files):
    for midi_file in files:
        file_hash = midi_file.stem.split("-")
        length_in_bars = file_hash[-1]
        if length_in_bars == str(2):
            sequence_handler = NoteSequenceHandler.from_midi(midi_file)
            assert sequence_handler.duration_bars <= 2
            assert sequence_handler.duration_bars > 1.8
            assert sequence_handler.frame_lengths == [0.0, 2.0, 4.0]

            count = 0
            for f in sequence_handler.frames:
                count += 1
            assert count == 2
        if length_in_bars == str(1.5):
            sequence_handler = NoteSequenceHandler.from_midi(midi_file)
            assert sequence_handler.frame_lengths == [0.0, 2.0, 4.0]
            assert len(sequence_handler.base_note_sequence.notes) == 6
            assert round(sequence_handler.duration_seconds, 5) == 2.55208


def test_split_into_frames(files):
    for midi_file in files:
        if midi_file.stem == "M-1.5":
            m1half_sequence_handler = NoteSequenceHandler.from_midi(midi_file)
            for i, f in enumerate(m1half_sequence_handler.frames):
                if i == 1:
                    assert len(f.notes) == 2
                if i == 0:
                    assert len(f.notes) == 4
        if midi_file.stem == "M-1":
            sequence_handler = NoteSequenceHandler.from_midi(midi_file)
            for frame in sequence_handler.frames:
                assert frame.notes == sequence_handler.base_note_sequence.notes


def test_sequence_to_tensor(files, pitch_mapping):
    for midi_file in files:
        sequence_handler = NoteSequenceHandler.from_midi(midi_file)

        for seq in sequence_handler.frames:
            tensor = sequence_to_tensor(seq, pitch_mapping["pitches"])
            inputs = tensor.inputs[0]
            targets = tensor.outputs[0]
            assert inputs.sum() == len(seq.notes)
            onsets = inputs[:, :9]
            inputs_vo = inputs[:, 9:]
            velocities = targets[:, 9:18]
            offsets = targets[:, 18:27]

            assert_almost_equal(onsets, targets[:, :9])
            assert_almost_equal(inputs_vo, np.zeros(inputs_vo.shape))
            assert velocities.sum() > 0
            assert velocities.sum() < onsets.sum()
            for step in range(len(onsets)):
                for channel in range(len(onsets[0])):
                    assert onsets[step][channel] >= velocities[step][channel]
                    if onsets[step][channel] == 1.:
                        assert velocities[step][channel] > 0.
                    else:
                        assert velocities[step][channel] == 0.
                        assert offsets[step][channel] == 0.

        # TODO: Write tests for specific fixture instances
        # file_hash = midi_file.stem.split("-")
