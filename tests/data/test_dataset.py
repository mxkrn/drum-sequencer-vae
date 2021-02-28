import numpy as np
from numpy.testing import assert_almost_equal
import torch

from dsvae.data.dataset import NoteSequenceDataset


def test_sequence_handler_properties(files, pitch_mapping):

    for midi_file in files:
        dataset = NoteSequenceDataset.from_midi(
            midi_file, pitch_mapping["pitches"], False, 1
        )
        assert dataset.minute == 60
        assert dataset.second == 60
        assert dataset.meter == (4, 4)
        assert dataset.bars_per_frame == 1
        assert dataset.qpm > 0
        assert type(dataset.qpm) == float
        assert round(dataset.qpm) in [120, 152]
        assert round(dataset.seconds_per_bar, 2) in [2.00, 1.58]


def test_monophonic_1bar(files, pitch_mapping):
    for midi_file in files:
        if midi_file.stem == "M-1":  # monophonic 1-bar
            dataset = NoteSequenceDataset.from_midi(
                midi_file, pitch_mapping["pitches"], False, 1
            )
            assert dataset.frame_lengths == [0.0, 2.0]
            assert round(dataset.duration_seconds, 5) == 1.55729

            notes = dataset.base_note_sequence.notes
            assert len(notes) == 4
            for i, note in enumerate(notes):
                if i == 0:
                    duration = note.end_time
                assert note.pitch == 36
                assert note.velocity == 100
                delta = note.end_time - note.start_time
                assert_almost_equal(delta, duration)


def test_polyphonic_1bar(files, pitch_mapping):
    for midi_file in files:
        if midi_file.stem == "P-1":  # polyphonic 1-bar
            dataset = NoteSequenceDataset.from_midi(
                midi_file, pitch_mapping["pitches"], False, 1
            )
            # test length
            assert dataset.frame_lengths == [0.0, 2.0]
            assert round(dataset.duration_seconds, 5) == 1.93229

            # test notes
            notes = dataset.base_note_sequence.notes
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


def test_polyphonic_multiple_bars(files, pitch_mapping):
    for midi_file in files:
        file_hash = midi_file.stem.split("-")
        length_in_bars = file_hash[-1]
        if length_in_bars == str(2):
            dataset = NoteSequenceDataset.from_midi(
                midi_file, pitch_mapping["pitches"], False, 1
            )
            assert dataset.duration_bars <= 2
            assert dataset.duration_bars > 1.8
            assert dataset.frame_lengths == [0.0, 2.0, 4.0]
            assert len(dataset.data) == 2
        if length_in_bars == str(1.5):
            dataset = NoteSequenceDataset.from_midi(
                midi_file, pitch_mapping["pitches"], False, 1
            )
            assert dataset.frame_lengths == [0.0, 2.0, 4.0]
            assert len(dataset.base_note_sequence.notes) == 6
            assert round(dataset.duration_seconds, 5) == 2.55208


def test_sequence_to_tensor(files, pitch_mapping):
    for midi_file in files:
        dataset = NoteSequenceDataset.from_midi(
            midi_file, pitch_mapping["pitches"], False, 1
        )

        for inputs, targets in dataset.data:
            # onsets = inputs[:, :9]
            inputs_vo = inputs[:, 9:]
            velocities = targets[:, 9:18]
            # offsets = targets[:, 18:27]

            # assert_almost_equal(onsets, targets[:, :9])
            assert_almost_equal(inputs_vo, np.zeros(inputs_vo.shape))
            assert velocities.sum() > 0
            # assert velocities.sum() < onsets.sum()
            # for step in range(len(onsets)):
            #     for channel in range(len(onsets[0])):
            #         assert onsets[step][channel] >= velocities[step][channel]
            #         if onsets[step][channel] == 1.0:
            #             assert velocities[step][channel] > 0.0
            #         else:
            #             assert velocities[step][channel] == 0.0
            #             assert offsets[step][channel] == 0.0

        # TODO: Write tests for specific fixture instances
        # file_hash = midi_file.stem.split("-")


def test_scale_factor_equal_one(files, pitch_mapping):
    scale_factor = 1
    for midi_file in files:
        file_hash = midi_file.stem.split("-")
        length_in_bars = file_hash[-1]

        dataset = NoteSequenceDataset.from_midi(
            midi_file, pitch_mapping["pitches"], True, scale_factor
        )


def test_dataset_pattern_scale(files, pitch_mapping):
    for scale_factor in [2, 4, 30]:
        for midi_file in files:
            file_hash = midi_file.stem.split("-")
            length_in_bars = file_hash[-1]
            if length_in_bars == str(2):
                dataset = NoteSequenceDataset.from_midi(
                    midi_file, pitch_mapping["pitches"], True, scale_factor
                )
                assert dataset.duration_bars <= 2
                assert dataset.duration_bars > 1.8
                assert dataset.frame_lengths == [0.0, 2.0, 4.0]

                if scale_factor >= int(length_in_bars):
                    assert len(dataset.data) == 2 * dataset.duration_bars
                else:
                    assert len(dataset.data) == 2 * scale_factor

            if length_in_bars == str(1.5):
                dataset = NoteSequenceDataset.from_midi(
                    midi_file, pitch_mapping["pitches"], False, scale_factor
                )
                assert dataset.frame_lengths == [0.0, 2.0, 4.0]
                assert len(dataset.base_note_sequence.notes) == 6
                assert round(dataset.duration_seconds, 5) == 2.55208

                assert len(dataset.data) == 2 * 1

                for tensor_tuple in dataset.data:
                    assert torch.all(
                        torch.eq(tensor_tuple[0][:, :9], tensor_tuple[1][:, :9])
                    )


def test_long_pattern(files, pitch_mapping):
    for scale_factor in [4, 10, 30]:
        for midi_file in files:
            file_hash = midi_file.stem.split("-")
            length_in_bars = file_hash[-1]
            if float(length_in_bars) > 3:
                dataset = NoteSequenceDataset.from_midi(
                    midi_file, pitch_mapping["pitches"], True, scale_factor
                )
                assert dataset.duration_bars == int(length_in_bars)
                assert len(dataset.data) == dataset.duration_bars * dataset.scale_factor

                not_equal_count = 0
                for sample in dataset.data:
                    # check that input and target onsets are not 100% equal
                    if torch.any(torch.ne(sample[0][:, :9], sample[1][:, :9])):
                        not_equal_count += 1
                assert (
                    not_equal_count > 0
                ), "Something went wrong, all input and target patterns in this shuffled sample are equal"

            if float(length_in_bars) > 3:
                dataset = NoteSequenceDataset.from_midi(
                    midi_file, pitch_mapping["pitches"], False, scale_factor
                )
                assert dataset.duration_bars == int(length_in_bars)
                assert len(dataset.data) == dataset.duration_bars * dataset.scale_factor

                not_equal_count = 0
                for sample in dataset.data:
                    # check that input and target onsets are not 100% equal
                    if torch.any(torch.ne(sample[0][:, :9], sample[1][:, :9])):
                        not_equal_count += 1
                assert (
                    not_equal_count == 0
                ), "Something went wrong, all input and target patterns in this sample should be equal"
