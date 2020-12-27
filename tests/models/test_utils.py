import torch

from dsvae.models.utils import NoteDropout


def test_note_dropout():
    note_dropout = NoteDropout()

    input = torch.ones((2, 16, 27))
    
    # should be equal to input - 100% teacher forcing
    ratio = torch.tensor(1.)
    output = note_dropout(input, ratio)
    assert torch.all(torch.eq(input, output))
    
    # should be all zero - 0% teacher forcing
    ratio = torch.tensor(0.)
    output = note_dropout(input, ratio)
    assert torch.all(torch.eq(output, torch.zeros(input.shape)))

    for ratio in [0.2, 0.4, 0.6, 0.8]:
        ratio = torch.tensor(ratio)
        output = note_dropout(input, ratio)
        total_active = output.sum()
        total = input.sum()
        assert (total_active / total) < ratio + 0.1
        assert (total_active / total) > ratio - 0.1
