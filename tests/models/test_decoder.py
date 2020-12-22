from dsvae.models.decoder import LSTMDecoder
from dsvae.models.encoder import LSTMEncoder

import torch


def test_lstm_decoder(sample, hparams, channels):
    # default - bidirectional - 2 layers
    hparams.input_size = channels*3
    gate = torch.zeros([hparams.hidden_factor, sample[0].shape[0], hparams.hidden_size])
    cell = gate.clone()

    decoder = LSTMDecoder(hparams)
    out = decoder(sample[0], gate, cell)

    assert out.shape == sample[0].shape

    # bidirectional = False
    hparams.input_size = channels*3
    hparams.bidirectional = False
    hparams.hidden_factor = 2

    hparams.input_size = channels*3
    gate = torch.zeros([hparams.hidden_factor, sample[0].shape[0], hparams.hidden_size])
    cell = gate.clone()

    decoder = LSTMDecoder(hparams)
    out = decoder(sample[0], gate, cell)

    assert out.shape == sample[0].shape

    # one layer
    hparams.n_layers = 1
    hparams.hidden_factor = 1
    hparams.lstm_dropout = 0.
    hparams.input_size = channels*3
    gate = torch.zeros([hparams.hidden_factor, sample[0].shape[0], hparams.hidden_size])
    cell = gate.clone()
    
    decoder = LSTMDecoder(hparams)
    out = decoder(sample[0], gate, cell)

    assert out.shape == sample[0].shape
