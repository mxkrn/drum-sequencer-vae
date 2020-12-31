from dsvae.models.encoder import LSTMEncoder

import torch


def test_lstm_encoder(sample, hparams, channels):
    # default - 2 layers - bidirectional
    hparams.bidirectional = True
    hparams.n_layers = 2
    hparams.hidden_factor = 4
    hparams.input_size = channels*3
    encoder = LSTMEncoder(hparams)
    hidden = encoder(sample[0])

    # torch.all(torch.eq(sample[0], inputs))
    assert hidden[0].shape == torch.Size([sample[0].shape[0], hparams.hidden_size * hparams.hidden_factor])
    assert hidden[1].shape == torch.Size([sample[0].shape[0], hparams.hidden_size * hparams.hidden_factor])

    # bidirectional false
    hparams.bidirectional = False
    hparams.hidden_factor = 2
    encoder = LSTMEncoder(hparams)
    hidden = encoder(sample[0])

    # torch.all(torch.eq(sample[0], inputs))
    assert hidden[0].shape == torch.Size([sample[0].shape[0], hparams.hidden_size * hparams.hidden_factor])
    assert hidden[1].shape == torch.Size([sample[0].shape[0], hparams.hidden_size * hparams.hidden_factor])

    # one layer
    hparams.hidden_factor = 1
    hparams.n_layers = 1
    encoder = LSTMEncoder(hparams)
    hidden = encoder(sample[0])

    # torch.all(torch.eq(sample[0], inputs))
    assert hidden[0].shape == torch.Size([sample[0].shape[0], hparams.hidden_size * hparams.hidden_factor])
    assert hidden[1].shape == torch.Size([sample[0].shape[0], hparams.hidden_size * hparams.hidden_factor])
