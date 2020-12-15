from dsvae.models.encoder import LSTMEncoder
from dsvae.models.decoder import LSTMDecoder
# from dsvae.models.vae import VAE

import torch


def test_encoder(sample, hparams, channels):
    hparams.input_size = channels*3
    encoder = LSTMEncoder(hparams)
    hidden, inputs = encoder(sample[0])

    torch.all(torch.eq(sample[0], inputs))
    assert hidden[0].shape == torch.Size([sample[0].shape[1], hparams.hidden_size * hparams.hidden_factor])
    assert hidden[1].shape == torch.Size([sample[0].shape[1], hparams.hidden_size * hparams.hidden_factor])


# def test_decoder(sample, hparams, channels):
#     hparams.input_size = channels*3
#     encoder = LSTMEncoder(hparams)
#     (gate, cell), inputs = encoder(sample[0])
#     gate = gate.view(
        
#     )
#     decoder = LSTMDecoder(hparams)
#     out = decoder(sample[0], (gate, cell))

#     torch.all(torch.eq(sample[0], inputs))
#     assert hidden[0].shape == torch.Size([sample[0].shape[1], hparams.hidden_size * hparams.hidden_factor])
#     assert hidden[1].shape == torch.Size([sample[0].shape[1], hparams.hidden_size * hparams.hidden_factor])
