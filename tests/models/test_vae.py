from dsvae.models.vae import VAE

import torch


def test_vae(sample, hparams, channels):
    # bidirectional - 2 layers

    hparams.bidirectional = True
    hparams.n_layers = 2
    hparams.hidden_factor = 4
    hparams.input_size = channels * 3

    vae = VAE(hparams)
    onsets, velocities, offsets, z, z_loss = vae(
        sample[0], torch.zeros(hparams.latent_size), torch.tensor(0.0)
    )
    output = torch.cat((onsets, velocities, offsets), -1)
    assert output.shape == sample[0].shape

    # bidirectional false
    hparams.bidirectional = False
    hparams.hidden_factor = 2

    vae = VAE(hparams)
    onsets, velocities, offsets, z, z_loss = vae(
        sample[0], torch.zeros(hparams.latent_size), torch.tensor(0.0)
    )
    output = torch.cat((onsets, velocities, offsets), -1)
    assert output.shape == sample[0].shape

    # one layer
    hparams.n_layers = 1
    hparams.hidden_factor = 1
    hparams.lstm_dropout = 0.0

    vae = VAE(hparams)
    onsets, velocities, offsets, z, z_loss = vae(
        sample[0], torch.zeros(hparams.latent_size), torch.tensor(0.0)
    )
    output = torch.cat((onsets, velocities, offsets), -1)
    assert output.shape == sample[0].shape
