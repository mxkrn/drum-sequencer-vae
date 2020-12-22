from dsvae.models.vae import VAE

import torch


def test_vae(sample, hparams, channels):
    # default - bidirectional - 2 layers
    hparams.input_size = channels*3
    vae = VAE(hparams, channels)
    
    output, z, z_loss = vae(sample[0], torch.zeros(hparams.latent_size), torch.tensor(0.))
    assert output.shape == sample[0].shape
