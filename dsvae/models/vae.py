import torch
import torch.nn as nn

from dsvae.models.encoder import LSTMEncoder
from dsvae.models.decoder import LSTMDecoder
from dsvae.utils.ops import init_logger


class VAE(nn.Module):
    
    def __init__(self, hparams):
        super().__init__()
        self.logger = init_logger()
        self.input_size = hparams.input_size
        self.hidden_size = hparams.hidden_size
        self.hidden_factor = hparams.hidden_factor
        
        self.channels = hparams.channels

        self.encoder = LSTMEncoder(hparams)
        
        self.mu = nn.Linear(
            self.hidden_factor * self.hidden_size, self.latent_size)
        )
        self.logvar = nn.Linear(
            self.hidden_factor * self.n_layers, self.latent_size
        )
        self.from_latent = nn.Linear(
            self.latent_size, self.hidden_size * self.n_layers
        )
        self.decoder = LSTMDecoder(hparams)
        self.output_layer = nn.Linear(
            self.hidden_factor * self.hidden_size, self.input_size
        )

        self.onsets_act = nn.Sigmoid()
        self.velocities_act = nn.Sigmoid()
        self.offsets_act = nn.Tanh()

        self._apply(self._init_params)

    def _init_params(self, m):
        if type(m) == nn.Linear:
            mm.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, inputs: torch.Tensor, delta_z: torch.Tensor, note_dropout: float):
        mu, logvar = self._encode(inputs)

        z, z_loss = self._reparameterize(mu, logvar)
        z = torch.add(z, delta_z.to(z.device))  # Z-manipulation

        outputs, z = self._decoder(z, inputs, note_dropout.to(z.device))
        onsets, velocities, offsets = self._activation(outputs)

    def _encode(self, inputs: torch.Tensor):
        h, _ = self.encoder(inputs)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def _decode(self, z: torch.Tensor, inputs: torch.Tensor, note_dropout: float):
        hidden = self.from_latent(z)
        hidden = hidden.view(
            self.hidden_factor, z.shape[0], self.hidden_size
        )

        # TODO: Apply note dropout (teacher forcing)
        output, _ = self.deocder(inputs, hidden)
        output = self.output_layer(output)
        return output, z

    def _activation(self, outputs: torch.Tensor):
        onsets, velocities, offsets = torch.split(outputs, self.channels, dim=-1)

        onsets = self.onsets_act(onsets)
        velocities = self.velocities_act(velocities)
        offsets = self.offsets_act(offsets)

        return onsets, velocities, offsets
