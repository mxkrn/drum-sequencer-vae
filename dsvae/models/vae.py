from __future__ import annotations
from enum import Enum
import torch
import torch.nn as nn
from typing import Tuple, Dict, Union

from dsvae.models.encoder import LSTMEncoder
from dsvae.models.decoder import LSTMDecoder
from dsvae.models.utils import NoteDropout


class TrainTask(Enum):
    GROOVE = 1
    SYNCOPATE = 2
    FILL = 3

    @staticmethod
    def from_str(label: str) -> TrainTask:
        try:
            return TrainTask[label.upper()]
        except KeyError:
            raise ValueError(f"Could not create TrainTask from label: {label}")

    def __str__(self):
        return self.name.lower()


class VAE(nn.Module):
    def __init__(self, hparams: Dict[str, Union[int, float, str]]):
        super().__init__()
        self.input_size = hparams.input_size
        self.latent_size = hparams.latent_size
        self.hidden_size = hparams.hidden_size
        self.hidden_factor = hparams.hidden_factor
        self.n_layers = hparams.n_layers
        self.channels = hparams.input_size // 3  # onsets, velocities, and offsets
        self.input_size = hparams.input_size
        self.sequence_length = hparams.sequence_length
        self.batch_size = hparams.batch_size
        self.task = TrainTask.from_str(hparams.task)

        self._build(hparams)

    def _build(self, hparams: Dict[str, Union[int, float, str]]) -> None:
        self.encoder = LSTMEncoder(hparams)

        # self.mu = nn.Linear(self.hidden_size * 2, self.latent_size)
        # self.logvar = nn.Linear(self.hidden_size * 2, self.latent_size)
        self.mu = nn.Linear(self.hidden_factor * self.hidden_size * 2, self.latent_size)
        self.logvar = nn.Linear(
            self.hidden_factor * self.hidden_size * 2, self.latent_size
        )
        self.from_latent = nn.Linear(
            self.latent_size, self.hidden_size * self.hidden_factor * 2
        )

        self.note_dropout = NoteDropout()
        self.decoder = LSTMDecoder(hparams)

        self.onsets_act = nn.Sigmoid()
        self.velocities_act = nn.Sigmoid()
        self.offsets_act = nn.Tanh()

        # self._apply(self._init_params)

    def forward(
        self,
        input: torch.Tensor,
        delta_z: torch.Tensor,
        teacher_force_ratio: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        mu, logvar = self._encode(input)

        z, z_loss = self._reparameterize(mu, logvar)
        z = torch.add(z, delta_z.to(z.device))  # Z-manipulation

        output, z = self._decode(z, input, teacher_force_ratio)
        onsets, velocities, offsets = self._activation(output, self.channels)

        # TODO: Output as onsets, velocities, offsets
        return onsets, velocities, offsets, z, z_loss

    def _encode(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        (gate, cell) = self.encoder(input)
        h = torch.cat((gate, cell), -1)  # concatenate gate and cell

        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def _reparameterize(self, mu, logvar) -> Tuple[torch.Tensor, torch.Tensor]:
        # Reparameterize from normal distribution
        eps = torch.randn_like(mu).detach().to(mu.device)
        z = (logvar.exp().sqrt() * eps) + mu

        # KL-divergence
        z_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / mu.size(0)
        return z, z_loss

    def _decode(
        self, z: torch.Tensor, input: torch.Tensor, teacher_force_ratio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # get hidden representation
        hidden = self.from_latent(z)
        hidden = hidden.view(self.hidden_factor, z.shape[0], self.hidden_size * 2)
        gate, cell = torch.split(hidden, self.hidden_size, -1)

        # TODO: Apply note dropout (teacher forcing)
        masked_input = self.note_dropout(input, teacher_force_ratio)
        output = self.decoder(masked_input, gate, cell)
        return output, z

    def _activation(
        self, output: torch.Tensor, channels: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # CHECK
        onsets, velocities, offsets = torch.split(output, channels, dim=-1)

        onsets = self.onsets_act(onsets)
        velocities = self.velocities_act(velocities)
        offsets = self.offsets_act(offsets)

        return onsets, velocities, offsets

    def _init_params(self, m) -> None:
        if type(m) == nn.Linear:
            mm.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
