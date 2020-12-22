import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMDecoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.input_size = hparams.input_size
        self.latent_size = hparams.latent_size
        self.hidden_size = hparams.hidden_size
        self.output_size = hparams.input_size
        self.hidden_factor = hparams.hidden_factor
        self.n_layers = hparams.n_layers
        self.dropout = hparams.lstm_dropout
        self.bidirectional = hparams.bidirectional

        self._build()

    def _build(self):
        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.n_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        self.lstm.flatten_parameters()
        self.output_layer = nn.Linear(
            int(self.hidden_factor * self.hidden_size / self.n_layers), self.output_size
        )

    def forward(
        self, input: torch.Tensor, gate: torch.Tensor, cell: torch.Tensor
    ) -> torch.Tensor:
        gate = gate.view(self.hidden_factor, input.shape[0], self.hidden_size)
        cell = cell.view(self.hidden_factor, input.shape[0], self.hidden_size)
        output, _ = self.lstm(input, (gate, cell))
        output = self.output_layer(output)
        return output
