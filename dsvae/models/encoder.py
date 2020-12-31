import torch
import torch.nn as nn
from typing import Tuple


class LSTMEncoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.input_size = hparams.input_size
        self.input_shape = hparams.input_shape
        self.hidden_size = hparams.hidden_size
        self.hidden_factor = hparams.hidden_factor
        self.n_layers = hparams.n_layers
        self.dropout = hparams.lstm_dropout
        self.bidirectional = hparams.bidirectional

        self._build()

    def _build(self) -> None:
        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.n_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        self.lstm.flatten_parameters()

    def forward(self, input: torch.Tensor) -> Tuple[Tuple[torch.Tensor], torch.Tensor]:
        output, (gate, cell) = self.lstm(input)
        gate = gate.view(input.shape[0], self.hidden_factor * self.hidden_size)
        cell = cell.view(input.shape[0], self.hidden_factor * self.hidden_size)
        return (gate, cell)
