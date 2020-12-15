import torch.nn as nn


class LSTMDecoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.input_size = hparams.input_size
        self.latent_size = hparams.latent_size
        self.hidden_size = hparams.hidden_size
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
        )
        self.from_latent = nn.Linear(self.latent_size, self.hidden_size * self.n_layers)
        self.lstm.flatten_parameters()

    def forward(self, inputs, hidden):
        output, hidden = self.lstm(inputs, hidden)
        return hidden, inputs
