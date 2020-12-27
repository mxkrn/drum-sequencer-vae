import abc
import numpy as np
import torch.nn as nn


class Mutation:
    def __call__(self, tensor: np.ndarray):
        return self.call(tensor)

    @abc.abstractmethod
    def call(self, tensor: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NoteDropout(nn.Module):

    def __init__(self, channels: int, sequence_length: int):
        super().__init__()

    def forward(self, input: torch.Tensor, p: float) -> torch.Tensor:
        ones = torch.ones(input.shape, device=input.device)
        zeros = torch.zeros(input.shape, device=input.device)
        mask = np.random.random_sample(input.shape)
        mask = np.where(mask > p, mask, zeros)
        return torch.mul(input, mask)


class MetricNoteDropout(nn.Module):
    """Invokes a dropout operation on any note of the target sequence"""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def call(self, tensor: np.ndarray) -> np.ndarray:
        weights = self._init_weights(tensor)
        metric_profile = self._get_metric_profile(weights)
        weights = np.multiply(weights, metric_profile)
        return weights

    def _init_weights(self, tensor):
        weights = np.ones(
            tensor[:, 0, : self.channels].unsqueeze(1).shape, device=tensor.device
        )  # onsets-only
        batch_first = np.transpose(weights, 0, 1)
        return np.transpose(batch_first, 1, 2)

    def _get_metric_profile(self, tensor, weights):
        p = np.array([1.8, 0.4, 0.8, 0.4, 1.5, 0.8, 1.0, 0.4])
        ch = np.concatenate([p, p])
        profile = weights.clone()
        profile = np.concatenate([ch, ch, ch, ch, ch, ch, ch, ch, ch])
        return np.reshape(profile, weights.shape)


class InstrumentDropout(Mutation):
    """Drops one instrument from the target sequence."""

    def __init__(self, probability=0.5, weights=None):
        super(InstrumentDropout, self).__init__()
        # TODO: Determine weights
        if weights is None:
            self.weights = {
                0: 0.1,
                1: 0.05,
                2: 0.1,
                3: 0.15,
                4: 0.1,
                5: 0.1,
                6: 0.1,
                7: 0.1,
                8: 0.1,
            }

    def call(self, tensor: np.ndarray) -> np.ndarray:

        pass


# class Normalize(Operation):
#     def __init__(self):
#         super(Normalize, self).__init__()

#     def call(self, note_sequence):
#         pass


# class VelocityScaling(Operation):
#     """Invokes a scaling operation on the velocity values"""

#     def __init__(self, mu=95, stds=3):
#         super(VelocityScaling, self).__init__()

#         self.mu = mu
#         self.stds = stds

#     def call(self, note_sequence):
#         pass


# class OffsetsScaling(Operation):
#     """Invokes a scaling operation on the microtiming values."""

#     def __init__(self, stds=3):
#         super(OffsetsScaling, self).__init__()
#         self.stds = stds

#     def call(self, note_sequence):
#         pass
