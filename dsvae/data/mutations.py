import abc
import numpy as np


class Operation:
    def __call__(self, tensor: np.array):
        return self.call(tensor)

    @abc.abstractmethod
    def call(self, note_sequence):
        raise NotImplementedError


class Normalize(Operation):
    def __init__(self):
        super(Normalize, self).__init__()

    def call(self, note_sequence):
        pass


class VelocityScaling(Operation):
    """Invokes a scaling operation on the velocity values"""

    def __init__(self, mu=95, stds=3):
        super(VelocityScaling, self).__init__()

        self.mu = mu
        self.stds = stds

    def call(self, note_sequence):
        pass


class OffsetsScaling(Operation):
    """Invokes a scaling operation on the microtiming values."""

    def __init__(self, stds=3):
        super(OffsetsScaling, self).__init__()
        self.stds = stds

    def call(self, note_sequence):
        pass


class InstrumentDropout(Operation):
    """Invokes a weighted probability dropout operation on
    one instrument of the sequence.
    """

    def __init__(self, weights=None):
        super(InstrumentDropout, self).__init__()
        # TODO: Determine weights
        if weights is None:
            self.weights = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}

    def call(self, note_sequence):
        pass
