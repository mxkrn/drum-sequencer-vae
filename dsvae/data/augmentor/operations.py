import abc
from note_sequence import NoteSequenceHandler


class Operation:
    def __call__(self, note_sequence: NoteSequenceHandler):
        return self.call(note_sequence)

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

    def __init__(self, default=97, spread=30):
        super(VelocityScaling, self).__init__()

        self.default = default
        self.spread = spread
        self.range = [self.default - self.spread, self.default + self.spread]

    def call(self, note_sequence):
        pass


class MicrotimingScaling(Operation):
    """Invokes a scaling operation on the microtiming values."""

    def __init__(self):
        super(MicrotimingScaling, self).__init__()

    def call(self, note_sequence):
        pass


class InstrumentDropout(Operation):
    """Invokes a weighted probability dropout operation on
    one instrument of the sequence.
    """

    def __init__(self):
        super(InstrumentDropout, self).__init__()
        # TODO: Determine weights
        self.weights = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}

    def call(self, note_sequence):
        pass
