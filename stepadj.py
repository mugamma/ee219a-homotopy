import numpy as np

class StepAdjuster:
    """A General Time-step Controller"""
    def adjust(self, track, point):
        """Adjust the time step according to the current point on the track."""
        raise NotImplementedError

    @property
    def cur_step_size(self):
        """a float giving the current time step"""
        raise NotImplementedError

class ConstantStep(StepAdjuster):
    """A Time-step Controller with a Constant Step Size"""
    def __init__(self, h=1e-2):
        self.h = h

    def adjust(self, track, point):
        pass

    @property
    def cur_step_size(self):
        return self.h
