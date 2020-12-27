import numpy as np
from homotopy import HomotopyTrack

class ConvCriterion:
    """General Convergence Criterion"""
    def converged(self, *args) -> bool:
        """Return whether convergence is achieved."""
        raise NotImplementedError

    def reset(self):
        """Reset the state of the criterion tracker.

        Convergence criterion objects hold state, e.g. the number of iterations
        so far. This method resets the state of `self` so that it can be reused.
        """
        raise NotImplementedError

class FailedToConverge(Exception):
    """Exception Indicating Failure in Convergence"""
    pass

class HomotopyConvCriterion(ConvCriterion):
    """General Convergence Criterion for Homotopy Methods"""
    def converged(self, point, track: HomotopyTrack):
        raise NotImplementedError
