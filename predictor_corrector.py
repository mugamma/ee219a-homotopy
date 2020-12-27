import numpy as np

from homotopy import HomotopyTrack
from stepadj import StepAdjuster
from convergence import HomotopyConvCriterion

class PredictorCorrectorMethod:
    """A General Predictor Corrector Method"""
    def __init__(self, step_adjuster: StepAdjuster,
            trace_conv_criterion: HomotopyConvCriterion,
            corrector_conv_criterion: HomotopyConvCriterion):
        self.step_adjuster = step_adjuster
        self.trace_conv_criterion = trace_conv_criterion
        self.corrector_conv_criterion = corrector_conv_criterion

    def predict(self, track: HomotopyTrack, cur_point: np.ndarray):
        """Predict and return the next point on the homotopy track."""
        raise NotImplementedError

    def correct(self, estimate: np.ndarray, track: HomotopyTrack):
        """Correct the supplied estimate of the given point on the track.
        
        The correct method should only implement a single correction step.
        Correction steps are applied until the corrected estimate is accurate
        enough as indicated by `self.corrector_conv_criterion`, or an error is
        raised.
        """
        raise NotImplementedError

    def trace(self, track: HomotopyTrack):
        """Trace the given homotopy track and return the final point."""
        point = track.initial_point()
        self.trace_conv_criterion.reset()
        while not self.trace_conv_criterion.converged(point, track):
            estimate = self.predict(track, point)
            self.corrector_conv_criterion.reset()
            while not self.corrector_conv_criterion.converged(estimate, track):
                estimate = self.correct(estimate, track)
            point = estimate
            self.step_adjuster.adjust(track, point)
        return point

class EulerNewton(PredictorCorrectorMethod):
    def predict(self, track: HomotopyTrack, point):
        """Return the next point according to the forward Euler predictor.

        The next point is estimated using a step similar to the step used in
        the forward version of Euler's method.
        """
        return point + self.step_adjuster.cur_step_size * track.tangent(point)

    def correct(self, estimate, track: HomotopyTrack):
        """Return the correction of the estimate using a Newton correction.

        The Newton family of correctors use the current estimate as the initial
        guess in an iterative method similar to Newton's method to solve for
        the corrected point. This method uses `estimate` as the initial guess
        in one step of Newton's method and uses the pseudo-inverse of the
        gradient map in the iterate function.
        """
        J = track.gradient(estimate)
        # make sure gradient vectors are expressed as 1xn matrices.
        if len(J.shape) == 1:
            J = J.reshape(1, -1)
        return estimate - np.linalg.pinv(J) @ track.homotopy(estimate)
