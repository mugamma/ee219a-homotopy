import numpy as np
import matplotlib.pyplot as plt
from homotopy import HomotopyTrack
from predictor_corrector import EulerNewton
from stepadj import ConstantStep
from convergence import HomotopyConvCriterion, FailedToConverge

class SchmittTriggerHomotopyTrack(HomotopyTrack):
    def __init__(self, k=10):
        self.k = k
        self._prev_tan = np.array([0, 1])

    def initial_point(self):
        return np.array([1, -0.8])

    def final_param(self):
        return 0.8

    def tangent(self, point):
        grad = self.gradient(point)
        tan = np.array([-grad[1], grad[0]]) / np.linalg.norm(grad)
        tan *= np.sign(np.dot(self._prev_tan, tan))
        self._prev_tan = tan
        return tan

    def homotopy(self, point):
        x, lambda_ = point[:-1], point[-1]
        return self.f(x, lambda_)

    def gradient(self, point):
        x, lambda_ = point[:-1], point[-1]
        return np.array([self.df_dvo(x, lambda_), self.df_dvi(x, lambda_)]).flatten()

    def f(self, vo, vi):
        return np.tanh(self.k * (vo/2 - vi)) - vo

    def df_dvi(self, vo, vi):
        return -self.k * (1 - np.tanh(self.k * (vo/2 - vi))**2)

    def df_dvo(self, vo, vi):
        return self.k/2 * (1 - np.tanh(self.k * (vo/2 - vi))**2) - 1

class SimpleCorrectorConvCriterion(HomotopyConvCriterion):
    def __init__(self, tol, maxiters):
        self.tol = tol
        self.maxiters = maxiters
        self._iter = 0

    def reset(self):
        self._iter = 0

    def converged(self, point, track):
        if self._iter > self.maxiters:
            raise FailedToConverge('corrector failed to converge')
        self._iter += 1
        u = track.homotopy(point)
        return abs(u) < self.tol

class SimpleTracerConvCriterion(HomotopyConvCriterion):
    def __init__(self, tol, maxiters):
        self.tol = tol
        self.maxiters = maxiters
        self._iter = 0
        self.sol_arc = []

    def reset(self):
        self._iter = 0
        self.sol_arc = []

    def converged(self, point, track):
        if self._iter > self.maxiters:
            raise FailedToConverge('tracer failed to converge')
        self._iter += 1
        self.sol_arc.append(point)
        return point[-1] > track.final_param()

if __name__ == '__main__':
    stepsize = ConstantStep(1e-2)
    trace_conv_criterion = SimpleTracerConvCriterion(1e-6, 10000)
    corrector_conv_criterion = SimpleCorrectorConvCriterion(1e-6, 1000)
    tracer = EulerNewton(stepsize, trace_conv_criterion,\
                    corrector_conv_criterion)
    schmitt_trigger_track = SchmittTriggerHomotopyTrack()
    tracer.trace(schmitt_trigger_track)
    vos, lambdas = tuple(zip(*trace_conv_criterion.sol_arc))
    plt.plot(lambdas, vos)
    plt.show()
