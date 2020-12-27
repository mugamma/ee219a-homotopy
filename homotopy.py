from typing import Tuple
import numpy as np
import scipy

class HomotopyTrack:
    """The Homotopy Track Interface

       A homotopy track consists of a homotopy function and the interval on
       which parameter continuation should take place. The homotopy parameter
       is not supplied to the methods separately and the last component of the
       `point` parameter is interpreted as the homotopy parameter.
    """

    def initial_point(self) -> np.ndarray:
        """Return the initial point on the continuation track."""
        raise NotImplementedError

    def tangent(self, point: np.ndarray) -> np.ndarray:
        """Return the tangent vector of the continuation track at `point`.

        The tangent should be a unit vector in the kernel of the gradient map.
        The direction of the tangent indicates the direction of continuation.
        """
        raise NotImplementedError

    def homotopy(self, point: np.ndarray) -> np.ndarray:
        """Return the value of homotopy at `point`."""
        raise NotImplementedError

    def gradient(self, point: np.ndarray) -> np.ndarray:
        """Return the value of the gradient map of the homotopy at `point`.

        Consistent with the convention that the last component of `point` is
        interpreted as the homotopy parameter, the last column of the matrix
        representing the gradient map corresponds to the partial derivatives
        with respect to the homotopy parameter.
        """
        raise NotImplementedError

    @property
    def param_range(self) -> Tuple[float, float]:
        """2-tuple indicating the interval of continuation"""
        raise NotImplementedError
