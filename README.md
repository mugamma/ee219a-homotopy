# A solver of non-linear systems of equations using homotopy continuation methods

This is my final project for a [graduate course on numerical simulation and modeling at UC Berkeley](https://www2.eecs.berkeley.edu/Courses/EE219A/). The code provides an interface for the implementation of homotopy continuation methods with the Euler-Newton method (forward Euler predictor + Newton's corrector) implemented for reference. Example usage is provided in `schmitt_trigger.py` and `schmitt_trigger.ipynb` with the latter explaining the usage and implementation in  detail.

[`autograd`](https://github.com/HIPS/autograd) is used for automatic differentiation.
