import autograd

class AutoHomotopyTrack(HomotopyTrack):    
    def __init__(self, homotopy, init_root=None, init_param=0, final_param=1):
        self._homotopy = homotopy
        self.param_range = (init_param, final_param)
        self.init_root = init_root
        self._prev_tan = self.__init_tan()

    def initial_point(self):
        if self._init_root is None:
            raise NotImplementedError
        return np.hstack([self._init_root, self.param_range[0]])

    def tangent(self, point):
        grad = self.gradient(point) 
        tan = __ker_unit_basis(grad)
        tan *= np.sign(np.dot(tan, self._prev_tan))
        self._prev_tan = tan
        return tan

    def homotopy(self, point):
        return self._homotopy(point[:-1], point[-1])

    def gradient(self, point):
        return autograd.jacobian(self.homotopy)(point)

    def __init_tan(self):
        n = len(self.init_root) if isinstance(self.init_root, np.ndarray) else 1
        init_tan = np.zeros(n)
        init_tan[-1] = np.sign(self.param_range[1] - self.param_range[0])
        return init_tan
    
    @staticmethod
    def __ker_unit_basis(mat):
        # XXX precond: mat.shape is (n, n+1) and rank(mat) = n
        _, __, u = scipy.linalg.lu(mat)
        ker_basis = np.zeros(mat.shape[1])
        ker_basis[-1] = 1
        for i in reversed(range(mat.shape[0])):
            ker_basis[i] = -np.dot(ker_basis, u[i, :]) / u[i, i]
        ker_basis /= np.linalg.norm(ker_basis)
        return ker_basis
