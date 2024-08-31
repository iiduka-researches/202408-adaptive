import numpy as np

from .manifold import Manifold


class Grassmann(Manifold):
    def __init__(self, p: int, n: int) -> None:
        super(Grassmann, self).__init__()
        self.p = p
        self.n = n

    def __repr__(self) -> str:
        return f'Gr({self._p},{self._n})'

    @property
    def dim(self) -> int:
        return int(self.n * self.p - self.p ** 2)

    def projection(
        self, 
        point: np.ndarray,
        vector: np.ndarray
    ) -> np.ndarray:
        return vector - point @ (point.T @ vector) 

    def retraction(
        self, 
        point: np.ndarray,
        tangent_vector: np.ndarray
    ) -> np.ndarray:
        u, _, vt = np.linalg.svd(point + tangent_vector, full_matrices=False)
        return u @ vt
