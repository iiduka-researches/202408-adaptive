import numpy as np

from .manifold import Manifold


class Stiefel(Manifold):
    def __init__(self, p: int, n: int) -> None:
        super(Stiefel, self).__init__()
        self.p = p
        self.n = n

    def __repr__(self) -> str:
        return f'St({self.p},{self.n})'

    @property
    def dim(self) -> int:
        return int(self.n * self.p - self.p * (self.p + 1) / 2)

    def projection(
        self, 
        point: np.ndarray,
        vector: np.ndarray
    ) -> np.ndarray:
        return vector - point @ sym(point.T @ vector)    

    def retraction(
        self, 
        point: np.ndarray,
        tangent_vector: np.ndarray
    ) -> np.ndarray:
        return np.linalg.qr(point + tangent_vector)[0]


def sym(m: np.ndarray) -> np.ndarray:
    return (m + m.T) / 2
