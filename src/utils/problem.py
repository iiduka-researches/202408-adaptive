import numpy as np
from abc import ABC, abstractmethod
from manifolds import Manifold


class Problem(ABC):
    def __init__(self, manifold: Manifold) -> None:
        self.manifold = manifold

    @abstractmethod
    def f(self, U):
        ...

    @abstractmethod
    def egrad(self, U, idx: int):
        ...

    @abstractmethod
    def minibatch_grad(
        self,
        U: np.ndarray,
        batch_size: int
    ) -> np.ndarray:
        ...

    @abstractmethod
    def full_grad(
        self,
        U: np.ndarray,
    ) -> np.ndarray:
        ...
