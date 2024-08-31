import numpy as np
from abc import ABC, abstractmethod


class Manifold(ABC):
    def __init__(self) -> None:
        ...

    @abstractmethod
    def projection(
        self, 
        point: np.ndarray,
        vector: np.ndarray
    ) -> np.ndarray:
        ...    

    @abstractmethod
    def retraction(
        self,
        point: np.ndarray,
        tangent_vector: np.ndarray
    ) -> np.ndarray:
        ...
