import numpy as np
from abc import ABC, abstractmethod


class LearningRate(ABC):
    def __init__(self, base_lr: float) -> None:
        if not 0.0 <= base_lr:
            raise ValueError(f'Invalid base learning rate: {base_lr}')
        self.base_lr = base_lr

    def __call__(self, k: int):
        return self.get_lr(k=k)

    @abstractmethod
    def get_lr(self, k: int) -> float:
        ...


class ConstantLR(LearningRate):
    def __init__(self, base_lr: float) -> None:
        super(ConstantLR, self).__init__(base_lr)
    
    def __repr__(self) -> str:
        return str(self.base_lr)
    
    def get_lr(self, k: int) -> float:
        return self.base_lr


class DiminishingLR(LearningRate):
    def __init__(self, base_lr: float) -> None:
        super(DiminishingLR, self).__init__(base_lr)
    
    def __repr__(self) -> str:
        return str(self.base_lr) + ' / sqrt(k + 1)'

    def get_lr(self, k: int) -> float:
        return self.base_lr / np.sqrt(k + 1)
