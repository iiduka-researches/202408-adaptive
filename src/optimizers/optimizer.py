from abc import ABC, abstractmethod
from .learning_rate import LearningRate


class Optimizer(ABC):
    def __init__(self, batch_size: int, lr: LearningRate) -> None:
        self.lr = lr
        self.batch_size = batch_size
        self.history = dict()
    
    def init_history(self) -> None:
        self.history = dict(
            optimizer=repr(self),
            batch_size=self.batch_size,
            lr=self.lr,
            loss=[],
            grad_norm=[],
            elapsed_time=[]
        )
    
    def update_history(self, loss: float=None, grad_norm: float=None, elapsed_time: float=None) -> None:
        if loss:
            self.history['loss'].append(loss)
        if grad_norm:
            self.history['grad_norm'].append(grad_norm)
        if elapsed_time:
            self.history['elapsed_time'].append(elapsed_time)

    @abstractmethod
    def solve():
        ...
