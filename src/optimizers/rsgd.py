import numpy as np
from time import time
from tqdm import tqdm

from utils import Problem
from .optimizer import Optimizer
from .learning_rate import LearningRate


class RSGD(Optimizer):
    '''
    Implements Riemannian Stochastic Grafient Descent.

    Attribute
    ---------
    batch_size (int):
        batch size.
    lr (LearningRate):
        learning rate.
    '''
    def __init__(
            self,
            batch_size: int,
            lr: LearningRate
        ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        super(RSGD, self).__init__(batch_size=batch_size, lr=lr)
    
    def __repr__(self) -> str:
        return 'RSGD'
    
    def solve(self, problem: Problem, point: np.ndarray, max_iter: int=100):
        self.init_history()
        self.update_history(
            loss=problem.f(point),
            grad_norm=np.linalg.norm(problem.full_grad(point)),
            elapsed_time=None
        )

        for k in tqdm(range(max_iter)):
            start_time = time()
            grad = problem.minibatch_grad(point, self.batch_size)
            d_p = -self.lr(k + 1) / np.sqrt(k + 1) * grad
            point = problem.manifold.retraction(point, d_p)
            end_time = time()

            self.update_history(
                loss=problem.f(point),
                grad_norm=np.linalg.norm(problem.full_grad(point)),
                elapsed_time=end_time - start_time
            )
        return self.history
