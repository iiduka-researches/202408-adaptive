import numpy as np
from time import time
from tqdm import tqdm

from utils import Problem
from .optimizer import Optimizer
from .learning_rate import LearningRate


class RASA(Optimizer):
    '''
    Implements Riemannian Adaptive Stochastic Algorithm.

    Attribute
    ---------
    batch_size (int):
        batch size.
    lr (LearningRate):
        learning rate.
    beta (float=0.99):
        coefficients used for computing running averages of gradient.
    '''
    def __init__(
            self,
            batch_size: int,
            lr: LearningRate,
            beta: float=0.99,
            variant: str='LR'
        ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= beta < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {beta}')
        super(RASA, self).__init__(batch_size=batch_size, lr=lr)
        self.beta = beta
        self.variant = variant

    def __repr__(self) -> str:
        return 'RASA-' + self.variant
    
    def solve(self, problem: Problem, point: np.ndarray, max_iter: int=100):
        self.init_history()
        self.update_history(
            loss=problem.f(point),
            grad_norm=np.linalg.norm(problem.full_grad(point)),
            elapsed_time=None
        )

        exp_avg_left = np.zeros(point.shape[0])
        max_exp_avg_left = np.zeros(point.shape[0])
        exp_avg_right = np.zeros(point.shape[1])
        max_exp_avg_right = np.zeros(point.shape[1])

        for k in tqdm(range(max_iter)):
            start_time = time()

            grad = problem.minibatch_grad(point, self.batch_size)
            left_matrix = np.eye(point.shape[0])
            right_matrix = np.eye(point.shape[1])

            if 'L' in self.variant:
                exp_avg_left = self.beta * exp_avg_left + (1 - self.beta) * np.diag(grad @ grad.T) / point.shape[1]
                max_exp_avg_left = np.maximum(max_exp_avg_left, exp_avg_left)
                left_matrix = np.diag(max_exp_avg_left ** -0.25)

            if 'R' in self.variant:
                exp_avg_right = self.beta * exp_avg_right + (1 - self.beta) * np.diag(grad.T @ grad) / point.shape[0]
                max_exp_avg_right = np.maximum(max_exp_avg_right, exp_avg_right)
                right_matrix = np.diag(max_exp_avg_right ** -0.25)

            d_p = -self.lr(k + 1) / np.sqrt(k + 1) * problem.manifold.projection(point, left_matrix @ grad @ right_matrix)
            point = problem.manifold.retraction(point, d_p)
            end_time = time()

            self.update_history(
                loss=problem.f(point),
                grad_norm=np.linalg.norm(problem.full_grad(point)),
                elapsed_time=end_time - start_time
            )
        return self.history
