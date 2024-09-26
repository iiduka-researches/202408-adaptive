import numpy as np
from time import time
from tqdm import tqdm

from utils import Problem
from .optimizer import Optimizer
from .learning_rate import LearningRate


class RAdam(Optimizer):
    '''
    Implements Riemannian Adam.

    Attribute
    ---------
    batch_size (int):
        batch size.
    lr (LearningRate):
        learning rate.
    betas (tuple=(0.9,0.999)):
        coefficients used for computing running averages of gradient and its square.
    eps: (float=1e-8):
        term added to the denominator to improve numerical stability.
    amsgrad (bool=False):
        whether to use the RAMSGrad variant of this algorithm.
    '''
    def __init__(
            self,
            batch_size: int,
            lr: LearningRate,
            betas: tuple=(0.9, 0.999),
            eps: float=1e-8,
            amsgrad: bool=False
        ) -> None:
        if not 0.0 <= eps:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        super(RAdam, self).__init__(batch_size=batch_size, lr=lr)
        self.betas = betas
        self.eps = eps,
        self.amsgrad = amsgrad

    def __repr__(self) -> str:
        if self.amsgrad:
            return 'RAMSGrad'
        return 'RAdam'
    
    def solve(self, problem: Problem, point: np.ndarray, max_iter: int=100):
        self.init_history()
        self.update_history(
            loss=problem.f(point),
            grad_norm=np.linalg.norm(problem.full_grad(point)),
            elapsed_time=None
        )

        amsgrad = self.amsgrad
        beta1, beta2 = self.betas
        exp_avg = np.zeros_like(point)
        exp_avg_sq = np.zeros_like(point)
        max_exp_avg_sq = np.zeros_like(point)
        for k in tqdm(range(max_iter)):
            start_time = time()

            bias_correction1: float = 1.
            bias_correction2: float = 1.
            if not amsgrad:
                bias_correction1 = 1. - beta1 ** (k + 1)
                bias_correction2 = 1. - beta2 ** (k + 1)

            grad = problem.minibatch_grad(point, self.batch_size)
            exp_avg = beta1 * exp_avg + (1 - beta1) * grad
            exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
            max_exp_avg_sq = np.maximum(max_exp_avg_sq, exp_avg_sq / bias_correction2)
            denom = np.sqrt(max_exp_avg_sq) + self.eps
            d_p = -self.lr(k + 1) * problem.manifold.projection(point, exp_avg / denom / bias_correction1)
            point = problem.manifold.retraction(point, d_p)
            end_time = time()

            self.update_history(
                loss=problem.f(point),
                grad_norm=np.linalg.norm(problem.full_grad(point)),
                elapsed_time=end_time - start_time
            )
        return self.history
