import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt

from optimizers import RSGD, RAdam, RASA, ConstantLR
from manifolds import Manifold, Grassmann
from utils import Problem


class LowRankMatrixCompletion(Problem):
    def __init__(
        self,
        manifold: Manifold,
        data: np.ndarray
    ) -> None:
        super(LowRankMatrixCompletion, self).__init__(manifold)
        self.data = data
        
    def f(self, point: np.ndarray) -> float:
        X = self.data
        N: int = X.shape[0]
    
        _sum: float = 0.
        for idx in range(N):
            x = np.zeros(point.shape[0])
            nonzero_indices: list[int] = X.indices[X.indptr[idx]: X.indptr[idx + 1]]
            nonzero_values = X.data[X.indptr[idx]: X.indptr[idx + 1]]
            x[nonzero_indices] = nonzero_values

            U_omega = point[nonzero_indices]
            a = np.linalg.lstsq(U_omega, nonzero_values, rcond=None)[0]

            indicator = np.zeros(point.shape[0])
            indicator[nonzero_indices] = 1.

            _sum += np.linalg.norm(indicator * (point @ a) - x) ** 2
        return _sum / N

    def egrad(self, point: np.ndarray, idx: int) -> np.ndarray:
        X = self.data
        x = np.zeros(point.shape[0])
        nonzero_indices: list[int] = X.indices[X.indptr[idx]: X.indptr[idx + 1]]
        nonzero_values = X.data[X.indptr[idx]: X.indptr[idx + 1]]
        x[nonzero_indices] = nonzero_values

        U_omega = point[nonzero_indices]
        a = np.linalg.lstsq(U_omega, nonzero_values, rcond=None)[0]

        indicator = np.zeros(point.shape[0])
        indicator[nonzero_indices] = 1.

        return np.expand_dims((indicator * (point @ a) - x), axis=1) @ np.expand_dims(a, axis=1).T

    def minibatch_grad(
        self,
        point: np.ndarray,
        batch_size: int
    ) -> np.ndarray:
        X = self.data
        N: int = X.shape[0]
        egrad = np.zeros_like(point)
        for _ in range(batch_size):
            idx = np.random.randint(0, N)
            egrad += self.egrad(point, idx)
        return self.manifold.projection(point, egrad) / batch_size / 2

    def full_grad(
        self,
        point: np.ndarray,
    ) -> np.ndarray:
        X = self.data
        N: int = X.shape[0]
        egrad = np.zeros_like(point)
        for idx in range(N):
            egrad += self.egrad(point, idx)
        return self.manifold.projection(point, egrad) / N / 2
    

if __name__ == '__main__':
    seed = 0
    np.random.seed(seed=seed)

    dataset = 'jester'
    if dataset == 'ml1m':
        df = pd.read_csv('data/ml-1m/ratings.csv')
        data = csr_matrix((df.Rating, (df.UserID - 1, df.MovieID - 1))) / 5
    elif dataset == 'jester':
        df = pd.read_csv('data/jester/jester.csv')
        data = csr_matrix((df.data, (df.col, df.row)))

    N: int = data.shape[0]
    n: int = data.shape[1]
    p: int = 10
    print(f'{(N, n, p)=}')

    max_iter: int = 10
    batch_size: int = 2 ** 8

    base_lr: float = 1e-3
    lr = ConstantLR(base_lr=base_lr)

    pkl_dir: str = os.path.join('results', os.path.join('constant', dataset))
    if not os.path.isdir(pkl_dir):
        os.makedirs(pkl_dir)

    initial = np.linalg.qr(np.random.rand(n, p))[0]
    manifold = Grassmann(p, n)
    problem = LowRankMatrixCompletion(manifold=manifold, data=data)

    optimizers = dict(
        rsgd=RSGD(batch_size=batch_size, lr=lr),
        radam=RAdam(batch_size=batch_size, lr=lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False),
        ramsgrad=RAdam(batch_size=batch_size, lr=lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=True),
        rasal=RASA(batch_size=batch_size, lr=lr, beta=0.99, variant='L'),
        rasar=RASA(batch_size=batch_size, lr=lr, beta=0.99, variant='R'),
        rasalr=RASA(batch_size=batch_size, lr=lr, beta=0.99, variant='LR'),
    )

    for key in optimizers:
        print(optimizers[key])
        history = optimizers[key].solve(problem, initial, max_iter=max_iter)
        pkl_name: str =  f'{key}.pkl'
        pd.to_pickle(history, os.path.join(pkl_dir, pkl_name))

        plt.plot(range(len(history['loss'])), history['loss'], label=history['optimizer'])
        
    plt.legend()
    plt.show()
