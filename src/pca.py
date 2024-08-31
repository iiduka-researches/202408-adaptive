import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms

from optimizers import RSGD, RAdam, RASA, ConstantLR
from manifolds import Manifold, Stiefel
from utils import Problem


class PrincipalComponentAnalysis(Problem):
    def __init__(
        self,
        manifold: Manifold,
        data: np.ndarray
    ) -> None:
        super(PrincipalComponentAnalysis, self).__init__(manifold)
        self.data = data
        
    def f(self, point: np.ndarray) -> float:
        X = self.data
        N: int = X.shape[0]
        return np.linalg.norm(X.T - point @ point.T @ X.T) ** 2 / N

    def egrad(self, point: np.ndarray, idx: int) -> np.ndarray:
        X = self.data
        x: np.ndarray = X[idx]
        _x = np.expand_dims(x, axis=1)
        return -2 * (_x @ _x.T @ point)

    def minibatch_grad(
        self,
        point: np.ndarray,
        batch_size: int
    ) -> np.ndarray:
        N: int = self.data.shape[0]
        samples = [np.random.randint(0, N) for _ in range(batch_size)]
        X = self.data[samples]
        return -2 * self.manifold.projection(point, (X.T @ X @ point) / batch_size)

    def full_grad(
        self,
        point: np.ndarray,
    ) -> np.ndarray:
        X = self.data
        N: int = X.shape[0]
        return -2 * self.manifold.projection(point, (X.T @ X @ point) / N)


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed=seed)

    dataset = 'MNIST'
    if dataset == 'MNIST':
        data = MNIST(root="data", download=True, train=False, transform=transforms.ToTensor())
        data = data.data.view(-1, 28 * 28) / 255
        data = data.numpy().copy()
        N: int = data.shape[0]
        n: int = data.shape[1]
        p: int = 10
    elif dataset == 'COIL100':
        data = pd.read_csv('data/coil100/coil100.csv', header=None).values / 255
        N: int = data.shape[0]
        n: int = data.shape[1]
        p: int = 100
    print(f'{(N, n, p)=}')

    max_iter: int = 10
    batch_size: int = 2 ** 10

    base_lr: float = 1e-3
    lr = ConstantLR(base_lr=base_lr)

    pkl_dir: str = os.path.join('results', os.path.join('constant', dataset))
    if not os.path.isdir(pkl_dir):
        os.makedirs(pkl_dir)

    manifold = Stiefel(p, n)
    initial = np.linalg.qr(np.random.rand(n, p))[0]
    problem = PrincipalComponentAnalysis(manifold=manifold, data=data)

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
