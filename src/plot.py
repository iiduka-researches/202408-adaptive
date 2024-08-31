import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# {'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'}

def compound_plot(
    x: list[np.ndarray],
    color: str=None,
    marker: str='o',
    markevery: int=10,
    label: str=None
) -> None:
    N: int = len(x[0])
    _max: np.ndarray = np.maximum.reduce(x)
    _min: np.ndarray = np.minimum.reduce(x)
    _avg: np.ndarray = np.add.reduce(x) / len(x)
    plt.plot(range(N), _max, color=color, linewidth=0.2)
    plt.plot(range(N), _avg, color=color, linewidth=1., label=label, marker=marker, markevery=markevery, markersize=5)
    plt.plot(range(N), _min, color=color, linewidth=0.2)
    plt.fill_between(range(N), _max, _min, fc=color, alpha=0.1)


optimizers = [
    ('rsgd', 'b', 'x'),
    ('radam', 'g', '.'),
    ('ramsgrad', 'r', '.'),
    ('rasal', 'm', '|'),
    ('rasar', 'orange', '|'),
    ('rasalr', 'c', '|')
]

stepsize = 'diminishing'
dataset = 'coil100'
y_axis = 'loss'

for optimizer, color, marker in optimizers:
    l = []
    for i in range(3):
        d = pd.read_pickle(f'pkl/{dataset}/{stepsize}/{optimizer}-{i}.pkl')
        x = d[y_axis]
        l.append(x)
    compound_plot(l, color=color, marker=marker, label=d['optimizer'] + f'(alpha={d['lr']})')

plt.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.99)

plt.xlabel('Number of iterations')
if y_axis == 'loss':
    plt.ylabel('Objective function value')
elif y_axis == 'grad_norm':
    plt.ylabel('Norm of the gradient of the objective function')
plt.legend()
plt.grid()
plt.show()
