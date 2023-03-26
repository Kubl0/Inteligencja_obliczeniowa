import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history
import numpy as np
from matplotlib import pyplot as plt

options = {"c1": 0.5, "c2": 0.3, "w": 0.9}

x_max = np.ones(6)
x_min = np.zeros(6)
my_bounds = (x_min, x_max)

optimizer = ps.single.GlobalBestPSO(
    n_particles=10, dimensions=6, options=options, bounds=my_bounds
)


def endurance(x):
    return (
        np.exp(-2 * (x[:, 1] - np.sin(x[:, 0])) ** 2)
        + np.sin(x[:, 2] * x[:, 3])
        + np.cos(x[:, 4] * x[:, 5])
    )


def f(x):
    return -1 * endurance(x)


optimizer.optimize(f, iters=1000)

plot_cost_history(optimizer.cost_history)
plt.show()
