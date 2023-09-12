"""
======================
@author: 王方舟
@time: 2023-08-31:18:50
======================
"""

import numpy as np
import matplotlib.pyplot as plt
from differential_equation.runge_kutta_4 import RungeKutta4


def func(x, y): return np.exp(x) + y


if __name__ == '__main__':
    a = RungeKutta4(func, [0, 2], 1, 10)
    t, y = a.main()

    plt.plot(t, y, lw=2, label="'RK-4's result")
    plt.plot(t, np.exp(t) * (t + 1), '-.', label="Real")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
