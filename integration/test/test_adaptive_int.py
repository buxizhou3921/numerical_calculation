"""
======================
@author: 王方舟
@time: 2023-08-27:17:30
======================
"""

import numpy as np
import matplotlib.pyplot as plt
from integration.adaptive_integral_algorithm import AdaptiveIntegralAlgorithm


def fun(x):
    return 1 / (np.sin(6 * np.pi * x) + np.sqrt(x))


if __name__ == "__main__":
    adaint = AdaptiveIntegralAlgorithm(fun, [1, 2], eps=1e-10)
    adaint.cal_int()
    print(adaint.int_value, len(adaint.x_node))

    plt.figure(figsize=(8, 12))
    plt.subplot(211)
    xi = np.linspace(1, 2, 100)
    yi = fun(xi)
    y_value = fun(adaint.x_node)
    plt.plot(adaint.x_node, y_value, "k.")
    plt.fill_between(xi, yi, color="c", alpha=0.4)
    plt.subplot(212)
    bins = np.linspace(1, 2, 11)
    n = plt.hist(adaint.x_node, bins=bins, color="r", alpha=0.5)
    plt.plot((n[1][:-1] + n[1][1:]) / 2, n[0], "ko-", lw=2)
    plt.show()
