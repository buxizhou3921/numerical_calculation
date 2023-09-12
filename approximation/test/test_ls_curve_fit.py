"""
======================
@author: 王方舟
@time: 2023-08-30:19:37
======================
"""

import numpy as np
import matplotlib.pyplot as plt
from approximation.least_square_poly_fit import LeastSquarePolyFit

if __name__ == "__main__":
    x = np.linspace(0, 5, 15)
    np.random.seed(0)
    y = 2 * np.sin(x) * np.exp(-x) + np.random.randn(15) / 50
    xi = np.linspace(0, 5, 100)
    plt.figure(figsize=(8, 6))

    orders = [5, 8, 12, 15]
    line_style = ["--", ":", "-", "-."]
    for k, line in zip(orders, line_style):
        ls = LeastSquarePolyFit(x, y, k=k)
        ls.fit_ls_curve()
        yi = ls.cal_x0(xi)  # 拟合值
        plt.plot(xi, yi, line, lw=1.5, label="order = %d,mse = %.2e" % (k, ls.mse))
    plt.plot(x, y, "ro", label="Original data")
    plt.legend()
    plt.xlabel("x", fontdict={"fontsize": 12})
    plt.ylabel("Y", fontdict={"fontsize": 12})
    plt.title("The least square fitted curve and different orders", fontdict={"fontsize": 14})
    plt.show()
