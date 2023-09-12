"""
======================
@author: 王方舟
@time: 2023-08-17:15:34
======================
"""
import numpy as np
from matplotlib import pyplot as plt

from interpolation.lagrange import LagrangeInterpolation


def fun(x):
    """
    龙格函数
    """
    return 1 / (x ** 2 + 1)


if __name__ == '__main__':
    plt.figure(figsize=(8, 6))
    xi = np.linspace(-5, 5, 100, endpoint=True)
    for n in range(3, 12, 2):
        x = np.linspace(-5, 5, n, endpoint=True)
        y = fun(x)
        lag_interp = LagrangeInterpolation(x, y)
        lag_interp.fit_interp()  # 生成拉格朗日插值多项式
        yi = lag_interp.cal_interp_x0(xi)  # 拉格朗日插值多项式求解插值点
        plt.plot(xi, yi, lw=0.7, label="n = %d" % (n - 1))

    plt.plot(xi, fun(xi), "k-", label=r"$\frac{1}{1 + x^ {2}}\qquad$")
    plt.xlabel("x", fontdict={"fontsize": 12})
    plt.ylabel("y", fontdict={"fontsize": 12})
    plt.title("Runge phenomenon of lagrange interpolation of different orders", fontdict={"fontsize": 14})
    plt.legend()
    plt.show()
