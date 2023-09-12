"""
======================
@author: 王方舟
@time: 2023-08-17:15:11
======================
"""
import numpy as np
from matplotlib import pyplot as plt


def cal_interp_x0(polynomial, x0):
    """
    计算给定插值点数值
    :param polynomial:
    :param x0:
    :return:
    """
    x0 = np.asarray(x0, dtype=np.float64)
    n0 = len(x0)  # 所求插值点的个数
    y_0 = np.zeros(n0)  # 存储插值点x0所对应的插值
    t = polynomial.free_symbols.pop()  # 获取插值多项式的自由变量
    for i in range(n0):
        y_0[i] = polynomial.evalf(subs={t: x0[i]})
    return y_0


def plt_interpolation(params):
    """
    可视化插值图像和所求的插值点
    :param params:
    :return:
    """
    polynomial, x, y, title, x0, y0 = params
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, "ro", label="Interpolation base points")
    xi = np.linspace(min(x), max(x), 100)  # 模拟100个值
    yi = cal_interp_x0(polynomial, xi)
    plt.plot(xi, yi, "b--", label="Interpolation polynomial")
    if x0 is not None and y0 is not None:
        plt.plot(x0, y0, "g+", label="Interpolation point values")
    plt.legend()
    plt.xlabel("x", fontdict={"fontsize": 12})
    plt.ylabel("y", fontdict={"fontsize": 12})
    plt.title(title + " interpolation polynomial and values", fontdict={"fontsize": 14})
    plt.grid(ls=":")
    plt.show()
