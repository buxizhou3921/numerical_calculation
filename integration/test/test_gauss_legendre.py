"""
======================
@author: 王方舟
@time: 2023-08-28:10:55
======================
"""

import numpy as np
import matplotlib.pyplot as plt
from integration.gauss_legendre import GaussLegendre


def fun(x):
    return np.sin(x) * np.exp(-x)


if __name__ == "__main__":
    gauss_zeros_num = np.arange(10, 21, 1)
    int_res = 0.5 * (1 - np.exp(-8) * (np.sin(8) + np.cos(8)))
    precision = []
    for num in gauss_zeros_num:
        gl = GaussLegendre(fun, [0, 8], zeros_num=num)
        int_value = gl.cal_int()
        precision.append(int_res - int_value)
        print("num:%d，积分值:%.15f，误差:%.15e" % (num, int_value, precision[-1]))

    plt.figure(figsize=(8, 6))
    plt.plot(gauss_zeros_num, precision, "ko-")
    plt.show()

