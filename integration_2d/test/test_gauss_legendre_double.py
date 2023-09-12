"""
======================
@author: 王方舟
@time: 2023-08-29:14:27
======================
"""

import numpy as np
from integration_2d.gauss_legendre_double import GaussLegendreDouble


def fun(x, y):
    return np.exp(-x ** 2 - y ** 2)


if __name__ == "__main__":
    gld = GaussLegendreDouble(fun, [0, 1], [0, 1], zeros_num=10)
    res = gld.cal_2d_int()
    print("二重积分结果:%.15f，精度:%.15e" % (res, 0.557746285351034 - res))

