"""
======================
@author: 王方舟
@time: 2023-08-29:14:39
======================
"""

import numpy as np
import matplotlib.pyplot as plt
from integration_3d.gauss_legendre_triple import GaussLegendreTriple


def fun(x, y, z):
    return 4 * x * z * np.exp(-x ** 2 * y - z ** 2)


if __name__ == '__main__':
    glt = GaussLegendreTriple(fun, [0, 1], [0, np.pi], [0, np.pi], zeros_num=[10, 12, 15])
    res = glt.cal_3d_int()
    print("积分值:%.15f, 精度:%.15e" % (res, 1.7327622230312205 - res))

    # zeros_num = np.arange(10, 20, 1, np.int64)
    # int_values = []
    # for num in zeros_num:
    #     glt = GaussLegendreTriple(fun, [0, 1], [0, np.pi], [0, np.pi], zeros_num=num)
    #     res = glt.cal_3d_int()
    #     # print("零点数:%d, 积分值:%.15f, 精度:%.15e" % (num, res, 1.7327622230312205 - res))
    #     int_values.append(res)
    #
    # plt.figure(figsize=(8, 6))
    # plt.plot(zeros_num, int_values, "ro-")
    # plt.xlabel("Zero Points Number", fontdict={"fontsize": 12})
    # plt.ylabel("Integration Values", fontdict={"fontsize": 12})
    # plt.title("Gauss Legendre 3d-int Convergence Curve", fontdict={"fontsize": 14})
    # plt.show()

