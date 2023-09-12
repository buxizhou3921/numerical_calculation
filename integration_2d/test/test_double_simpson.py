"""
======================
@author: 王方舟
@time: 2023-08-27:23:45
======================
"""

import numpy as np
import matplotlib.pyplot as plt
from integration_2d.composite_double_simpson import CompositeDoubleSimpson


def fun(x, y):
    return np.exp(-x ** 2 - y ** 2)


if __name__ == "__main__":
    cds = CompositeDoubleSimpson(fun, [0, 1], [0, 1], eps=1e-12)
    cds.cal_2d_int()
    print("划分区间数:%d, 积分近似值:%.15f" % (cds.interval_num, cds.int_value))
    print("积分精度:%.15e" % (0.557746285351034 - cds.int_value))
    cds.plt_precison()

