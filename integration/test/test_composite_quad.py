"""
======================
@author: 王方舟
@time: 2023-08-24:13:06
======================
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from integration.composite_quadrature import CompositeQuadrature

if __name__ == "__main__":
    t = sp.Symbol("t")

    # int_fun = sp.sin(t)
    # cq = CompositeQuadrature(int_fun, [0,  np.pi / 2], interval_num=25, int_type="trapezoid")
    # int_value = cq.cal_int()
    # print("复合梯形公式积分值:", int_value, "余项为:", cq.int_remainder)
    #
    # cq = CompositeQuadrature(int_fun, [0, np.pi / 2], interval_num=25, int_type="simpson")
    # int_value = cq.cal_int()
    # print("复合辛普森公式积分值:", int_value, "余项为:", cq.int_remainder)
    #
    # cq = CompositeQuadrature(int_fun, [0, np.pi / 2], interval_num=25, int_type="cotes")
    # int_value = cq.cal_int()
    # print("复合科特斯公式积分值:", int_value, "余项为:", cq.int_remainder)

    fun1 = sp.exp(t ** 2)  # 分段函数1
    fun2 = 80 / (4 - sp.sin(16 * np.pi * t))  # 分段函数2
    int_fun = sp.Piecewise((fun1, t <= 2), (fun2, t <= 4))
    fun_expr = sp.lambdify(t, int_fun)

    # plt.figure(figsize=(8, 6))
    # xi = np.linspace(0, 4, 500)
    # yi = fun_expr(xi)
    # plt.plot(xi, yi, "k-")
    # plt.fill_between(xi, yi, color="c", alpha=0.5)
    # plt.xlabel("x", fontdict={"fontsize": 12})
    # plt.ylabel("Y", fontdict={"fontsize": 12})
    # plt.title("Integral region of piecewise function", fontdict={"fontsize": 14})
    # plt.show()

    interval_num = np.arange(1000, 21001, 2000)
    for num in interval_num:
        cq = CompositeQuadrature(int_fun, [0, 4], interval_num=num, int_type="cotes")
        int_value = cq.cal_int()
        print("划分区间数%d,积分值%.15f,误差%.10e" % (num, int_value, 57.764450125048512 - int_value))
