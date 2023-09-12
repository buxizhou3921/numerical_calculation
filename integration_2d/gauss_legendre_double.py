"""
======================
@author: 王方舟
@time: 2023-08-29:03:08
======================
"""

import math
import numpy as np
import sympy as sp


class GaussLegendreDouble:
    """
    高斯—勒让德二重积分
    """

    def __init__(self, int_fun, x_span, y_span, zeros_num=10):
        self.int_fun = int_fun  # 被积函数
        self.ax, self.bx = x_span[0], x_span[1]  # x的积分上下限
        self.ay, self.by = y_span[0], y_span[1]  # y的积分上下限
        self.n = zeros_num  # 零点数
        self.int_value = None  # 最终积分值

    def cal_2d_int(self):
        # 计算勒让德的零点与Ak系数
        A_k, zero_points = self._cal_Ak_zeros_()
        # 积分区间变换[a, b] --> [-1, 1]
        A_k_x = A_k * (self.bx - self.ax) / 2
        A_k_y = A_k * (self.by - self.ay) / 2
        zero_points_x = (self.bx - self.ax) / 2 * zero_points + (self.bx + self.ax) / 2
        zero_points_y = (self.by - self.ay) / 2 * zero_points + (self.by + self.ay) / 2

        xy = np.meshgrid(zero_points_x, zero_points_y)
        f_val = self.int_fun(xy[0], xy[1])
        self.int_value = 0.0
        for i in range(self.n):
            for j in range(self.n):
                self.int_value += A_k_x[i] * A_k_y[j] * f_val[i, j]
        return self.int_value

    def _cal_Ak_zeros_(self):
        """
        计算勒让德的零点与Ak系数
        :return:
        """
        t = sp.Symbol("t")
        # 勒让德多项式
        p_n = (t ** 2 - 1) ** self.n / math.factorial(self.n) / 2 ** self.n
        diff_p_n = sp.diff(p_n, t, self.n)  # 多项式的n阶导数
        # 求解多项式全部零点
        zeros_points = np.asarray(sp.solve(diff_p_n, t), dtype=np.float64)
        # 求解Ak系数
        Ak_poly = sp.lambdify(t, 2 / (1 - t ** 2) / (diff_p_n.diff(t, 1) ** 2))
        A_k = Ak_poly(zeros_points)
        return A_k, zeros_points


