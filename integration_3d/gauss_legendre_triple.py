"""
======================
@author: 王方舟
@time: 2023-08-29:14:33
======================
"""

import math
import numpy as np
import sympy as sp


class GaussLegendreTriple:
    """
    高斯—勒让德三重积分
    """

    def __init__(self, int_fun, x_span, y_span, z_span, zeros_num=None):
        self.int_fun = int_fun  # 被积函数
        self.ax, self.bx = x_span[0], x_span[1]  # x的积分上下限
        self.ay, self.by = y_span[0], y_span[1]  # y的积分上下限
        self.az, self.bz = z_span[0], z_span[1]  # z的积分上下限
        if zeros_num is None:
            self.n_x, self.n_y, self.n_z = 10, 10, 10
        else:
            if len(zeros_num) != 3:
                raise ValueError("零点数设置格式为[nx, ny, nz].")
            else:
                self.n_x, self.n_y, self.n_z = zeros_num[0], zeros_num[1], zeros_num[2]
        self.int_value = None  # 最终积分值

    def cal_3d_int(self):
        # 计算勒让德的零点与Ak系数
        A_k_x, zero_points_x = self._cal_Ak_zeros_(self.n_x)
        A_k_y, zero_points_y = self._cal_Ak_zeros_(self.n_y)
        A_k_z, zero_points_z = self._cal_Ak_zeros_(self.n_z)
        # 积分区间变换[a, b] --> [-1, 1]
        A_k_x = A_k_x * (self.bx - self.ax) / 2
        A_k_y = A_k_y * (self.by - self.ay) / 2
        A_k_z = A_k_z * (self.bz - self.az) / 2
        zero_points_x = (self.bx - self.ax) / 2 * zero_points_x + (self.bx + self.ax) / 2
        zero_points_y = (self.by - self.ay) / 2 * zero_points_y + (self.by + self.ay) / 2
        zero_points_z = (self.bz - self.az) / 2 * zero_points_z + (self.bz + self.az) / 2

        xyz = np.meshgrid(zero_points_x, zero_points_y, zero_points_z)
        f_val = self.int_fun(xyz[0], xyz[1], xyz[2])
        self.int_value = 0.0
        for j in range(self.n_y):
            for i in range(self.n_x):
                for k in range(self.n_z):
                    self.int_value += A_k_x[i] * A_k_y[j] * A_k_z[k] * f_val[j, i, k]
        return self.int_value

    @staticmethod
    def _cal_Ak_zeros_(n):
        """
        计算勒让德的零点与Ak系数
        :return:
        """
        t = sp.Symbol("t")
        # 勒让德多项式
        p_n = (t ** 2 - 1) ** n / math.factorial(n) / 2 ** n
        diff_p_n = sp.diff(p_n, t, n)  # 多项式的n阶导数
        # 求解多项式全部零点
        zeros_points = np.asarray(sp.solve(diff_p_n, t), dtype=np.float64)
        # 求解Ak系数
        Ak_poly = sp.lambdify(t, 2 / (1 - t ** 2) / (diff_p_n.diff(t, 1) ** 2))
        A_k = Ak_poly(zeros_points)
        return A_k, zeros_points



