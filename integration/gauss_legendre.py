"""
======================
@author: 王方舟
@time: 2023-08-28:10:38
======================
"""

import numpy as np
import sympy as sp
import math


class GaussLegendre:
    """
    高斯-勒让德
    """

    def __init__(self, int_fun, int_interval, zeros_num=10):
        self.int_fun = int_fun  # 被积函数
        if len(int_interval) == 2:
            self.a, self.b = int_interval[0], int_interval[1]  # 被积区间
        else:
            raise ValueError("积分区间参数设置错误")
        self.n = int(zeros_num)  # 正交多项式零点数
        self.zeros_points = None  # 勒让德高斯零点
        self.int_value = None  # 积分值结果
        self.A_k = None  # 求积系数

    def cal_int(self):
        self._cal_Ak_coef_()
        f_val = self.int_fun(self.zeros_points)  # 零点函数值
        self.int_value = np.dot(self.A_k, f_val)  # 插值型求积
        return self.int_value

    def _cal_gauss_zeros_points_(self):
        """
        计算高斯零点
        """
        t = sp.Symbol("t")
        # 勒让德多项式
        p_n = (t ** 2 - 1) ** self.n / math.factorial(self.n) / 2 ** self.n
        diff_p_n = sp.diff(p_n, t, self.n)  # 多项式的n阶导数
        # 求解多项式全部零点
        self.zeros_points = np.asarray(sp.solve(diff_p_n, t), dtype=np.float64)
        return diff_p_n, t

    def _cal_Ak_coef_(self):
        """
        计算Ak系数
        :return:
        """
        diff_p_n, t = self._cal_gauss_zeros_points_()
        Ak_poly = sp.lambdify(t, 2 / (1 - t ** 2) / (diff_p_n.diff(t, 1) ** 2))
        self.A_k = Ak_poly(self.zeros_points)  # 求解Ak系数
        # 区问转换，[a, b] --> [-1, 1]
        self.A_k = self.A_k * (self.b - self.a) / 2
        self.zeros_points = (self.b - self.a) / 2 * self.zeros_points + (self.a + self.b) / 2
