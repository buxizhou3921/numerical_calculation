"""
======================
@author: 王方舟
@time: 2023-08-26:14:12
======================
"""

import numpy as np
import sympy as sp


class RombergAccelerationQuadrature:
    """
    龙贝格加速法求积
    """

    def __init__(self, int_fun, int_interval, acceleration_num=10):
        self.int_fun = int_fun  # 被积函数(符号定义)
        if len(int_interval) == 2:
            self.a, self.b = int_interval[0], int_interval[1]  # 被积区间
        else:
            raise ValueError("积分区间参数设置错误")
        self.n = int(acceleration_num)  # 外推次数
        self.int_value = None  # 积分值结果
        self.romberg_table = None  # 龙贝格加速表

    def cal_int(self):
        """
        龙贝格求积公式
        """
        self.romberg_table = np.zeros((self.n + 1, self.n + 1))
        # 第一列存储逐次分半梯形公式积分值


