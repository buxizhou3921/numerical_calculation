"""
======================
@author: 王方舟
@time: 2023-08-23:23:46
======================
"""

import numpy as np
import sympy as sp
from scipy import optimize  # 优化模块


class CompositeQuadrature:
    """
    复合求积公式: 复合梯形公式、复合辛普森公式、复合科特斯公式
    """

    def __init__(self, int_fun, int_interval, interval_num=16, int_type="simpson"):
        self.int_fun = int_fun  # 被积函数(符号定义)
        if len(int_interval) == 2:
            self.a, self.b = int_interval[0], int_interval[1]  # 被积区间
        else:
            raise ValueError("积分区间参数设置错误")
        self.n = int(interval_num)  # 默认16等分子区间
        self.int_type = int_type  # 积分公式类型,默认采用复合辛普森
        self.int_value = None  # 积分值结果
        self.int_remainder = None  # 积分余项值

    def cal_int(self):
        """
        积分类型选择
        :return:
        """
        t = self.int_fun.free_symbols.pop()  # 被积函数自由变量
        fun_expr = sp.lambdify(t, self.int_fun)  # 转换为lambda函数
        if self.int_type == "trapezoid":
            self.int_value = self._cal_trapezoid_(t, fun_expr)
        elif self.int_type == "simpson":
            self.int_value = self._cal_simpson_(t, fun_expr)
        elif self.int_type == "cotes":
            self.int_value = self._cal_cotes_(t, fun_expr)
        else:
            raise ValueError("复合积分类型仅支持trapezoid,simpson,cotes")
        return self.int_value

    def _cal_trapezoid_(self, t, fun_expr):
        """
        复合梯形公式
        :param t: 自由变量
        :param fun_expr: 被积函数
        :return:
        """
        h = (self.b - self.a) / self.n  # 子区间长度
        x_k = np.linspace(self.a, self.b, self.n + 1)  # 等分点
        f_val = fun_expr(x_k)  # 对应等分点的函数值
        int_value = h / 2 * (f_val[0] + f_val[-1] + 2 * sum(f_val[1:-1]))  # 复合梯形公式
        # 余项计算
        diff_fun = self.int_fun.diff(t, 2)  # 被积函数的2阶导函数
        max_val = self._fun_maximize_(t, diff_fun)  # 求函数的最大值
        self.int_remainder = (self.b - self.a) / 12 * h ** 2 * max_val  # 余项公式
        return int_value

    def _cal_simpson_(self, t, fun_expr):
        """
        复合辛普森公式
        :param t: 自由变量
        :param fun_expr: 被积函数
        :return:
        """
        h = (self.b - self.a) / (2 * self.n)  # 子区间长度
        x_k = np.linspace(self.a, self.b, 2 * self.n + 1)  # 等分点
        f_val = fun_expr(x_k)  # 对应等分点的函数值
        idx = np.linspace(0, 2 * self.n, 2 * self.n + 1, dtype=np.int64)  # 节点的索引下标
        f_val_even = f_val[np.mod(idx, 2) == 0]  # 子区间的端点值
        f_val__odd = f_val[np.mod(idx, 2) == 1]  # 子区间中点值
        int_value = h / 3 * (f_val[0] + f_val[-1] + 4 * sum(f_val__odd) + 2 * sum(f_val_even[1:-1]))  # 复合辛普森公式
        # 余项计算
        diff_fun = self.int_fun.diff(t, 4)  # 被积函数的4阶导函数
        max_val = self._fun_maximize_(t, diff_fun)  # 求函数的最大值
        self.int_remainder = (self.b - self.a) / 180 * (h / 2) ** 4 * max_val  # 余项公式
        return int_value

    def _cal_cotes_(self, t, fun_expr):
        """
        复合科特斯公式
        :param t: 自由变量
        :param fun_expr: 被积函数
        :return:
        """
        h = (self.b - self.a) / (4 * self.n)  # 子区间长度
        x_k = np.linspace(self.a, self.b, 4 * self.n + 1)  # 等分点
        f_val = fun_expr(x_k)  # 对应等分点的函数值
        idx = np.linspace(0, 4 * self.n, 4 * self.n + 1, dtype=np.int64)  # 节点的索引下标
        f_val_0 = f_val[np.mod(idx, 4) == 0]  # 4k点
        f_val_1 = f_val[np.mod(idx, 4) == 1]  # 4k+1点
        f_val_2 = f_val[np.mod(idx, 4) == 2]  # 4k+2点
        f_val_3 = f_val[np.mod(idx, 4) == 3]  # 4k+3点
        int_value = 2 * h / 45 * (7 * (f_val[0] + f_val[-1]) + 14 * sum(f_val_0[1:-1]) +
                                  32 * (sum(f_val_1) + sum(f_val_3)) + 12 * sum(f_val_2))  # 复合科特斯公式
        # 余项计算
        diff_fun = self.int_fun.diff(t, 6)  # 被积函数的4阶导函数
        max_val = self._fun_maximize_(t, diff_fun)  # 求函数的最大值
        self.int_remainder = (self.b - self.a) / 945 * 2 * h ** 6 * max_val  # 余项公式
        return int_value

    def _fun_maximize_(self, t, diff_fun):
        """
        求函数最大值
        :param t: 自由变量
        :param diff_fun: 被积函数的n阶导函数
        :return:
        """
        fun_min = sp.lambdify(t, -diff_fun)
        res = optimize.minimize_scalar(fun_min, bounds=(self.a, self.b), method="Bounded")
        return -res.fun
