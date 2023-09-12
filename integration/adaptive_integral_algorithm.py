"""
======================
@author: 王方舟
@time: 2023-08-27:15:45
======================
"""
import numpy as np


class AdaptiveIntegralAlgorithm:
    """
    自适应积分算法
    """

    def __init__(self, int_fun, int_interval, eps=1e-8):
        self.int_fun = int_fun  # 被积函数(符号定义)
        if len(int_interval) == 2:
            self.a, self.b = int_interval[0], int_interval[1]  # 被积区间
        else:
            raise ValueError("积分区间参数设置错误")
        self.eps = eps  # 精度
        self.int_value = None  # 积分值结果
        self.x_node = [self.a, self.b]  # 最终划分的结点分布情况

    def cal_int(self):
        """
        自适应积分算法,采用递归求解
        """
        self.int_value = self._sub_cal_int_(self.a, self.b)
        self.x_node = np.asarray(sorted(self.x_node))
        return self.int_value

    def _sub_cal_int_(self, a, b):
        """
        递归计算每个子区间的积分值，并根据精度要求是否再次划分区间
        :param a:
        :param b:
        :return:
        """
        complete_int_value = self._simpson_int_(a, b)  # 子区间采用辛普森公式
        mid = (a + b) / 2
        left_half = self._simpson_int_(a, mid)
        right_half = self._simpson_int_(mid, b)  # 精度判别
        if abs(complete_int_value - (left_half + right_half)) <= 5 * self.eps:
            int_value = left_half + right_half
        else:
            self.x_node.append(mid)
            int_value = self._sub_cal_int_(a, mid) + self._sub_cal_int_(mid, b)
        return int_value

    def _simpson_int_(self, a, b):
        """
        子区间采用辛普森公式
        :param a: 子区间左端点
        :param b: 子区间右端点
        :return:
        """
        mid = (a + b) / 2  # 中点
        return (b - a) / 6 * (self.int_fun(a) + self.int_fun(b) + 4 * self.int_fun(mid))

