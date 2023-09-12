"""
======================
@author: 王方舟
@time: 2023-08-27:19:46
======================
"""

import numpy as np
import matplotlib.pyplot as plt


class CompositeDoubleSimpson:
    """
    复合辛普森二重积分(自适应划分)
    """

    def __init__(self, int_fun, x_span, y_span, eps=1e-6, max_split=100, increment=10):
        self.int_fun = int_fun  # 被积函数
        self.x_span = np.asarray(x_span, np.float64)  # x积分区间
        self.y_span = np.asarray(y_span, np.float64)  # y积分区间
        self.eps = eps  # 自适应积分精度
        self.max_split = max_split  # 最大划分区间数
        self.increment = increment  # 每次递增区间数
        self._integral_values_ = []  # 存储自适应过程中的每次积分值
        self._n_splits_ = []  # 存储自适应过程中每次划分的区间数
        self.int_value = None  # 最终积分值
        self.interval_num = 0  # 最终划分的区间数

    def cal_2d_int(self):
        for i in range(self.max_split):
            n = self.increment * (i + 1)  # 每次递增10个区间
            hx, hy = np.diff(self.x_span) / n, np.diff(self.y_span) / n  # 区间长度
            xi = np.linspace(self.x_span[0], self.x_span[1], n + 1, endpoint=True)  # x和y等分节点
            yi = np.linspace(self.y_span[0], self.y_span[1], n + 1, endpoint=True)  # x和y等分节点
            xy = np.meshgrid(xi, yi)

            int1 = np.sum(self.int_fun(xy[0][:-1, :-1], xy[1][:-1, :-1]))
            int2 = np.sum(self.int_fun(xy[0][1:, :-1], xy[1][1:, :-1]))
            int3 = np.sum(self.int_fun(xy[0][:-1, 1:], xy[1][:-1, 1:]))
            int4 = np.sum(self.int_fun(xy[0][1:, 1:], xy[1][1:, 1:]))
            xci = np.divide(xy[0][:-1, :-1] + xy[0][1:, 1:], 2)  # x的各中点值
            yci = np.divide(xy[1][:-1, :-1] + xy[1][1:, 1:], 2)  # y的各中点值
            int5 = np.sum(self.int_fun(xci, xy[1][:-1, :-1])) + np.sum(self.int_fun(xy[0][:-1, :-1], yci)) + \
                   np.sum(self.int_fun(xci, xy[1][1:, 1:])) + np.sum(self.int_fun(xy[0][1:, 1:], yci))
            int6 = np.sum(self.int_fun(xci, yci))
            int_value = hx * hy / 36 * (int1 + int2 + int3 + int4 + 4 * int5 + 16 * int6)
            self._integral_values_.append(int_value)
            self._n_splits_.append(n)

            if len(self._integral_values_) > 1 and np.abs(int_value - self._integral_values_[-2]) < self.eps:
                break
        self.int_value = self._integral_values_[-1]
        self.interval_num = self._n_splits_[-1]

    def plt_precison(self):
        """
        绘制自适应积分过程精度收敛曲线
        :return:
        """
        plt.figure(figsize=(8, 6))
        plt.plot(self._n_splits_, self._integral_values_, "ko-")
        plt.xlabel("The number of interval divisions", fontdict={"fontsize": 12})
        plt.ylabel(" Integration values", fontdict={"fontsize": 12})
        plt.title("Composite Double Simpson Integration Convergence Curve", fontdict={"fontsize": 14})
        plt.show()

