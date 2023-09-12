"""
======================
@author: 王方舟
@time: 2023-08-31:14:59
======================
"""

import numpy as np


class RungeKutta4:
    def __init__(self, func, x_span, y0, n):
        """
        初始化函数
        :param func:导函数f(x,y)
        :param x_span: x的范围
        :param y0: Y的初始值
        :param n: 分割数
        """
        self.func = func
        self.n = n
        self.x_array = np.linspace(x_span[0], x_span[1], n)
        self.y_array = np.zeros(n)
        self.y_array[0] = y0
        self.h = (x_span[1] - x_span[0]) / (n - 1)

    def main(self):
        """
        4阶龙格库塔算法
        :return: x_array
        :return: y_array
        """
        for i in range(0, self.n - 1):
            K1 = self.func(self.x_array[i], self.y_array[i])
            K2 = self.func(self.x_array[i] + self.h / 2, self.y_array[i] + self.h / 2 * K1)
            K3 = self.func(self.x_array[i] + self.h / 2, self.y_array[i] + self.h / 2 * K2)
            K4 = self.func(self.x_array[i] + self.h, self.y_array[i] + self.h * K3)
            self.y_array[i + 1] = self.y_array[i] + self.h / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
        return self.x_array, self.y_array
