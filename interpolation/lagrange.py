"""
======================
@author: 王方舟
@time: 2023-08-17:11:24
======================
"""
import numpy as np
import sympy as sp
from interpolation.utils import interp_utils


class LagrangeInterpolation:
    """
    拉格朗日插值
    """

    def __init__(self, x, y):
        """
        拉格朗日参数初始化
        :param x: 已知数据的x坐标点
        :param y: 已知数据的y坐标点
        """
        # 类型转换array
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        if len(self.x) > 1 and len(self.x) == len(self.y):
            self.n = len(self.x)  # 离散数据点个数
        else:
            raise ValueError("插值数据(x,y)维度不匹配!")
        self.polynomial = None  # 插值多项式(符号表示)
        self.poly_coefficient = None  # 插值多项式系数向量
        self.coefficient_order = None  # 对应多项式系数阶次
        self.y0 = None  # 所求插值点的值(单个值或向量)

    def fit_interp(self):
        """
        拉格朗日插值多项式生成
        :return:
        """
        t = sp.Symbol("t")  # 定义符号t
        self.polynomial = 0.0  # 实例化插值多项式
        for i in range(self.n):
            # 针对每个数据点,构造插值基函数
            basis_fun = self.y[i]  # 插值基函数
            for j in range(self.n):
                if i != j:
                    basis_fun *= (t - self.x[j]) / (self.x[i] - self.x[j])
            self.polynomial += basis_fun  # 插值多项式累加
        # 插值多项式特征
        self.polynomial = sp.expand(self.polynomial)  # 多项式展开
        polynomial = sp.Poly(self.polynomial, t)  # 根据多项式构造多项式对象
        self.poly_coefficient = polynomial.coeffs()  # 获取多项式的系数
        self.coefficient_order = polynomial.monoms()  # 多项式系数对应的阶次

    def cal_interp_x0(self, x0):
        """
        计算给定插值点数值
        :param x0:
        :return:
        """
        self.y0 = interp_utils.cal_interp_x0(self.polynomial, x0)
        return self.y0

    def plt_interpolation(self, x0=None, y0=None):
        """
        可视化插值图像和所求的插值点
        :param x0:
        :param y0:
        :return:
        """
        params = (self.polynomial, self.x, self.y, "Lagrange", x0, y0)
        interp_utils.plt_interpolation(params)
