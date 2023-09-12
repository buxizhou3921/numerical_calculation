"""
======================
@author: 王方舟
@time: 2023-08-17:11:50
======================
"""

from interpolation.lagrange import LagrangeInterpolation
import numpy as np

if __name__ == "__main__":
    # x = np.linspace(0, 2 * np.pi, 10, endpoint=True)
    # y = np.sin(x)
    # x0 = np.array([np.pi / 2, 2.158, 3.58, 4.784])
    x = np.linspace(0, 24, 13, endpoint=True)
    y = np.array([12, 9, 9, 10, 18, 24, 28, 27, 25, 20, 18, 15, 13])
    x0 = np.array([1, 10.5, 13, 18.7, 22.3])

    lag_interp = LagrangeInterpolation(x=x, y=y)
    lag_interp.fit_interp()

    print("拉格朗日多项式:")
    print(lag_interp.polynomial)
    print("拉格朗日插值多项式系数向量和对应阶次:")
    print(lag_interp.poly_coefficient)
    print(lag_interp.coefficient_order)

    y0 = lag_interp.cal_interp_x0(x0)
    print("所求插值点的值: ", y0, "精确值是: ", np.sin(x0))

    lag_interp.plt_interpolation(x0, y0)
