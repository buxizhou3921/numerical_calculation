"""
======================
@author: 王方舟
@time: 2023-09-01:23:11
======================
"""
import numpy as np
from interpolation_2d.slice_double_linear import SliceDoubleLinear

if __name__ == "__main__":
    # x = np.linspace(-2, 2, 25)
    # y = np.linspace(-2, 2, 25)
    # xi, yi = np.meshgrid(x, y)
    # Z = xi * np.exp(- xi ** 2 - yi ** 2)
    # x0 = np.array([-1.5, -0.58, 0.58, 1.65])
    # y0 = np.array([-1.25, -0.69, 0.76, 1.78])
    # sdl = SliceDoubleLinear(x, y, Z.T, x0, y0)
    # Z0 = sdl.fit_2d_interp()
    # print("插值点值: ", Z0)
    # print("精确值是: ", x0 * np.exp(- x0 ** 2 - y0 ** 2))
    # sdl.plt_3d_surface()

    x = np.linspace(1, 6, 25)
    y = np.linspace(2, 7, 25)
    xi, yi = np.meshgrid(x, y)
    Z = np.sin(xi) * np.cos(yi)
    x0 = np.array([1.5, 2.58, 3.58, 4.65])
    y0 = np.array([2.25, 3.69, 3.76, 5.78])
    sdl = SliceDoubleLinear(x, y, Z.T, x0, y0)
    Z0 = sdl.fit_2d_interp()
    print("插值点值: ", Z0)
    print("精确值是:", np.sin(x0) * np.cos(y0))
    sdl.plt_3d_surface()
