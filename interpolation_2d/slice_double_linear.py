"""
======================
@author: 王方舟
@time: 2023-09-01:11:39
======================
"""
import numpy as np
from matplotlib import pyplot as plt


class SliceDoubleLinear:
    """
    二维分片双线性插值
    """

    def __init__(self, x, y, z, x0, y0):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.Z = np.asarray(z, dtype=np.float64)
        if self.Z.shape[0] != len(self.x) or self.Z.shape[1] != len(self.y):
            raise ValueError("离散数据插值组(x,y,Z)维度不匹配!")
        if len(x0) == len(y0):
            self.x0 = np.asarray(x0, dtype=np.float64)
            self.y0 = np.asarray(y0, dtype=np.float64)
            self.n0 = len(self.x0)  # 所求插值点的个数
        else:
            raise ValueError("所求插值点(x0,y0)的维度不匹配!")
        self.n_x, self.n_y = len(self.x), len(self.y)
        self.Z0 = None  # 所求插值点(x0, y0)所对应的Z0值

    def fit_2d_interp(self):
        self.Z0 = np.zeros(self.n0)  # 初始化
        for k in range(self.n0):
            # 针对每一对(x0[k], y0[k])求解插值Z0[k]
            Lxy = self._fit_bilinear_(self.x0[k], self.y0[k])
            v_ = np.array([1, self.x0[k], self.y0[k], self.x0[k] * self.y0[k]])
            self.Z0[k] = np.dot(v_, Lxy)
        return self.Z0

    def _fit_bilinear_(self, x_k, y_k):
        """
        求解所求插值点对(x_k, y_k)所在的分片双线性函数,即求a, b, c, d
        :param x_k:
        :param y_k:
        :return:
        """
        idx, idy = self._find_index_(x_k, y_k)  # 查找xk和yk所在的子区间段索引
        x_1i, xi = self.x[idx], self.x[idx + 1]  # 分片上的x值
        y_1i, yi = self.y[idy], self.y[idy + 1]  # 分片上的y值#构造矩阵求解a,b. c. d
        node_mat = np.array([[1, x_1i, y_1i, x_1i * y_1i], [1, xi, yi, xi * yi],
                             [1, x_1i, yi, x_1i * yi], [1, xi, y_1i, xi * y_1i]])
        vector_z = np.array([self.Z[idx, idy], self.Z[idx + 1, idy + 1],
                             self.Z[idx, idy + 1], self.Z[idx + 1, idy]])
        coefficient = np.linalg.solve(node_mat, vector_z)
        return coefficient

    def _find_index_(self, x_k, y_k):
        """
        查找xk和yk所在的子区间段索引
        :param x_k:
        :param y_k:
        :return:
        """
        idx, idy = np.infty, np.infty
        for i in range(self.n_x - 1):
            if self.x[i] <= x_k <= self.x[i + 1] or self.x[i + 1] <= x_k <= self.x[i]:
                idx = i
                break
        for j in range(self.n_y - 1):
            if self.y[j] <= y_k <= self.y[j + 1] or self.y[j + 1] <= y_k <= self.y[j]:
                idy = j
                break
        if idx is np.infty or idy is np.infty:
            raise ValueError("所求插值点(x0, y0)在区间外, 不能进行外插!")
        return idx, idy

    def plt_3d_surface(self):
        """
        可视化三维图像
        :return:
        """

        def _cal_xy_plt3d_(x_i, y_i):
            """求解模拟划分的数据点所对应的二维插值zi"""
            lxy = self._fit_bilinear_(x_i, y_i)
            v_ = np.array([1, x_i, y_i, x_i * y_i])
            return np.dot(v_, lxy)

        x = np.linspace(min(self.x), max(self.x), 100)
        y = np.linspace(min(self.y), max(self.y), 100)
        xi, yi = np.meshgrid(x, y)
        zi = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                zi[i, j] = _cal_xy_plt3d_(xi[i, j], yi[i, j])
        # 三维绘图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(xi, yi, zi.T, cmap=plt.get_cmap("coolwarm"), linewidth=0)
        plt.title("Slice double linear interpolation", fontdict={"fontsize": 16})
        ax.set_xlabel("x(100 equal sections)", fontdict={"fontsize": 14})
        ax.set_ylabel("Y(100 equal sections)", fontdict={"fontsize": 14})
        ax.set_zlabel("Z", fontdict={"fontsize": 14})
        ax.grid(ls=":")
        plt.show()
