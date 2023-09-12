"""
======================
@author: 王方舟
@time: 2023-09-02:17:49
======================
"""
import numpy as np
from matplotlib import pyplot as plt


class BivariateThreePointsLagrange:
    """
    分片二元三点拉格朗日插值
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
        self.Z0 = np.zeros(self.n0)  # 所求离散插值点(x0, yO)的插值
        for k in range(self.n0):
            # 针对每一对(x0[k], y0[k])求解插值Z0[k]
            self.Z0[k] = self._cal_xy_interp_val_(self.x0[k], self.y0[k])
        return self.Z0

    def _cal_xy_interp_val_(self, xi, yi):
        """
        求解所给(x, y)的插值z
        :param xi: 单个插值点的x坐标值
        :param yi: 单个插值点的y坐标值
        :return:
        """
        # idx和idy分别是三个最靠近点的索引下标向量
        idx, idy = self._find_index_slice_(xi, yi)  # 查找最靠近(x, y)的三个点索引下标
        val = 0.0
        for i in range(3):
            i1, i2 = np.mod(i + 1, 3), np.mod(i + 2, 3)
            val_x = (xi - self.x[idx[i1]]) * (xi - self.x[idx[i2]]) / (self.x[idx[i]] - self.x[idx[i1]]) / (
                        self.x[idx[i]] - self.x[idx[i2]])
            for j in range(3):
                j1, j2 = np.mod(j + 1, 3), np.mod(j + 2, 3)
                val_y = (yi - self.y[idy[j1]]) * (yi - self.y[idy[j2]]) / (self.y[idy[j]] - self.y[idy[j1]]) / (
                            self.y[idy[j]] - self.y[idy[j2]])
                # 边界情况:
                if idx[i] == self.n_x - 1 and idy[j] < self.n_y - 1:  # x边界，非y边界
                    val += val_x * val_y * self.Z[-1, idy[j]]
                elif idx[i] < self.n_x - 1 and idy[j] == self.n_y - 1:  # y边界，非x边界
                    val += val_x * val_y * self.Z[idx[i], -1]
                elif idx[i] == self.n_x - 1 and idy[j] == self.n_y - 1:  # x边界，y边界
                    val += val_x * val_y * self.Z[-1, -1]
                else:  # 非边界情况
                    val += val_x * val_y * self.Z[idx[i], idy[j]]
        return val

    def _find_index_slice_(self, xi, yi):
        """
        查找xk和yk所在的分片索引
        :return:
        """
        idx, idy = np.infty, np.infty
        for i in range(self.n_x - 1):
            if self.x[i] <= xi <= self.x[i + 1] or self.x[i + 1] <= xi <= self.x[i]:
                idx = i
                break
        for j in range(self.n_y - 1):
            if self.y[j] <= yi <= self.y[j + 1] or self.y[j + 1] <= yi <= self.y[j]:
                idy = j
                break
        if idx is np.infty or idy is np.infty:
            raise ValueError("所求插值点(x0, y0)在区间外, 不能进行外插!")

        # 查找xi值所在区间的最近三个点索引
        if idx:  # 不在第一个区间, 即不在x轴第一个分片上
            if idx == self.n_x - 2:  # 所求点在x轴上最后一个分片
                near_idx = np.array([self.n_x - 3, self.n_x - 2, self.n_x - 1])
            else:  # 所求点在区间内部,非边界
                if np.abs(self.x[idx - 1] - xi) > np.abs(self.x[idx + 2] - xi):
                    near_idx = np.array([idx, idx + 1, idx + 2])
                else:
                    near_idx = np.array([idx - 1, idx, idx + 1])
        else:  # 在第一个区间,即x轴第一个分片上
            near_idx = np.array([0, 1, 2])  # 离散数据点的索引下标

        # 查找yi值所在区间的最近三个点索引
        if idy:  # 不在第一个区间, 即不在y轴第一个分片上
            if idy == self.n_y - 2:  # 所求点在y轴上最后一个分片
                near_idy = np.array([self.n_y - 3, self.n_y - 2, self.n_y - 1])
            else:  # 所求点在区间内部,非边界
                if np.abs(self.y[idy - 1] - yi) > np.abs(self.y[idy + 2] - yi):
                    near_idy = np.array([idy, idy + 1, idy + 2])
                else:
                    near_idy = np.array([idy - 1, idy, idy + 1])
        else:  # 在第一个区间,即y轴第一个分片上
            near_idy = np.array([0, 1, 2])  # 离散数据点的索引下标

        return near_idx, near_idy

    def plt_3d_surface(self):
        """
        可视化三维图像
        :return:
        """
        x = np.linspace(min(self.x), max(self.x), 100)
        y = np.linspace(min(self.y), max(self.y), 100)
        xi, yi = np.meshgrid(x, y)
        zi = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                zi[i, j] = self._cal_xy_interp_val_(xi[i, j], yi[i, j])
        # 三维绘图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(xi, yi, zi.T, cmap=plt.get_cmap("coolwarm"), linewidth=0)
        plt.title("Bivariate three points lagrange interpolation", fontdict={"fontsize": 14})
        ax.set_xlabel("x(100 equal sections)", fontdict={"fontsize": 12})
        ax.set_ylabel("Y(100 equal sections)", fontdict={"fontsize": 12})
        ax.set_zlabel("Z", fontdict={"fontsize": 12})
        ax.grid(ls=":")
        plt.show()
