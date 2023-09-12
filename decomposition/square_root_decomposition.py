"""
======================
@author: 王方舟
@time: 2023-08-30:20:05
======================
"""
import numpy as np


class SquareRootDecompositionAlgorithm:
    """
    平方根分解法: cholesky分解和改进的平方根分解法
    """

    def __init__(self, A, b, sol_method="improved"):
        self.A = np.asarray(A, dtype=np.float64)
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("系数矩阵不是方阵,不能用高斯消元法求解!")
        else:
            self.n = self.A.shape[0]  # 矩阵维度
        self._check_symmetric_positive_definite_matrix_()  # 对称正定矩阵判断
        self.b = np.asarray(b, dtype=np.float64)
        if len(self.b) != self.n:
            raise ValueError("右端向量维度与系数矩阵维度不匹配!")
        # 平方根分解法的类型,平方根法cholesky和改进的平方根法improved
        self.sol_method = sol_method
        self.x, self.y = None, None  # 线性方程组的解
        self.eps = None  # 险证精度
        self.L, self.D = None, None  # A=LL T或A=LDL T

    def _check_symmetric_positive_definite_matrix_(self):
        """
        对称正定矩阵判断．采用自带函数det计算行列式值
        :return:
        """
        if (self.A == self.A.T).all():  # 对称
            if self.A[0, 0] > 0:
                for i in range(1, self.n):  # 各顺序主子式行列式大于0
                    if np.linalg.det(self.A[i:, i:]) <= 0:
                        raise ValueError("非正定矩阵.")
            else:
                raise ValueError("非正定知阵")
        else:
            raise ValueError("非对称知阵")

    def fit_solve(self):
        """
        cholesky分解和改进的平方恨分解法
        :return:
        """
        self.L, self.D = np.eye(self.n), np.zeros((self.n, self.n))
        self.y, self.x = np.zeros(self.n), np.zeros(self.n)
        if self.sol_method == "cholesky":  # 不选主元
            self.x = self._solve_cholesky_()
        elif self.sol_method == "improved":  # 改进的平方根法
            self.x = self._solve_improved_cholesky_()
        else:
            raise ValueError("仅适合doolittle LU分解法和选主元LU分解法")
        return self.x

    def _solve_cholesky_(self):
        """
        平方恨法(cholesky分解法)
        :return:
        """
        # 1.按照公式求解L
        for j in range(self.n):  # 每次循环求L的一列元素
            self.L[j, j] = np.sqrt(self.A[j, j] - sum(self.L[j, :j] ** 2))
            for i in range(j + 1, self.n):
                self.L[i, j] = (self.A[i, j] - np.dot(self.L[i, :j], self.L[j, :j])) / self.L[j, j]
        for i in range(self.n):
            self.y[i] = (self.b[i] - np.dot(self.L[i, :i], self.y[:i])) / self.L[i, i]
        for i in range(self.n - 1, -1, -1):
            self.x[i] = (self.y[i] - np.dot(self.L[i:, i], self.x[i:])) / self.L[i, i]
        self.eps = np.dot(self.A, self.x) - self.b
        return self.x

    def _solve_improved_cholesky_(self):
        self.D[0, 0] = self.A[0, 0]
        t = np.zeros((self.n, self.n))
        for i in range(1, self.n):
            for j in range(i):
                t[i, j] = self.A[i, j] - np.dot(t[i, :j], self.L[j, :j])
                self.L[i, j] = t[i, j] / self.D[j, j]
            self.D[i, i] = self.A[i, i] - np.dot(t[i, :i], self.L[i, :i])
        for i in range(self.n):
            self.y[i] = self.b[i] - np.dot(self.L[i, :i], self.y[:i])
        for i in range(self.n - 1, -1, -1):
            self.x[i] = self.y[i] / self.D[i, i] - np.dot(self.L[i:, i], self.x[i:])
        self.eps = np.dot(self.A, self.x) - self.b
        return self.x

