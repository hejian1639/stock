# 局部加权线性回归
import numpy as np
import matplotlib.pyplot as plt


# 局部加权线性回归函数
def lwlr(x_i, x, y, k=1.0):
    # 读入数据并创建所需矩阵
    m = len(x)
    one = np.ones(length)
    xMat = np.mat((x, one))
    # 权重，创建对角矩阵，维数与xMat维数相同
    weights = np.mat(np.eye((m)))  # m维的单位对角矩阵

    w = np.zeros(length)  # m维的单位对角矩阵
    '''
    权重矩阵是一个方阵,阶数等于样本点个数。也就是说,该矩阵为每个样本点初始
        化了一个权重。接着,算法将遍历数据集,计算每个样本点对应的权重值,
    '''
    for j in range(m):
        diff = x[j] - x_i

        # 采用高斯核函数进行权重赋值，样本附近点将被赋予更高权重
        w[j] = weights[j, j] = np.exp(-diff * diff / k) / np.sqrt(2 * np.pi)

    theta = (xMat * weights * xMat.T).I * xMat * weights * np.mat(y).T
    print(theta)

    return np.mat((x_i, 1)) * theta


# 样本点依次做局部加权
def lwlrTest(x, y, k=1.0):
    m = len(x)
    yHat = np.zeros(m)
    for i in range(m):  # 为样本中每个点，调用lwlr()函数计算ws值以及预测值yHat
        yHat[i] = lwlr(x[i], x, y, k)
    return yHat


np.random.seed(1)

bias = 0

length = 100
# create data
x = np.random.rand(length).astype(np.float32)
# x *= 2
# x -= 1

y = np.power(x, 2)  # shape (100, 1) + some noise

plt.scatter(x, y)
plt.show()

yHat = lwlrTest(x, y, 1)

plt.scatter(x, yHat)
plt.show()
