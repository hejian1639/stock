import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)  # 如果deriv为true，求导数
    return 1 / (1 + np.exp(-x))


def relu(x, deriv=False):
    ret = x.copy()
    if (deriv == True):
        for i in np.nditer(ret, op_flags=['readwrite']):
            if i > 0:
                i[...] = 1
            else:
                i[...] = 0
        return ret

    for i in np.nditer(ret, op_flags=['readwrite']):
        if i < 0:
            i[...] = 0
    return ret


X = np.linspace(-2, 2, 10)

Y = X ** 2

plt.scatter(X, Y)

plt.show()

one = np.ones(len(X))

# x = np.array([[0.35], [0.9]])  # 输入层
# y = np.array([[0.5]])  # 输出值
W0 = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
W1 = np.array([[0.0, 0.0, 0.0]])
print('original ', W0, '\n', W1)


def back_propagation(W0, W1, activation_fuction):
    for j in range(100):
        for (x, y) in zip(X.flat, Y.flat):
            l0 = np.array([[x], [1]])  # 相当于文章中x0
            l1 = activation_fuction(np.dot(W0, l0))  # 相当于文章中y1
            l2 = np.dot(W1, l1)  # 相当于文章中y2
            l2_error = y - l2
            Error = (y - l2) ** 2
            l2_delta = l2_error  # this will backpack
            l1_error = l2_delta * W1  # 反向传播
            l1_delta = l1_error.T * activation_fuction(l1, deriv=True)
            W1 += 0.1 * l2_delta * l1.T  # 修改权值
            W0 += 0.1 * np.kron(l1_delta, l0.T)

    return W0, W1, Error


W0, W1, Error = back_propagation(W0, W1, relu)
print("Error:", Error)
print(W0, '\n', W1)

h1 = np.dot(W0, np.vstack([X, one]))
active = relu(h1)
h2 = np.dot(W1, active)

plt.scatter(X, h2)

plt.show()
