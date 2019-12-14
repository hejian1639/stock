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


def lrelu(l):
    def inner(x, deriv=False):
        ret = x.copy()
        if (deriv == True):
            for i in np.nditer(ret, op_flags=['readwrite']):
                if i > 0:
                    i[...] = 1
                else:
                    i[...] = l
            return ret

        for i in np.nditer(ret, op_flags=['readwrite']):
            if i < 0:
                i[...] = l * i
        return ret

    return inner


def back_propagation(W0, W1, activation_fuction):
    for j in range(10000):
        for (x, y) in zip(X.flat, Y.flat):
            l0 = np.array([[x], [1]])  # 相当于文章中x0
            l1 = activation_fuction(np.dot(W0, l0))  # 相当于文章中y1
            l2 = np.dot(W1, l1)  # 相当于文章中y2
            l2_error = y - l2
            Error = (y - l2) ** 2

            l2_delta = l2_error  # this will backpack
            l1_error = l2_delta * W1  # 反向传播
            l1_delta = l1_error.T * activation_fuction(l1, deriv=True)
            W1 += 0.01 * l2_delta * l1.T  # 修改权值
            W0 += 0.01 * np.kron(l1_delta, l0.T)

        if j % 100 == 0:
            print("j:", j)
            print("Error:", Error)
            print(W0, '\n', W1)

            h1 = np.dot(W0, np.vstack([X, one]))
            active = activation_fuction(h1)
            h2 = np.dot(W1, active)

            plt.scatter(X, h2)

            plt.show()

    return W0, W1, Error


X = np.linspace(-1, 1, 50)

Y = X ** 2

plt.scatter(X, Y)

plt.show()

one = np.ones(len(X))

hide_dim = 10
W0 = np.array([[-3.97345031e-01, -1.05253014e+00],
               [-8.78409821e+01, -1.32620785e+02],
               [-1.09653821e+01, -1.24476429e+01],
               [7.97194597e-01, 1.26352673e+00],
               [-1.47757630e+01, -2.05247259e+01],
               [-3.13047652e+01, -4.34848426e+01],
               [-5.75327308e+01, -7.80573821e+01],
               [2.07187770e-01, -4.28254594e-01],
               [-1.10081585e+02, -1.45108082e+02],
               [-2.66475502e+02, -3.57186891e+02]])
# W0 = np.random.normal(0, 1, 2 * hide_dim).reshape(hide_dim, 2)
W1 = np.zeros((1, hide_dim))
W1 = np.array([[0., -1.7806362, -0.17524982, 0.48986932, -0.24071698, -0.50999657,
                -0.95531785, 0., -1.79742974, -4.41420607]])
print('original ', W0, '\n', W1)

W0, W1, Error = back_propagation(W0, W1, relu)
print("Error:", Error)
print(W0, '\n', W1)
