# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import numpy as np
import matplotlib.pyplot as plt

length = 100

x = np.random.rand(length).astype(np.float32)
y = (x * 2 - 1) ** 2

# plt.scatter(x, y)
#
# plt.show()

one = np.ones(length)
x = np.vstack([x, one])

I_10 = np.eye(10)


def relu(arg_x):
    ret = arg_x.copy()
    for i in np.nditer(ret, op_flags=['readwrite']):
        if i < 0:
            i[...] = 0
    return ret


def relu_derivative(arg_x):
    ret = arg_x.flatten()
    for i in np.nditer(ret, op_flags=['readwrite']):
        if i > 0:
            i[...] = 1
        else:
            i[...] = 0
    return ret


def back_propagation(arg_x, arg_y, w1, w2, alpha, activation, activation_derivative_function):
    count = 0

    while True:
        h1 = np.dot(w1, arg_x)
        active = activation(h1)
        h2 = np.dot(w2, active)

        loss_derivative = arg_y - h2
        loss = np.dot(loss_derivative, np.transpose(loss_derivative))[0, 0]

        derivative = np.dot(loss_derivative, np.transpose(active))
        nw2 = w2 + np.dot(alpha, derivative)

        derivative = np.kron(np.transpose(arg_x), I_10)
        active_derivative = activation_derivative_function(h1)
        derivative = np.dot(np.diag(active_derivative), derivative)
        derivative = np.dot(np.kron(loss_derivative, w2), derivative)
        nw1 = w1 + np.dot(alpha, derivative.reshape(10, 2))

        h1 = np.dot(nw1, arg_x)
        active = activation(h1)
        h2 = np.dot(nw2, active)

        cur_delta = arg_y - h2
        current = np.dot(cur_delta, np.transpose(cur_delta))[0, 0]

        if current > loss:
            alpha /= 2
            continue

        if loss < 0.00001:
            break

        w1 = nw1
        w2 = nw2

        count += 1

    print('loss=', loss)
    print('alpha=', alpha)
    print(count)

    return w1, w2


print(back_propagation(x, y, np.zeros((10, 2)), np.zeros((1, 10)), 1, relu, relu_derivative))
