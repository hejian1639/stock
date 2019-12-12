# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 10)

y = x ** 2

plt.scatter(x, y)

plt.show()

one = np.ones(len(x))


def sigmod(arg_x, deriv=False):
    if (deriv == True):
        ret = arg_x.flatten()
        for i in np.nditer(ret, op_flags=['readwrite']):
            i[...] = i * (1 - i)
        return ret

    else:
        ret = arg_x.copy()
        for i in np.nditer(ret, op_flags=['readwrite']):
            i[...] = 1 / (1 + np.exp(-i))
        return ret


def linear(arg_x, deriv=False):
    if (deriv == True):
        ret = arg_x.flatten()
        for i in np.nditer(ret, op_flags=['readwrite']):
            i[...] = 1
        return ret

    else:
        ret = arg_x.copy()
        return ret


def relu(arg_x, deriv=False):
    if (deriv == True):
        ret = arg_x.flatten()
        for i in np.nditer(ret, op_flags=['readwrite']):
            if i > 0:
                i[...] = 1
            else:
                i[...] = 0
        return ret
    else:
        ret = arg_x.copy()
        for i in np.nditer(ret, op_flags=['readwrite']):
            if i < 0:
                i[...] = 0
        return ret


def back_propagation(arg_x, arg_y, w1, w2, activation):
    count = 0

    alpha = 1

    while True:

        h1 = np.dot(w1, arg_x)
        active = activation(h1)
        h2 = np.dot(w2, active)

        loss_derivative = arg_y - h2
        loss = np.dot(loss_derivative, np.transpose(loss_derivative))[0, 0]

        derivative = np.dot(loss_derivative, np.transpose(active))

        while True:
            nw2 = w2 + np.dot(alpha, derivative)
            h2 = np.dot(nw2, active)

            loss_derivative = arg_y - h2
            current = np.dot(loss_derivative, np.transpose(loss_derivative))[0, 0]
            if current > loss:
                alpha /= 2
                continue

            w2 = nw2

            break

        alpha *= 2

        h1 = np.dot(w1, arg_x)
        active = activation(h1)
        h2 = np.dot(w2, active)

        loss_derivative = arg_y - h2
        loss = np.dot(loss_derivative, np.transpose(loss_derivative))[0, 0]

        [row, col] = w1.shape

        derivative = np.kron(np.transpose(arg_x), np.eye(row))
        active_derivative = activation(active, deriv=True)
        derivative = np.dot(np.diag(active_derivative), derivative)
        derivative = np.dot(np.kron(loss_derivative, w2), derivative)
        derivative = np.transpose(derivative.reshape(col, row))
        # derivative = derivative.reshape(row, col)

        while True:
            nw1 = w1 + np.dot(alpha, derivative)
            h1 = np.dot(nw1, arg_x)
            active = activation(h1)
            h2 = np.dot(w2, active)

            loss_derivative = arg_y - h2
            current = np.dot(loss_derivative, np.transpose(loss_derivative))[0, 0]
            if current > loss:
                alpha /= 2
                continue

            w1 = nw1

            break

        alpha *= 2

        if count > 100:
            break

        count += 1

    print('loss=', loss)
    print('alpha=', alpha)
    print(count)

    return w1, w2


w1, w2 = back_propagation(np.vstack([x, one]), y,
                          np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
                          np.array([[0.0, 0.0, 0.0]]), relu)

print(w1)
print(w2)

h1 = np.dot(w1, np.vstack([x, one]))
active = relu(h1)
h2 = np.dot(w2, active)

plt.scatter(x, h2)

plt.show()
