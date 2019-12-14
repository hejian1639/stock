# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import numpy as np
import matplotlib.pyplot as plt


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


def lrelu(l):
    def inner(arg_x, deriv=False):
        if (deriv == True):
            ret = arg_x.flatten()
            for i in np.nditer(ret, op_flags=['readwrite']):
                if i > 0:
                    i[...] = 1
                else:
                    i[...] = l
            return ret

        ret = arg_x.copy()
        for i in np.nditer(ret, op_flags=['readwrite']):
            if i < 0:
                i[...] = l * i
        return ret

    return inner


def back_propagation(arg_x, arg_y, w1, w2, activation):
    print('w1 = ', w1)
    print('w2 = ', w2)

    alpha = 1

    for count in range(10000):

        h1 = np.dot(w1, arg_x)
        active = activation(h1)
        h2 = np.dot(w2, active)

        loss_derivative = arg_y - h2
        loss = np.dot(loss_derivative, np.transpose(loss_derivative))[0, 0]

        w2_delta = np.dot(loss_derivative, np.transpose(active))

        [row, col] = w1.shape

        derivative = np.kron(np.transpose(arg_x), np.eye(row))
        active_derivative = activation(active.T, deriv=True)
        derivative = np.dot(np.diag(active_derivative), derivative)
        derivative = np.dot(np.kron(loss_derivative, w2), derivative)
        w1_delta = np.transpose(derivative.reshape(col, row))

        while True:
            nw1 = w1 + alpha * w1_delta
            nw2 = w2 + alpha * w2_delta
            h1 = np.dot(nw1, arg_x)
            active = activation(h1)
            h2 = np.dot(nw2, active)

            loss_derivative = arg_y - h2
            current = np.dot(loss_derivative, np.transpose(loss_derivative))[0, 0]
            if current - loss > 0:
                alpha /= 2
                continue

            w1 = nw1
            w2 = nw2

            break

        # if loss < 0.01:
        #     break

        if count % 100 == 0:
            # h1 = np.dot(w1, np.vstack([X, one]))
            # active = activation(h1)
            # h2 = np.dot(w2, active)
            #
            # loss_derivative = arg_y - h2
            # loss = np.dot(loss_derivative, np.transpose(loss_derivative))[0, 0]

            print('loss=', loss)
            print('alpha=', alpha)
            print(count)
            print('w1 = ', w1)
            print('w2 = ', w2)

            plt.scatter(X, h2)

            plt.show()

        alpha *= 2

    print('loss=', loss)
    print('alpha=', alpha)
    print(count)

    return w1, w2


X = np.linspace(-1, 1, 50)
one = np.ones(len(X))

Y = X ** 2

plt.scatter(X, Y)

plt.show()

hide_dim = 10

w1 = np.random.normal(0, 1, 2 * hide_dim).reshape(hide_dim, 2)
# w1 = np.ones(2 * hide_dim).reshape(hide_dim, 2)
w2 = np.zeros((1, hide_dim))

w1, w2 = back_propagation(np.vstack([X, one]), Y, w1, w2, relu)
