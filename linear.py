# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import numpy as np

length = 100
# create data
x = np.random.rand(length).astype(np.float32)
z = np.random.rand(length).astype(np.float32)
deviation = np.random.rand(length).astype(np.float32)
y = x * 0.1 + + z * 0.2 + 0.3 + (deviation - 0.5) * 0.01

one = np.ones(length)

# plt.scatter(x, y)
#
# plt.show()

x = np.vstack([x, z, one])


def h(arg_w, arg_x):
    return np.dot(arg_w, arg_x)


def derivative(arg_y, hypothesis):
    return arg_y - hypothesis


def back_propagation(arg_x, arg_y, w, hypothesis, alpha):
    h_y = hypothesis(w, arg_x)
    delta = arg_y - h_y

    loss = arg_y - h_y

    count = 0

    while True:

        theta = np.dot(delta, np.transpose(x))
        nw = w + np.dot(alpha, theta)

        evalue = np.dot(theta, np.transpose(theta))[0, 0]

        h_y = hypothesis(nw, arg_x)
        cur_delta = arg_y - h_y
        current = arg_y - h_y
        if current > loss:
            alpha /= 2
            continue

        if evalue < 0.00001:
            break

        loss = current
        delta = cur_delta

        w = nw

        count += 1

    print('evalue=', evalue)
    print('alpha=', alpha)
    print(count)

    return w


print(back_propagation(x, y, np.zeros((1, 3)), h, 1, derivative))
