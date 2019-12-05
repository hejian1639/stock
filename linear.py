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
deviation = np.random.rand(length).astype(np.float32)
y = x * 0.1 + 0.3 + (deviation - 0.5) * 0.01

one = np.ones(length)


# plt.scatter(x, y)
#
# plt.show()


def h(arg_w, arg_x):
    return np.dot(arg_w, arg_x)


x = np.vstack([x, one])


def back_propgation(arg_x, arg_y, w, hypothesis, alpha):
    delta = arg_y - hypothesis(w, arg_x)

    last = np.dot(delta, np.transpose(delta))

    count = 0

    while True:

        theta = np.dot(delta, np.transpose(x))
        nw = w + np.dot(alpha, theta)

        evalue = np.dot(theta, np.transpose(theta))[0, 0]

        cur_delta = y - h(nw, x)
        current = np.dot(cur_delta, np.transpose(cur_delta))
        if current > last:
            alpha /= 2
            continue

        if evalue < 0.00001:
            break

        last = current
        delta = cur_delta

        w = nw

        count += 1

    print('evalue=', evalue)
    print('alpha=', alpha)
    print(count)

    return w


print(back_propgation(x, y, np.zeros((1, 2)), h, 1))
