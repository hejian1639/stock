# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import matplotlib.pyplot as plt
import numpy as np

length = 100
# create data
x = np.random.rand(length).astype(np.float32)
deviation = np.random.rand(length).astype(np.float32)
y = x * 0.1 + 0.3 + (deviation - 0.5) * 0.01

one = np.mat(np.ones((length, 1)))

plt.scatter(x, y)

plt.show()

w = 0
b = 0
alpha = 1


def h(arg_w, arg_b):
    return x * arg_w + arg_b


delta = y - h(w, b)
last = np.dot(delta, delta)

count = 0
while True:

    theta = np.mat(delta) * np.mat(x).T
    nw = w + alpha * theta[0, 0]
    theta = np.mat(delta) * one
    nb = b + alpha * theta[0, 0]

    cur_delta = y - h(nw, nb)
    current = np.dot(cur_delta, cur_delta)
    if current > last:
        alpha /= 2
        continue

    last = current
    delta = cur_delta
    if alpha < 0.01:
        break

    w = nw
    b = nb

    print('alpha=', alpha)
    print(w, b)

    count += 1

print(count)
