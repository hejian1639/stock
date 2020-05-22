# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import numpy as np
import tensor

length = 100
np.random.seed(1)
data_x = np.random.rand(length).astype(np.float32)
# print('data_x = ', data_x)

np.random.seed(2)
deviation = np.random.rand(length).astype(np.float32)
# print('deviation = ', deviation)
data_y = data_x * 0.1 + 0.3 + (deviation - 0.5) * 0.01

w = tensor.Variable(0)
b = tensor.Variable(0)
x = tensor.Input()
y = tensor.Input()
t_y = w * x + b

# derivative = ((w * x) ** 2).derivative()
# res = run(derivative, {x: data_x, y: data_y})
# print('derivative = ', res)

# derivative = ((y - t_y) ** 2).derivative()
# res = run(derivative, {x: data_x, y: data_y})
# print('derivative = ', res)

loss = tensor.sum((y - t_y) ** 2)
# derivative = loss.derivative()
# res = run(derivative, {x: data_x, y: data_y})
# print('derivative = ', res)


train = tensor.minimize(loss)

for step in range(100):
    print('loss = ', tensor.run(train, {x: data_x, y: data_y}))
    print('w = ', tensor.run(w), ' b = ', tensor.run(b), 'step = ', train.step)
