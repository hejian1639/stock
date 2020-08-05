# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import matplotlib.pyplot as plt
import numpy as np
import tensor

np.random.seed(1)

bias = 0
# fake data
x = np.linspace(-2, 2, 100)[:, np.newaxis]  # shape (100, 1)
y = np.power(x, 2) + bias  # shape (100, 1) + some noise

# plot data
plt.scatter(x, y)
plt.show()

tf_x = tensor.Input()
tf_y = tensor.Input()


def add_input_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tensor.Variable(np.random.normal(size=in_size * out_size).reshape(in_size, out_size))
    biases = tensor.Variable(np.zeros([1, out_size]))
    Wx_plus_b = Weights * inputs + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def add_hide_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tensor.Variable(np.zeros([in_size, out_size]))
    Wx = Weights * inputs
    if activation_function is None:
        outputs = Wx
    else:
        outputs = activation_function(Wx)
    return outputs

l1 = add_input_layer(x, 1, 10, activation_function=tensor.relu)
# l1 = add_input_layer(x, 1, 10)

# l1.derivative()


# add output layer
y = add_hide_layer(l1, 10, 1, activation_function=None)

loss = tensor.sum((y - tf_y) ** 2)
train = tensor.minimize(loss)

for step in range(100):
    print('loss = ', tensor.run(train, {tf_x: x, tf_y: y}))
