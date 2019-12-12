"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
matplotlib
numpy
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

bias = 0
# fake data
x = np.linspace(-2, 2, 10)[:, np.newaxis]  # shape (100, 1)
y = np.power(x, 2) + bias  # shape (100, 1) + some noise

# plot data
plt.scatter(x, y)
plt.show()

tf_x = tf.placeholder(tf.float32, x.shape)  # input x
tf_y = tf.placeholder(tf.float32, y.shape)  # input y


def add_input_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def add_hide_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.zeros([in_size, out_size]))
    Wx = tf.matmul(inputs, Weights)
    if activation_function is None:
        outputs = Wx
    else:
        outputs = activation_function(Wx)
    return outputs


l1 = add_input_layer(tf_x, 1, 3, activation_function=tf.nn.relu)
# add output layer
output = add_hide_layer(l1, 3, 1, activation_function=None)

# neural network layers
# l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)  # hidden layer
# output = tf.layers.dense(l1, 1)  # output layer

global_step = tf.Variable(0)
LEARNING_RATE = tf.train.exponential_decay(1.0, global_step, 1, 0.99, staircase=True)
LEARNING_RATE = 1

loss = tf.losses.mean_squared_error(tf_y, output)  # compute cost
loss = tf.reduce_mean(tf.square(tf_y - output))
# optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
optimizer = tf.train.AdagradOptimizer(LEARNING_RATE)
train_op = optimizer.minimize(loss, global_step=global_step)

sess = tf.Session()  # control training and others
sess.run(tf.global_variables_initializer())  # initialize var in graph

plt.ion()  # something about plotting

for step in range(1000):
    # train and net output
    _, l, pred = sess.run([train_op, loss, output], {tf_x: x, tf_y: y})
    if step % 100 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x, y)
        plt.plot(x, pred, 'r-')
        plt.text(0.5, bias, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
        # plt.text(-0.5, bias+1, 'LEARNING_RATE=%f' % sess.run(LEARNING_RATE), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
