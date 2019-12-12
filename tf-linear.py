# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
# from __future__ import print_function
import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(110).astype(np.float32)
deviation = np.random.rand(110).astype(np.float32)
y_data = x_data * 0.1 + 0.3 + (deviation - 0.5) * 0.01
x = tf.placeholder(tf.float32, name="x-input")
y = tf.placeholder(tf.float32, name='y-input')

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y_ = Weights * x + biases

batch_size = 10

loss = tf.losses.mean_squared_error(y, y_)
loss = tf.reduce_sum(tf.square(y - y_))
optimizer = tf.train.GradientDescentOptimizer(1)
optimizer = tf.train.AdagradOptimizer(1)
train = optimizer.minimize(loss)

### create tensorflow structure end ###

sess = tf.Session()
print(tf.__version__)

init = tf.global_variables_initializer()
sess.run(init)

# print(sess.run(Weights))

for step in range(100):
    start = (step * batch_size) % 100
    end = start + batch_size
    t, l = sess.run([train, loss], feed_dict={x: x_data[start:end], y: y_data[start:end]})

    # sess.run(train)
    # if step % 2 == 0:
    print(step, sess.run(Weights), sess.run(biases), l)
