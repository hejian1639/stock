# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import numpy as np
import tensorflow as tf

# create data
np.random.seed(1)
x_data = np.random.rand(100).astype(np.float32)
np.random.seed(2)
deviation = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3 + (deviation - 0.5) * 0.01
x = tf.placeholder(tf.float32, name="x-input")
y = tf.placeholder(tf.float32, name='y-input')

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0), "Weights")
Weights = tf.Variable(tf.zeros([1]), "Weights")
biases = tf.Variable(tf.zeros([1]), "biases")

y_ = Weights * x + biases
tf.summary.tensor_summary('y', y_)

batch_size = 10

loss = tf.losses.mean_squared_error(y, y_)
loss = tf.reduce_sum((y - y_) ** 2)
tf.summary.scalar('loss', loss)  # add loss to scalar summary
optimizer = tf.train.GradientDescentOptimizer(0.001)
# optimizer = tf.train.AdagradOptimizer(1)
train = optimizer.minimize(loss)

writer = tf.summary.FileWriter("tf-linear", tf.get_default_graph())
writer.close()

### create tensorflow structure end ###

print(tf.__version__)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run(y_, feed_dict={x: x_data}))

    for step in range(10):
        _, l = sess.run([train, loss], feed_dict={x: x_data, y: y_data})

        print(step, sess.run(Weights), sess.run(biases), l)
