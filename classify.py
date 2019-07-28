# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
# from __future__ import print_function
import tensorflow as tf
from numpy.random import RandomState
import matplotlib.pyplot as plt

# create data
rdm = RandomState(1)
X = rdm.rand(100, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y = tf.placeholder(tf.float32, name='y-input')

### create tensorflow structure start ###
weight = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
biase = tf.Variable(tf.zeros([1]))

y_ = tf.nn.sigmoid(tf.matmul(x, weight) + biase)

cross_entropy = -tf.reduce_mean((y * tf.log(y_) + ((1 - y) * tf.log(1 - y_))))
# optimizer = tf.train.GradientDescentOptimizer(1)
optimizer = tf.train.AdamOptimizer(1)
train = optimizer.minimize(cross_entropy)

### create tensorflow structure end ###

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

batch_size = 100

plt.ion()  # something about plotting

for step in range(100):
    start = (step * batch_size) % 100
    end = start + batch_size
    sess.run(train, feed_dict={x: X[start:end], y: Y[start:end]})

    # sess.run(train)
    if step % 10 == 0:
        plt.cla()
        plt.scatter(X[:, 0], X[:, 1])
        w = sess.run(weight)
        b = sess.run(biase)
        print("step = {}, weight = {}, biase = {}".format(step, w, b))
        plt.plot([-b[0] / w[0][0], 0], [0, -b[0] / w[1][0]], 'r-')
        plt.pause(0.1)

plt.ioff()
plt.show()
