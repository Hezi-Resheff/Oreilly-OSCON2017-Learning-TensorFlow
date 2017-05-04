"""
We add a bias term to the regression model. To do so we need to change only 2 lines of code.
First, we define a bias variable (one for each of the 10 digits):
b = tf.Variable(tf.zeros([10]))

Next, we add it to the model:
y_pred = tf.matmul(x, W) + b

So the model is now:
    y_pred(i) = <x, w_i> + b_i,
and in matrix form:
    y_pred = Wx + b
"""
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import pandas as pd
from sklearn.metrics import confusion_matrix

# This is where the MNIST data will be downloaded to. If you already have it on your
# machine then set the path accordingly to prevent an extra download.
DATA_DIR = '/tmp/data' if not 'win' in sys.platform else "c:\\tmp\\data"

# Load data
data = input_data.read_data_sets(DATA_DIR, one_hot=True)

NUM_STEPS = 1000
MINIBATCH_SIZE = 100

# We start by building the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W) + b

cross_entropy = \
    tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:

    # Train
    sess.run(tf.global_variables_initializer())
    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})

    # Test
    is_correct, acc = sess.run([correct_mask, accuracy],
                               feed_dict={x: data.test.images, y_true: data.test.labels})

print("Accuracy: {:.4}%".format(acc*100))