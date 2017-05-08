"""
We construct a confusion matrix for the softmax regression on MNIST digits.
1. Model is built and run like before.
2. During the test phase we use y_true, y_pred in the fetch argument, since these
are the vars we will need for the confusion matrix.
3. We then use the built-in confusion_matrix method in sklearn.metrics to complete
the task.
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
y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)

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
    # Here we use the fetches [y_true, y_pred] since those are the vars we will need to
    # construct the confusion matrix.
    y_true_vec, y_pred_vec = sess.run([y_true, y_pred],
                              feed_dict={x: data.test.images, y_true: data.test.labels})

# confusion_matrix() requires the actual predictions, not the probability vectors, so we use
# .argmax(axis=1) to select the class with the largest probability.
conf_mat = confusion_matrix(y_true_vec.argmax(axis=1), y_pred_vec.argmax(axis=1))

# pd.DataFrame is used for the nice print format
print(pd.DataFrame(conf_mat))
