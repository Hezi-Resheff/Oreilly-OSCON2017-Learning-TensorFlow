# -*- coding: utf-8 -*-

tf.reset_default_graph()
sess = tf.InteractiveSession()
with tf.name_scope('input_x'):
    x = tf.placeholder(tf.float32, [None, 784])
with tf.name_scope('input_label'):
    y_true = tf.placeholder(tf.float32, [None, 10]) 
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)
    
with tf.name_scope('weights'):    
    W = tf.Variable(tf.zeros([784, 10]))
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(W)
      tf.summary.scalar('mean', mean)
      tf.summary.histogram('histogram', W)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(W - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(W))
      tf.summary.scalar('min', tf.reduce_min(W))
      
with tf.name_scope('biases'):    
    b = tf.Variable(tf.zeros([10]))
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(b)
      tf.summary.scalar('mean', mean)
      tf.summary.histogram('histogram', b)

with tf.name_scope('Wx_b'):     
    y_pred = tf.add(tf.matmul(x, W),b)
    tf.summary.histogram('Wx_b', y_pred)

with tf.name_scope('loss'): 
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_pred, labels = y_true))
    tf.summary.scalar('loss', cross_entropy)
    
with tf.name_scope('train'):
    gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
with tf.name_scope('correct_pred'):
        correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
with tf.name_scope('accuracy'):
  accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))
  
  
# Merge all the summaries and write them out to LOG_DIR
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(os.path.join(DATA_DIR,"logs\\ex1\\train"), sess.graph)
test_writer = tf.summary.FileWriter(os.path.join(DATA_DIR,"logs\\ex1\\test"))
# Train
sess.run(tf.global_variables_initializer())

for i in range(NUM_STEPS):

    batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
    summary, _ = sess.run([merged, gd_step], feed_dict={x: batch_xs, y_true: batch_ys})
    train_writer.add_summary(summary, i)
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict={x: data.test.images, y_true: data.test.labels})
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
train_writer.close()
test_writer.close()        