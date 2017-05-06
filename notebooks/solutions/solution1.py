
import tensorflow as tf
import numpy as np 

g = tf.Graph()
# graph A
with g.as_default():
    a = tf.constant(5)
    b = tf.constant(2)
    c = tf.constant(3)
    d = tf.multiply(a,b)
    e = tf.add(b,c)
    f = tf.subtract(d,e)
    with tf.Session() as sess:
        outs = sess.run([a,b,c,d,e,f])
        
print('Outputs for graph A: {}'.format(outs))
    
    
# graph B
with g.as_default():
    a = tf.constant(5,dtype=tf.float32)
    b = tf.constant(3,dtype=tf.float32)
    c = tf.multiply(a,b)
    d = tf.sin(c)
    e = tf.divide(b,d)
    with tf.Session() as sess:
        outs = sess.run([a,b,c,d,e])
        
print('Outputs for graph B: {}'.format(outs))

# graph C
with g.as_default():
    a = tf.constant(2,dtype=tf.int32)
    b = tf.constant(5,dtype=tf.int32)
    c = tf.multiply(a,b)
    d = tf.add(a,b)
    e = tf.floordiv(b,a)
    f = tf.subtract(d,e)
    with tf.Session() as sess:
        outs = sess.run([a,b,c,d,e,f])
        
print('Outputs for graph C: {}'.format(outs))