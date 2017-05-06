

with g.as_default():
    x = tf.constant([[5,3,2],[1,3,-5],[0,1,-8]],shape=[3,3],dtype=tf.float32)
    w = tf.constant([2,-5,4],shape=[3,1],dtype=tf.float32)
    c = tf.matmul(x,w)
    d = tf.reduce_sum(c)
    e = tf.reduce_mean(c)
    f = tf.reduce_max(c)
    with tf.Session() as sess:
        outs = sess.run([x,w,c,d,e,f])

print('Outputs for graph A:\n{}')
for n,out in zip(['x','w','c','d','e','f'],outs):        
    print('Node {} :\n{}'.format(n,out))