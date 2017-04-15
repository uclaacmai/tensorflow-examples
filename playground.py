import tensorflow as tf
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
s = tf.add(a,b)
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	print(sess.run(s, feed_dict = {a: 1, b: 3}))
