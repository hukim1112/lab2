
import tensorflow as tf
ds = tf.contrib.distributions
from tensorflow.python.ops.losses import losses
#test code

import tensorflow as tf

a_ = tf.constant( [ [3.5, 2, 3, 4], [1.5, 3, 2, 4], [1, 6, 7, 4], [1, 6, 7, 4]], tf.float32 )


a = tf.constant( [ [1.3, 4, 7, 2], [3, 2, 8, 6], [2, 4, 7, 4], [8, 3, 2, 1]], tf.float32 )
variables = tf.Variable([[1, 0, 0, 0,],[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=tf.float32)

b = tf.matmul(a, variables)
# ones = tf.ones_like(a[:, 0], tf.float32)
# bias = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(a[:, 0], ones), 2.0))

mean = tf.reduce_mean(b, axis = 0)
k = tf.subtract(a, mean)
p = tf.pow(k, 2)
each_variance = tf.reduce_mean(p, axis=0)
relative = each_variance[0] / tf.reduce_mean(each_variance)

mean_loss = 10*tf.pow(mean[0] , 2)




b_ = tf.matmul(a_, variables)
# ones = tf.ones_like(a[:, 0], tf.float32)
# bias = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(a[:, 0], ones), 2.0))

mean_ = tf.reduce_mean(b_, axis = 0)
k_ = tf.subtract(a_, mean_)
p_ = tf.pow(k_, 2)
each_variance_ = tf.reduce_mean(p_, axis=0)
relative_ = each_variance_[0] / tf.reduce_mean(each_variance_)

mean_loss_ = 10*tf.pow(mean_[0] , 2)








solver = tf.train.AdamOptimizer(0.01, beta1=0.5).minimize(-relative -relative_ + mean_loss + mean_loss_) 
optimizer = tf.train.AdamOptimizer() 
gradients = optimizer.compute_gradients(relative, variables)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print('f(a) : \n', sess.run(b))
	print('f(a_) : \n', sess.run(b_))
	print('mean : ',sess.run(mean))
	print('mean : ',sess.run(mean_))
	print('relative_variance : \n', sess.run(relative), sess.run(relative_))

	for i in range(1000):
		sess.run(solver)
	print('f(a) : \n', sess.run(b))
	print('f(a_) : \n', sess.run(b_))
	print('mean : ',sess.run(mean))
	print('mean : ',sess.run(mean_))
	print('relative_variance : \n', sess.run(relative), sess.run(relative_))