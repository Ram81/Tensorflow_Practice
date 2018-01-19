import os
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.logging.set_verbosity(tf.logging.ERROR)

# Reading input
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
	'''
		Generates a Weight Variable of given shape 
	'''
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	'''
		Generates a Bias Variable of given shape
	'''
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(X, W):
	'''
		Returns a conv2d layer of stride = 1
	'''
	return tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(X):
	'''
		Returns a downsampled feature map by 2X
	'''
	return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# reshape original input vectors
X_image = tf.reshape(X, [-1, 28, 28, 1])

# First Convolution Operation
W_conv1 = weight_variable([5, 5, 1, 32])
B_conv1 = bias_variable([32])

H_conv1 = tf.nn.relu(conv2d(X_image, W_conv1) + B_conv1)
H_pool1 = max_pool_2x2(H_conv1)

# Second Convolution Operation
W_conv2 = weight_variable([5, 5, 32, 64])
B_conv2 = bias_variable([64])

H_conv2 = tf.nn.relu(conv2d(H_pool1, W_conv2) + B_conv2)
H_pool2 = max_pool_2x2(H_conv2)

# Densely Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
B_fc1 = bias_variable([1024])

H_pool2_flat = tf.reshape(H_pool2, [-1, 7 * 7 * 64])
H_fc1 = tf.nn.relu(tf.matmul(H_pool2_flat, W_fc1) + B_fc1)

# Dropout Layer
keep_prob = tf.placeholder(tf.float32)
H_fc1_dropout = tf.nn.dropout(H_fc1, keep_prob)

# Readout Layer
W_fc2 = weight_variable([1024, 10])
B_fc2 = bias_variable([10])

Y_conv = tf.matmul(H_fc1_dropout, W_fc2) + B_fc2

# Train & Evaluate Model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_conv, labels=Y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	for i in range(20000):
		batch = mnist.train.next_batch(50)
		if i%100 == 0:
			train_accuracy = accuracy.eval(feed_dict={X: batch[0], Y: batch[1], keep_prob: 1.0})
			print('Training accuracy in step %d is %g'%(i, train_accuracy))
		train_step.run(feed_dict={X: batch[0], Y: batch[1], keep_prob: 0.5})

	print('Test accuracy is %g'%(accuracy.eval(feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})))
