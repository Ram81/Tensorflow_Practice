import os
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.logging.set_verbosity(tf.logging.ERROR)

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

# Define First conv layer
W_conv1 = weight_variable([5, 5, 1, 32])
B_conv1 = bias_variable([32])

# reshape original input vectors
X_image = tf.reshape(X, [-1, 28, 28, 1])

# First convolution operation
H_conv1 = tf.nn.relu(conv2d(X_image, W_conv1) + B_conv1)
H_pool1 = max_pool_2x2(H_conv1)
