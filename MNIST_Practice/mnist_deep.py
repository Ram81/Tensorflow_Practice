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
