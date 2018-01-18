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

