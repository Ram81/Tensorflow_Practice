import os, math
import numpy as np 
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.logging.set_verbosity(tf.logging.ERROR)

# number of output classes
NUM_CLASSES = 10

# MNIST images 28 x 28 pixels
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def inference(images, hidden1_units, hidden2_units):
	'''
		Build MNIST model upto where it may be used for inference
		Input: images placeholder, size of first hidden unit, size of second hidden unit
		Output : softmax_linear Output tensor with computed logits
	'''

	#Hidden Unit 1 with relu activation
	#tf.name.scope returns a context manager for use when defining python op
	with tf.name_scope('hidden1'):

		#truncated_normal outputs random values from a truncated normal distribution
		weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units], stddev = 1.0/math.sqrt(float(IMAGE_PIXELS))), name = 'weights')
		
		biases = tf.Variable(tf.zeros([hidden1_units]), name = 'biases')

		hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

	#Hidden Unit 2 with relu activation
	with tf.name_scope('hidden2'):

		weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units], stddev = 1.0/math.sqrt(float(hidden1_units))), name = 'weights')

		biases = tf.Variable(tf.zeros([hidden2_units]), name = 'biases')

		hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

	#Linear layer with softmax
	with tf.name_scope('softmax_linear'):

		weights = tf.Variable(tf.truncated_normal([hidden2_units, NUM_CLASSES], stddev = 1.0/math.sqrt(float(hidden2_units))), name = 'weights')

		biases = tf.Variable(tf.zeros([NUM_CLASSES]), name = 'biases')

		logits = tf.matmul(hidden2, weights) + biases
	return logits


def loss(logits, labels):
	'''
		Further building of graph by adding required loss ops
		calculates loss from parameters logits and labels
		Returns loss tensor of type float
	'''

	labels = tf.to_int64(labels)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = logits, name = 'xentropy')
	return tf.reduce_mean(cross_entropy, name = 'xentropy_mean')

def training(loss, learning_rate):
	'''
		Sets up training ops for minimizing loss via Gradient Descent
		Returns op for training
	'''
	
	#Adding a scalar summary for snapshot loss
	tf.summary.scalar('loss', loss)

	#creating a GradientDescent Optimizer with given learning rate
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)

	global_step = tf.Variable(0, name = 'global_step', trainable = False)

	#Applying GradientDescent using optimizer
	train_op = optimizer.minimize(loss, global_step = global_step)
	return train_op

def evaluation(logits, labels):
	'''
		Evaluate quality of logits for predicting labels
		Returns tensor with number of outputs correctly predicted
	'''
	correct = tf.nn.in_top_k(logits, labels, 1)
	return tf.reduce_sum(tf.cast(correct, tf.int32))

