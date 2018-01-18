import os
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.logging.set_verbosity(tf.logging.ERROR)

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Start tf Session
session = tf.InteractiveSession()

X = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# define Weights and Biases
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

session.run(tf.global_variables_initializer())

# Predict Class and loss function
y = tf.matmul(X, W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

# Training the model
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
	batch_ = mnist.train.next_batch(100)
	train_step.run(feed_dict={X: batch_[0], y_: batch_[1]})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={X: mnist.test.images, y_: mnist.test.labels}))

