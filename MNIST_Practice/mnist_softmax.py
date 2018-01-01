import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(X, W) + b

# Not using because numerically unstable
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y), reduction_indices=[1]))

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=Y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

session = tf.InteractiveSession()
tf.global_variables_initializer().run()

print("Training....")
# Implement Stochastic Gradient Descent
for epochs in range(1000):
	batch_X, batch_Y = mnist.train.next_batch(100)
	session.run(train_step, feed_dict={X: batch_X, Y: batch_Y})

print("Training Complete")

# Evaluate Model
correct_predictions = tf.equal(tf.argmax(Y, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

print(session.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))