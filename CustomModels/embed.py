import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

slim = tf.contrib.slim

inputs = tf.placeholder(tf.float32,[None,256, 256, 3])

net = slim.separable_conv2d(inputs, 128, [3,3], depth_multiplier=1)
#net = slim.batch_norm(net)
net = slim.separable_conv2d(net, 128, [3,3], depth_multiplier=1)
#net = slim.batch_norm(net)
net = slim.max_pool2d(net, [3,3], stride=2, padding='same')

sess = tf.Session()
tf.train.write_graph(sess.graph.as_graph_def(add_shapes=True), '/tmp', '/home/rockstar/sep3d_1.pbtxt', True)
