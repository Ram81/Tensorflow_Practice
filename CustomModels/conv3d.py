import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

inputs_ = tf.placeholder(tf.float32,[None,16])
#targets_ = tf.placeholder(tf.float32,[None,28,28,1])

def lrelu(x,alpha=0.1):
    return tf.maximum(alpha*x,x)

'''
fc1 = tf.contrib.layers.fully_connected(inputs_, 32)
bn1 = tf.contrib.layers.batch_norm(fc1)

fc2 = tf.contrib.layers.fully_connected(bn1, 32)
'''
fc1 = tf.layers.dense(inputs_, 32, kernel_initializer=tf.zeros_initializer(), name='abc')

fc2 = tf.layers.dense(inputs_, 32, kernel_initializer=tf.truncated_normal_initializer(), name='def')


#bn2 = tf.contrib.layers.batch_norm(fc2)
# conv => 16*16*16
'''
conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3], padding='same', activation=tf.nn.relu, name='conv2')

# pool => 8*8*8
pool3 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=2)
        
# conv => 8*8*8.

conv4 = tf.layers.conv3d(inputs=pool3, filters=64, kernel_size=[3,3,3], padding='same', activation=tf.nn.relu, name='conv3')
# conv => 8*8*8
conv5 = tf.layers.conv3d(inputs=conv4, filters=128, kernel_size=[3,3,3], padding='same', activation=tf.nn.relu, name='conv4')
# pool => 4*4*4
pool6 = tf.layers.max_pooling3d(inputs=conv5, pool_size=[2, 2, 2], strides=2)
        
cnn3d_bn = tf.layers.batch_normalization(inputs=pool6, training=True)

flattening = tf.layers.flatten(cnn3d_bn, name="hhh")

dense = tf.layers.dense(inputs=flattening, units=1024, activation=tf.nn.relu)
# (1-0.7) is the probability that the node will be kept


dropout = tf.nn.dropout(inputs=dense, rate=0.7, training=True, name="droped")
y_conv = tf.layers.dense(inputs=dropout, units=10)

loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=targets_)

learning_rate=tf.placeholder(tf.float32)
cost = tf.reduce_mean(loss)  #cost
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost) #optimizer
'''
sess = tf.Session()
tf.train.write_graph(sess.graph.as_graph_def(add_shapes=True), '/tmp', '/home/rockstar/fc_i.pbtxt', True)

#print(tf.summary())

#tf.reset_default_graph()