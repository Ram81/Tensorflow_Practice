import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

inputs_ = tf.placeholder(tf.float32,[None,28,28,1])
#targets_ = tf.placeholder(tf.float32,[None,28,28,1])

def lrelu(x,alpha=0.1):
    return tf.maximum(alpha*x,x)

### Encoder
conv1 = tf.layers.conv2d(inputs_,filters=32,kernel_size=(3,3),strides=(1,1),padding='SAME',use_bias=True,activation=lrelu,name='conv2d_1')
# Now 28x28x32

maxpool1 = tf.layers.max_pooling2d(conv1,pool_size=(2,2),strides=(2,2),name='max_pooling2d_1')
# Now 14x14x32

conv2 = tf.layers.conv2d(maxpool1,filters=32,kernel_size=(3,3),strides=(1,1),padding='SAME',use_bias=True,activation=lrelu,name='conv2d_2')
# Now 14x14x32

encoded = tf.layers.max_pooling2d(conv2,pool_size=(2,2),strides=(2,2),name='max_pooling2d_2')
# Now 7x7x32.
#latent space

### Decoder

conv3 = tf.layers.conv2d(encoded,filters=32,kernel_size=(3,3),strides=(1,1),padding='SAME',use_bias=True,activation=lrelu,name='conv2d_3')
#Now 7x7x32        
upsample1 = tf.layers.conv2d_transpose(conv3,filters=32,kernel_size=3,padding='same',strides=2,name='conv2d_transpose_1')
# Now 14x14x32
upsample2 = tf.layers.conv2d_transpose(upsample1,filters=32,kernel_size=3,padding='same',strides=2,name='conv2d_transpose_2')
# Now 28x28x32
logits = tf.layers.conv2d(upsample2,filters=1,kernel_size=(3,3),strides=(1,1),padding='SAME',use_bias=True,name='conv2d_4')
#Now 28x28x1
# Pass logits through sigmoid to get reconstructed image
decoded = tf.sigmoid(logits,name='activation_1')
'''
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=targets_)

learning_rate=tf.placeholder(tf.float32)
cost = tf.reduce_mean(loss)  #cost
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost) #optimizer
'''
sess = tf.Session()
tf.train.write_graph(sess.graph.as_graph_def(add_shapes=True), '/tmp', '/home/rockstar/denoise_autoenc.pbtxt', True)

#print(tf.summary())

#tf.reset_default_graph()