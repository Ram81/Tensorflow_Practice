import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)

#declaring a list of features
features = [tf.contrib.layers.real_valued_column("x", dimension = 1)]

#estimator is a frontend to invoke fitting(training) and inferencing(evaluation) a model
estimator = tf.contrib.learn.LinearRegressor(feature_columns = features)

#setting up datasets
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])

input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size = 4, num_epochs=1000)

#we can invoke 1000 training steps by using fit method
estimator.fit(input_fn = input_fn, steps = 1000)

#evaluate model
print(estimator.evaluate(input_fn = input_fn));
