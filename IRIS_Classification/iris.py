import os, urllib
import numpy as np 
import tensorflow as tf 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.logging.set_verbosity(tf.logging.ERROR)

#Data set
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():
	'''
		Downloading the training and testing data if not available locally
	'''
	if not os.path.exists(IRIS_TRAINING):
		raw = urllib.urlopen(IRIS_TRAINING_URL).read()
		with open(IRIS_TRAINING, "w") as f:
			f.write(raw)

	if not os.path.exists(IRIS_TEST):
		raw = urllib.urlopen(IRIS_TEST_URL).read()
		with open(IRIS_TEST, "w") as f:
			f.write(raw)

	training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename = IRIS_TRAINING, target_dtype = np.int, features_dtype = np.float32)
	test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename = IRIS_TEST, target_dtype = np.int, features_dtype = np.float32)

	#Specify that all features have real valued data
	feature_columns = [tf.contrib.layers.real_valued_column("", dimension = 4)]	

	#Build 3 layer DNN
	classifier = tf.contrib.learn.DNNClassifier(feature_columns = feature_columns, hidden_units = [10, 20, 10], n_classes = 3, model_dir = '/tmp/iris_model')


	def get_train_inputs():
		'''
			Define Training Inputs
		'''
		X = tf.constant(training_set.data)
		y = tf.constant(training_set.target)

		return X, y

	classifier.fit(input_fn = get_train_inputs, steps = 2000)

	def get_test_inputs():
		'''
			Define Test inputs
		'''
		X = tf.constant(test_set.data)
		y = tf.constant(test_set.target)

		return X, y

	accuracy_score = classifier.evaluate(input_fn = get_test_inputs, steps = 1)["accuracy"]

	print('\n Test Accuracy Score : {0:f} \n'.format(accuracy_score))

	def new_samples():
		return np.array([[6.4, 3.2, 4.5, 1.5],[5.8, 3.1, 5.0, 1.7]], dtype = np.float32)

	predictions = list(classifier.predict(input_fn = new_samples))

	print('\n New Samples, Class Predictions: {}\n'.format(predictions))


if __name__ == '__main__':
	main()
