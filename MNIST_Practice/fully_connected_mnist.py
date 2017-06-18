import os, sys, argparse, time
import numpy as np 
import tensorflow as tf 

from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.examples.tutorials.mnist import mnist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.logging.set_verbosity(tf.logging.ERROR)

FLAGS = None

def placeholder_inputs(batch_size):
	'''
		Generate Placeholder variables to represent input tensors
	'''
	images_placeholder = tf.placeholder(tf.float32, shape = (batch_size, mnist.IMAGE_PIXELS))
	labels_placeholder = tf.placeholder(tf.int32, shape = (batch_size))
	return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, image_pl, labels_pl):
	'''
		Fill feed_dict for particular step
		Returns feed dictionary mapping from placeholder to values
	'''
	#creating feed_dict for next Batch
	images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)

	feed_dict = {
		image_pl : images_feed,
		labels_pl : labels_feed,
	}
	return feed_dict

def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
	'''
		Run one evaluation against full epoch of data
	'''
	#Number of Correct Predictions
	true_count = 0
	steps_per_epoch = data_set.num_examples // FLAGS.batch_size
	num_examples = steps_per_epoch * FLAGS.batch_size
	
	for step in xrange(steps_per_epoch):
		feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
		true_count += sess.run(eval_correct, feed_dict = feed_dict)

	precision = float(true_count) / num_examples
	print(' Number of Examples : %d, Number of Correct : %d, Precision @ 1 : %0.04f'%(num_examples, true_count, precision))



def run_training():
	'''
		Training MNIST for number of steps
	'''
	data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

	#Tell Tensorflow that model will be built in default Graph
	with tf.Graph().as_default():
		#Generate Placeholders for input
		images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

		#Build a Graph that Computes predictions from inference models
		logits = mnist.inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)

		#Add to the Graph ops for calculating loss
		loss = mnist.loss(logits, labels_placeholder)

		#Add to the Graph ops that calculate and apply gradients
		train_op = mnist.training(loss, FLAGS.learning_rate)

		#Add to Graph ops to compare logits to label during evaluation
		eval_correct = mnist.evaluation(logits, labels_placeholder)

		#Build summary tensor based on TF collection of summaries.
		summary = tf.summary.merge_all()

		init = tf.global_variables_initializer()

		#Saving checkpoints of training
		saver = tf.train.Saver()

		sess = tf.Session()

		#Instantiate SummaryWriter to write summaries
		summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

		sess.run(init)

		#Start training Loop
		for step in xrange(FLAGS.max_steps):
			start_time = time.time()

			feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder)

			#Run one step of model
			_, loss_value = sess.run([train_op, loss], feed_dict = feed_dict)
			
			duration = time.time() - start_time

			#Write summaries and print overview
			if step%100 == 0:

				print('Step %d : loss = %.2f (%.3f sec)'%(step, loss_value, duration))
				summary_str = sess.run(summary, feed_dict = feed_dict)
				summary_writer.add_summary(summary_str, step)
				summary_writer.flush()

			#Save Checkpoint and evaluate model
			if (step + 1)%1000 == 0 or (step + 1) == FLAGS.max_steps:
				checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
				saver.save(sess, checkpoint_file, global_step = step)

				#Evaluating against training set
				print('Training Data Eval ')
				do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.train)

				print('Validation Data Eval ')
				do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.validation)

				print('Testing Data Eval ')
				do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.test)

def main(_):
	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
	tf.gfile.MakeDirs(FLAGS.log_dir)
	run_training()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--learning_rate', type = float, default = 0.01, help = 'Initial Learning Rate')
	parser.add_argument('--max_steps', type = int, default = 10000, help = 'Number of Steps to run trainer')
	parser.add_argument('--hidden1', type = int, default = 128, help = 'Number of Inputs to hidden layer 1')
	parser.add_argument('--hidden2', type = int, default = 32, help = 'Number of Inputs to hidden layer 2')
	parser.add_argument('--batch_size', type = int, default = 100, help = 'Number of Inputs to be considered in batch for training')
	parser.add_argument('--input_data_dir', type = str, default = 'MNIST_data/', help = 'Directory for Input Data')
	parser.add_argument('--log_dir', type = str, default = 'logs/', help = 'Directory for Input Data')
	parser.add_argument('--fake_data', default = False, help = 'If True uses Fake Data', action = 'store_true')

	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main = main, argv = [sys.argv[0]] + unparsed)
