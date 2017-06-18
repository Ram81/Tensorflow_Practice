import os
import numpy as np 
import tensorflow as tf 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.logging.set_verbosity(tf.logging.ERROR)

FLAGS = None

def placeholder_inputs(batch_size):
	'''
		Generate Placeholder variables to represent input tensors
	'''
	images_placeholder = tf.placeholder(tf.float32, shape = (batch_size, mnist.IMAGE_PIXELS))
	labels_placeholder = tf.Placeholder(tf.int32, shape = (batch_size))
	return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, image_pl, labels_pl):
	'''
		Fill feed_dict for particular step
		Returns feed dictionary mapping from placeholder to values
	'''
	#creating feed_dict for next Batch
	images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)

	feed_dict = {
		images_pl : images_feed,
		labels_pl : labels_feed,
	}
	return feed_dict


def run_training():
	'''
		Training MNIST for number of steps
	'''
	data_sets = input_data.read_datasets(FLAGS.train_dir, FLAGS.fake_data)

	#Tell Tensorflow that model will be built in default Graph
	with tf.Graph().as_default():
		#Generate Placeholders for input
		images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

		#Build a Graph that Computes predictions from inference models
		logits = buildmnist.inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)

		#Add to the Graph ops for calculating loss
		loss = buildmnist.loss(logits, labels_placeholder)

		#Add to the Graph ops that calculate and apply gradients
		train_op = buildmnist.training(loss, FLAGS.learning_rate)

		#Add to Graph ops to compare logits to label during evaluation
		eval_correct = buildmnist.evaluation(logits, labels_placeholder)

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
			if i%100 == 0:

				print('Step %d : loss = %.2f (%.3f sec)'%(step, loss, duration))
				summary_str = sess.run(summary, feed_dict = feed_dict)
				summary_writer.add_summary(summary_str, step)
				summary_writer.flush()

			#Save Checkpoint and evaluate model
			if (step + 1)%1000 == 0 or (step + 1) == FLAGS.max_steps:
				checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
				saver.save(sess, checkpoint_file, global_step = step)

				#Evaluating against training set
				print('Training Data Eval ')
				
