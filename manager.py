
import tensorflow as tf
import cv2



class Manager():
	def __init__(self, args):
		self.learning_rate = args.learning_rate
		self.momentum = args.momentum

		self.continue_training = args.continue_training
		self.checkpoints_path = args.checkpoints_path
		self.graph_path = args.graph_path
		self.epochs = args.epochs
		self._global_step = tf.get_variable(
             'global_step', [],
             initializer=tf.constant_initializer(0),
             trainable=False)
		self.batch_size = args.batch_size
		self.decay_steps=args.decay_steps
		self.decay_rate=args.decay_rate

	#function to train the network
	def train(self, sess, model):
		learning_rate = tf.train.exponential_decay(
			learning_rate=self.learning_rate,
		    global_step=self._global_step,
		   	decay_steps=self.decay_steps,
		    decay_rate=self.decay_rate)

		learning_rate = tf.maximum(learning_rate, 1e-6)
		#optimizer
		optimizer = tf.train.AdamOptimizer(learning_rate, name="AdamOptimizer").minimize(model.loss, var_list=model.trainable_vars,global_step=self._global_step)

		epoch = 0
		step = 0
		overall_step = 0

		#saver
		saver = tf.train.Saver()
		if self.continue_training:
			last_ckpt = tf.train.latest_checkpoint(self.checkpoints_path)
			saver.restore(sess, last_ckpt)
			ckpt_name = str(last_ckpt)
			print "Loaded model file from " + ckpt_name
			epoch = int(last_ckpt[len(ckpt_name)-1])
		else:
			tf.global_variables_initializer().run()


		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)


		if hasattr(model, 'rec_img_summary'):
			all_summary = tf.summary.merge([model.img_summary,
											model.rec_img_summary,
											model.loss_summary,
											model.acc_summary])
		else:
			all_summary = tf.summary.merge([model.img_summary,
											model.loss_summary,
											model.acc_summary])

		writer = tf.summary.FileWriter(self.graph_path, sess.graph)

		while epoch < self.epochs:
			summary, loss, acc, _ = sess.run([all_summary,
											  model.loss,
											  model.accuracy,
											  optimizer])
			writer.add_summary(summary, overall_step)

			print "Epoch [%d] step [%d] Training Loss: [%.4f] Accuracy: [%.4f]" % (epoch, step, loss, acc)

			step += 1
			overall_step += 1

			if step*self.batch_size >= model.data_count:
				saver.save(sess, self.checkpoints_path + "model", global_step=epoch)
				print "Model saved at epoch %s" % str(epoch)
				epoch += 1
				step = 0

		coord.request_stop()
		coord.join(threads)
		sess.close()
		print("Done.")

	#function to start testing
	def test(self, sess, model):

		saver = tf.train.Saver()
		last_ckpt = tf.train.latest_checkpoint(self.checkpoints_path)
		saver.restore(sess, last_ckpt)
		print "Loaded model file from " + str(last_ckpt)

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		ave_acc = 0
		step = 0
		while 1:
			loss = sess.run(model.loss)
			accuracy = sess.run(model.accuracy)

			print "Step [%d] Test Loss: [%.4f] Accuracy [%.4f]" % (step, loss, accuracy)

			ave_acc += accuracy
			step += 1
			if step*self.batch_size > model.data_count:
				break

		print "Ave. Accuracy: [%.4f]"%(ave_acc/step)
		coord.request_stop()
		coord.join(threads)
		sess.close()
		print('Done')
