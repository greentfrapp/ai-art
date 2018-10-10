import numpy as np
import classes
import tensorflow as tf
from PIL import Image

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

from utils import VariableState


def mutate(old_vars, scale=1000):
	new_vars = []
	for i, var in enumerate(old_vars):
		new_vars.append(var * np.random.randn() * scale)
	return new_vars

def preprocess(sample_path):
	img = image.load_img(sample_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x

class Generator():

	def __init__(self, sess, name='Generator'):
		self.sess = sess
		self.name = name
		self.batch_size = 32
		with tf.variable_scope(name):
			self.build_model()
			self.vars = {}
			self.vars['old'] = VariableState(self.sess, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '{}/old'.format(self.name)))
			self.vars['new'] = VariableState(self.sess, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '{}/new'.format(self.name)))

	def build_model(self):
		self.input = tf.placeholder(
			dtype=tf.float32,
			shape=[self.batch_size, 1],
			name='z',
		)
		network_names = ['old', 'new']
		self.outputs = {}
		for name in network_names:
			with tf.variable_scope(name):
				dense_1 = tf.layers.dense(
					inputs=self.input,
					units=16*16,
					activation=tf.nn.relu,
					name='dense_1',
				)
				dense_2 = tf.layers.dense(
					inputs=dense_1,
					units=224*224*3,
					activation=tf.sigmoid,
					name='dense_2',
				)
				self.outputs[name] = tf.reshape(dense_2, (-1, 224, 224, 3))

	def mutate_weights(self):
		self.vars['new'].import_variables(mutate(self.vars['old'].export_variables()))

	def update_weights(self, reward, scale=1.):
		# temp = []
		# old_vars = self.vars['old'].export_variables()
		# new_vars = self.vars['new'].export_variables()
		# for i, var in enumerate(new_vars):
		# 	temp.append(old_vars[i] + (var - old_vars[i]) * reward * scale)
		# self.vars['old'].import_variables(temp)
		self.vars['old'].import_variables(self.vars['new'].export_variables())

def main():
	classifier = ResNet50(weights='imagenet')
	sample = preprocess('awkward_moment_seal.png')
	# predicted_label = np.argmax(classifier.predict(sample))
	# print(classes.class_ids[predicted_label])

	sess = tf.Session()
	generator = Generator(sess)

	sess.run(tf.global_variables_initializer())

	z = np.random.randn(32, 1)
	samples = sess.run(generator.outputs['new'], feed_dict={generator.input: z})
	samples = preprocess_input(samples)
	predictions = classifier.predict(samples)[:, 150]
	score = np.mean(predictions)

	for i in np.arange(100000):
		z = np.random.randn(32, 1)
		generator.mutate_weights()
		samples = sess.run(generator.outputs['new'], feed_dict={generator.input: z})
		samples = preprocess_input(samples)
		predictions = classifier.predict(samples)[:, 150]
		new_score = np.mean(predictions)
		if new_score > score:
			generator.update_weights(new_score - score)
			score = new_score
		# if (i % 50) == 0:
		print('Step {} - {}'.format(i, score))

	# predicted_label = np.argmax(classifier.predict(samples))
	# print(predicted_label)
	# print(classes.class_ids[predicted_label])


if __name__ == '__main__':
	main()

