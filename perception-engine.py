import numpy as np
import classes
import tensorflow as tf

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions


def preprocess(sample_path):
	img = image.load_img(sample_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x

class Generator():

	def __init__(self, name='Generator'):
		self.name = name
		self.batch_size = 1
		with tf.variable_scope(name):
			self.build_model()

	def build_model(self):
		self.input = tf.placeholder(
			dtype=tf.float32,
			shape=[self.batch_size, 1],
			name='z',
		)
		dense_1 = tf.layers.dense(
			inputs=self.input,
			units=224*224*3,
			activation=tf.nn.relu,
			name='dense_1',
		)
		# dense_2 = tf.layers.dense(
		# 	inputs=dense_1,
		# 	units=224*224*3,
		# 	activation=tf.nn.relu,
		# 	name='dense_2',
		# )
		self.output = tf.reshape(dense_1, (224, 224, 3))

def main():
	classifier = ResNet50(weights='imagenet')
	sample = preprocess('awkward_moment_seal.png')
	# predicted_label = np.argmax(classifier.predict(sample))
	# print(classes.class_ids[predicted_label])

	generator = Generator()

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	z = np.random.randn(1, 1)
	samples = sess.run(generator.output, feed_dict={generator.input: z})
	samples = np.expand_dims(samples, axis=0)
	samples = preprocess_input(samples)

	predicted_label = np.argmax(classifier.predict(samples))
	print(classes.class_ids[predicted_label])


if __name__ == '__main__':
	main()

