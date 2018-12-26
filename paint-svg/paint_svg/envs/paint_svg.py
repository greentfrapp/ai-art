"""
A custom Gym environment for drawing with SVG markup
Drawings will be classified by ResNet50 from the Keras library
"""

# gym imports
import gym
from gym import error, spaces, utils
from gym.utils import seeding
# graphics, svg and keras imports
from PIL import Image
from keras.applications.resnet50 import \
	preprocess_input, decode_predictions, ResNet50
from keras.preprocessing import image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from paint_svg.utils.imagenet_classes import imagenet_classes

import numpy as np

def preprocess(sample_path):
	img = image.load_img(sample_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x


class PaintSvg(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self, classifier=None, max_steps=5, colors=5, canvas_size=(200, 200), chosen_class=None):
		# Each action comprises a chosen color and three vertices
		# To make things easier, implement as 7 continuous variables
		# then we will round up the values
		# An alternative is to implement large Discrete space
		self.action_space = spaces.Box(
			low=np.array([0] * 7),
			high=np.array([colors] + list(canvas_size) * 3),
			dtype=np.float32
		)
		# An observation comprises the entire image
		self.observation_space = spaces.Box(
			low=0,
			high=255,
			shape=(np.product(canvas_size + (3,)),),
			dtype=np.float32,
		)
		
		self.classifier = classifier or ResNet50(weights='imagenet')
		self.steps = 0
		self.max_steps = max_steps
		self.colors = colors
		self.colors_dict = {
			0: '#000000',
			1: '#FF0000',
			2: '#00FF00',
			3: '#0000FF',
			4: '#FFFF00',
		}
		self.canvas_size = canvas_size
		self.filename = 'image.svg'
		self.svg_header = '<svg xmlns="http://www.w3.org/2000/svg" \
			x="0px" y="0px" width="{}px" height="{}px" \
			viewBox="0 0 {} {}">'.format(
				canvas_size[0], canvas_size[1],
				canvas_size[0], canvas_size[1]
			)
		self.svg_content = ''
		self.svg_footer = '</svg>'
		# self.chosen_class = chosen_class or np.random.choice(range(1000))
		self.chosen_class = chosen_class or 545
		self.threshold = 0.95

		self.reset()

	def draw(self):
		with open(self.filename, 'w') as file:
			file.write(self.svg_header)
			file.write(self.svg_content)
			file.write(self.svg_footer)
		drawing = svg2rlg(self.filename)
		renderPM.drawToFile(drawing, self.filename.split('.')[0] + '.png')
		return np.array(Image.open(self.filename.split('.')[0] + '.png').convert('RGB'), dtype=np.float32) / 255

	def step(self, action):
		action[0] = min(max(int(action[0]), 0), self.colors - 1)
		for i, a in enumerate(action[1:]):
			action[i + 1] = min(max(int(a), 0), 255)
		assert self.action_space.contains(action)
		assert self.steps < self.max_steps
		color = self.colors_dict[action[0]]
		vertices = [action[i:i + 2] for i in np.arange(1, len(action), 2)]
		done = False
		self.svg_content += '<path stroke="black" stroke-width="5" \
			fill="{}" d="'.format(color)
		self.svg_content += 'M {} {}'.format(*vertices[0])
		for vertex in vertices[1:]:
			self.svg_content += ' L {} {}'.format(*vertex)
		self.svg_content += ' Z"></path>'
		flattened_img = self.draw().flatten()
		predictions = self.classifier.predict(preprocess(self.filename.split('.')[0] + '.png'))
		reward = np.exp(predictions[0, self.chosen_class]) - 1
		self.steps += 1
		if predictions[0, self.chosen_class] > self.threshold:
			self.steps = self.max_steps
		if self.steps >= self.max_steps:
			done = True
		return flattened_img, reward, done, {'chosen_class': imagenet_classes[self.chosen_class]}

	def reset(self):
		self.steps = 0
		self.contents = ''
		flattened_img = self.draw().flatten()
		return flattened_img

	def render(self, mode='human', close=False):
		pass