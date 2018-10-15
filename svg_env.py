import numpy as np
# import drawSvg
from PIL import Image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import classes
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM


def preprocess(sample_path):
	img = image.load_img(sample_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x

class SvgEnv():

	def __init__(self, name, classifier=None):
		self.name = name
		self.classifier = classifier or ResNet50(weights='imagenet')
		self.steps = 0
		self.max_steps = 20
		self.episode_end = False
		self.filename = '{}.svg'.format(self.name)
		self.start = '<svg xmlns="http://www.w3.org/2000/svg" x="0px" y="0px" width="200px" height="200px" viewBox="0 0 200 200">'
		self.end = '</svg>'
		self.contents = ''
		# self.canvas = drawSvg.Drawing(200, 200)
		# self.canvas.append(drawSvg.Rectangle(0,0,200,200, fill='#ffffff'))

	def draw(self, action):
		self.contents += '<path fill="{}" d="'.format(action['color'])
		self.contents += 'M {} {}'.format(action['vertices'][0][0], action['vertices'][0][1])
		for i, val in enumerate(action['vertices'][1:]):
			self.contents += ' L {} {}'.format(val[0], val[1])
		self.contents += ' Z"></path>'
		with open(self.filename, 'w') as file:
			file.write(self.start)
			file.write(self.contents)
			file.write(self.end)
		drawing = svg2rlg(self.filename)
		renderPM.drawToFile(drawing, self.filename.split('.')[0] + '.png')

		# p = drawSvg.Path(fill=action['color'])
		# p.M(action['vertices'][0][0], action['vertices'][0][1])
		# for i, val in enumerate(action['vertices'][1:]):
		# 	p.l(val[0] - action['vertices'][i][0], val[1] - action['vertices'][i][1])
		# p.Z()
		# self.canvas.append(p)
		# self.canvas.savePng(self.filename)
		return np.array(Image.open(self.filename.split('.')[0] + '.png').convert('RGB'), dtype=np.float32) / 255

	def step(self, action):
		if not self.episode_end:
			img = self.draw(action)
			sample = preprocess(self.filename.split('.')[0] + '.png')
			predictions = self.classifier.predict(sample)
			reward = predictions[0, 376]
			self.steps += 1
			if self.steps > self.max_steps:
				self.episode_end = True
				# print(np.argmax(predictions))
				# print(predictions[0, np.argmax(predictions)])
				# print(predictions[0, 309])
				# quit()
			# if self.steps == 5:
			# 	quit()
			return np.expand_dims(img, axis=0), np.log(reward)/100, self.episode_end
		else:
			return None

	def reset(self):
		self.steps = 0
		self.episode_end = False
		# self.canvas = drawSvg.Drawing(200, 200)
		# self.canvas.append(drawSvg.Rectangle(0,0,200,200, fill='#ffffff'))
		# self.canvas.savePng(self.filename)
		self.contents = ''
		with open(self.filename, 'w') as file:
			file.write(self.start)
			file.write(self.contents)
			file.write(self.end)
		drawing = svg2rlg(self.filename)
		renderPM.drawToFile(drawing, self.filename.split('.')[0] + '.png')
		img = np.array(Image.open(self.filename.split('.')[0] + '.png').convert('RGB'), dtype=np.float32) / 255
		sample = preprocess(self.filename.split('.')[0] + '.png')
		predictions = self.classifier.predict(sample)
		reward = predictions[0, 376]
		return np.expand_dims(img, axis=0), np.exp(reward) - 1, self.episode_end

def main():
	env = SvgEnv('env')
	action = {
		'color': '#73748A',
		'vertices': [
			[100, 100],
			[20, 20],
			[0, 50],
		],
	}
	env.step(action)

if __name__ == '__main__':
	main()
