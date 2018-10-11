import pygame, random
import numpy as np
from PIL import Image
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import classes


class PaintEnv():

	def __init__(self, name, classifier):
		self.name = name
		self.classifier = classifier
		self.steps = 0
		self.max_steps = 20
		self.end = False
		self.filename = '{}.jpg'.format(self.name)
		self.screen = pygame.display.set_mode((100, 100))
		# pygame.draw.circle(self.screen, (255, 255, 255), (50, 50), 200)

	def step(self, action):
		if not self.end:
			draw(
				screen=self.screen, 
				color=action['color'],
				radius=action['radius'],
				start=action['start'],
				end=action['end'],
			)
			pygame.image.save(self.screen, self.filename)
			sample = preprocess(self.filename)
			predictions = self.classifier.predict(sample)
			# predicted_label = np.argmax(predictions)
			# confidence = np.max(predictions)
			# print(classes.class_ids[predicted_label])
			# print(confidence)
			reward = predictions[0, 150]
			canvas = np.array(Image.open(self.filename))
			self.steps += 1
			if self.steps > self.max_steps:
				self.end = True
			return np.expand_dims(canvas, axis=0), np.log(reward)/100, self.end
		else:
			return None

	def reset(self):
		self.steps = 0
		self.end = False
		# self.screen = pygame.display.set_mode((100, 100))
		pygame.draw.circle(self.screen, (255, 255, 255), (50, 50), 200)
		pygame.image.save(self.screen, self.filename)
		sample = preprocess(self.filename)
		predictions = self.classifier.predict(sample)
		# predicted_label = np.argmax(predictions)
		# confidence = np.max(predictions)
		reward = predictions[0, 150]
		canvas = np.array(Image.open(self.filename))
		return np.expand_dims(canvas, axis=0), np.log(reward)/100, self.end

def preprocess(sample_path):
	img = image.load_img(sample_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x

def roundline(srf, color, start, end, radius=1):
	dx = end[0]-start[0]
	dy = end[1]-start[1]
	distance = max(abs(dx), abs(dy))
	for i in range(distance):
		x = int( start[0]+float(i)/distance*dx)
		y = int( start[1]+float(i)/distance*dy)
		pygame.draw.circle(srf, color, (x, y), radius)

def draw(screen, color, radius, start, end):
	dx = end[0] - start[0]
	dy = end[1] - start[1]
	interval = np.floor(radius / 3)
	last_pos = pos = start
	n_steps = int(max(abs(dx), abs(dy)) / interval)
	for i in np.arange(n_steps):
		pos[0] += int(dx / n_steps)
		pos[1] += int(dy / n_steps)
		pygame.draw.circle(screen, color, tuple(pos), radius)
		roundline(screen, color, tuple(pos), tuple(last_pos),  radius)
		last_pos = pos

def manual():
	screen = pygame.display.set_mode((800,600))

	draw_on = False
	last_pos = (0, 0)
	color = (255, 128, 0)
	radius = 10
	try:
		while True:
			e = pygame.event.wait()
			if e.type == pygame.QUIT:
				raise StopIteration
			if e.type == pygame.MOUSEBUTTONDOWN:
				color = (random.randrange(256), random.randrange(256), random.randrange(256))
				pygame.draw.circle(screen, color, e.pos, radius)
				draw_on = True
			if e.type == pygame.MOUSEBUTTONUP:
				draw_on = False
				print(e.pos)
			if e.type == pygame.MOUSEMOTION:
				if draw_on:
					print(last_pos)
					pygame.draw.circle(screen, color, e.pos, radius)
					roundline(screen, color, e.pos, last_pos,  radius)
				last_pos = e.pos
			pygame.display.flip()

	except StopIteration:
		pass

	pygame.quit()


def main():
	# env = PaintEnv('test')
	# env.step({
	# 	'color': (255, 128, 50),
	# 	'radius': 20,
	# 	'start': [400, 300],
	# 	'end': [300, 200],
	# 	})
	# quit()
	screen = pygame.display.set_mode((800,800))
	pygame.draw.circle(screen, (255, 255, 255), (400, 400), 600)
	draw(
		screen=screen, 
		color=(255, 0, 0),
		radius=20,
		start=[400, 300],
		end=[300, 200]
	)
	pygame.image.save(screen, 'test.jpg')
	quit()
	

if __name__ == '__main__':
	main()