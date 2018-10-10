import pygame, random
import numpy as np


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
	interval = np.floor(radius / 2.5)
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
	screen = pygame.display.set_mode((800,800))
	pygame.draw.circle(screen, (255, 255, 255), (400, 400), 600)
	draw(
		screen=screen, 
		color=(255, 128, 0),
		radius=20,
		start=[400, 300],
		end=[300, 200]
	)
	pygame.image.save(screen, 'test.jpg')
	quit()
	

if __name__ == '__main__':
	main()