
import numpy as np
from image_processing import draw, transform, colorspace
import os
import cv2
#random function
#use np.random.rand(1) to get samples from uniform distribution over [0, 1]


def draw_canvas(image_size=(256, 256, 3), color=None):
	canvas =  np.zeros(image_size, dtype = np.uint8)
	if color != None:
		canvas[:, :, 0] = color[0]
		canvas[:, :, 1] = color[1]
		canvas[:, :, 2] = color[2]
	return canvas

def draw_rectangles(dataset_dir, image_size=(128, 128, 3), color = (255, 255, 255), background = (0, 0, 0), num=10000):
	path = os.path.join(dataset_dir, 'rectangle')
	os.makedirs(path, exist_ok = True)
	for i in range(num):
		#Make canvas to draw
		canvas = draw_canvas(image_size, color = background)

		#sample some values for random transformation
		#And define center and angle
		random_values = np.random.normal(loc=0, scale = 0.5, size=[2])
		#random_values = (np.random.rand(2) - 0.5)*2
		center = (int(  image_size[0]/2 + random_values[0]*(image_size[0]/5)  ), int(image_size[1]/2) )
		angle = random_values[1] * 60
		width = int(image_size[0]/8)
		height = width * 3
		pt1 = center[0]-int(width/2), center[1]-int(height/2)
		pt2 = center[0]+int(width/2), center[1]+int(height/2)

		#Draw the rectangle into canvas
		draw.rectangle(canvas, pt1, pt2, color=color, thickness = -1)
		#transform the image
		img = transform.rotate(canvas, center, angle, 1, border_color=background)
		cv2.imwrite(os.path.join(path, 'rect_'+str(i)+'.png'), colorspace.rgb2gray(img))

def draw_eclipses(dataset_dir, image_size=(128, 128, 3), color = (255, 255, 255), background = (0, 0, 0), num=10000):
	path = os.path.join(dataset_dir, 'eclipse')
	os.makedirs(path, exist_ok = True)
	for i in range(num):
		#Make canvas to draw
		canvas = draw_canvas(image_size, color = background)

		#sample some values for random transformation
		#And define center and angle
		random_values = np.random.normal(loc=0, scale = 0.5, size=[2])
		#random_values = (np.random.rand(2) - 0.5)*2
		center = (int(  image_size[0]/2 + random_values[0]*(image_size[0]/5)  ), int(image_size[1]/2) )
		angle = random_values[1] * 60
		width = int(image_size[0]/5)
		height = width * 2
		pt1 = center[0]-int(width/2), center[1]-int(height/2)
		pt2 = center[0]+int(width/2), center[1]+int(height/2)

		#Draw the rectangle into canvas
		draw.eclipse(canvas, center, width, height, angle = angle, color=color, thickness = -1)
		#transform the image
		
		cv2.imwrite(os.path.join(path, 'eclipse_'+str(i)+'.png'), colorspace.rgb2gray(canvas))

def draw_triangles(dataset_dir, image_size=(128, 128, 3), color = (255, 255, 255), background = (0, 0, 0), num=10000):
	path = os.path.join(dataset_dir, 'triangle')
	os.makedirs(path, exist_ok = True)
	for i in range(num):
		#Make canvas to draw
		canvas = draw_canvas(image_size, color = background)

		#sample some values for random transformation
		#And define center and angle
		random_values = np.random.normal(loc=0, scale = 0.5, size=[2])
		#random_values = (np.random.rand(2) - 0.5)*2
		center = (int(  image_size[0]/2 + random_values[0]*(image_size[0]/5)  ), int(image_size[1]/2) )
		#angle = random_values[1] * 60
		angle = 0
		width = int(image_size[0]/4)
		height = width * 3
		pt1 = (center[0], int(center[1]-height/2))
		pt2 = (int(center[0]-width/2), int(center[1] + height/2) )
		pt3 = (int(center[0]+width/2), int(center[1] + height/2) )
		pt_list = [pt1, pt2, pt3]
		
		#Draw the rectangle into canvas
		draw.fillpoly(canvas, pt_list, color=color)
		#transform the image
		img = transform.rotate(canvas, center, angle, 1, border_color=background)
		cv2.imwrite(os.path.join(path, 'triangle'+str(i)+'.png'), colorspace.rgb2gray(img))	



def run(dataset_dir):
	draw_rectangles(dataset_dir, image_size=(64, 64, 3))
	draw_eclipses(dataset_dir, image_size=(64, 64, 3))
	draw_triangles(dataset_dir, image_size=(64, 64, 3))