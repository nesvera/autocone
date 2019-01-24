#!/usr/bin/env python

import cv2
import numpy as np
import time
import collections
import time
import sys
import os
import signal
import pickle

import cv2.aruco as aruco
import ArucoTaura as AT
import Charuco as ch

from vision import Vision

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

img_source = "cv2"
parameters = None
received_img = False
ros_image = None

class Parameters:
	def __init__(self):
		self.largura = 1
		self.profundidade = 1
		self.afastamento = 1
		self.hmatrix = 1
		self.offset = 1
		self.bird_scale = 1
		self.bird_offset_x = 500
		self.bird_offset_y = 500
		self.warp_w = 800
		self.warp_h = 800
		self.linha_largura_min = 5
		self.linha_largura_max = 15
		self.linha_altura_min = 5
		self.linha_altura_max = 15
		

	def set(self, dict):
		self.largura 			= dict['largura']
		self.profundidade 		= dict['profundidade']
		self.afastamento 		= dict['afastamento']
		self.hmatrix			= np.array(dict['hmatrix'])
		self.offset				= dict['offset']
		self.bird_scale 		= dict['bird_scale']
		self.bird_offset_x 		= dict['bird_offset_x']
		self.bird_offset_y  	= dict['bird_offset_y']
		self.linha_largura_min 	= dict['linha_largura_min']
		self.linha_largura_max  = dict['linha_largura_max']
		self.linha_altura_min 	= dict['linha_altura_min']
		self.linha_altura_max 	= dict['linha_altura_max']

class Calibrator:

	def __init__(self):
		self.parameters = Parameters()
		self.hmatrix_registered = False

		self.mousePosX = 0
		self.mousePosY = 0

		self.points_plane = np.zeros((4,2), dtype=np.float32)
		
		self.points_camera = np.zeros((4,2), dtype=np.float32)

		self.click_ind = 0
		self.homography_calculated = False

		self.output_scale = 10.

		self.frame = np.zeros((1,1))
		self.new_frame = False

		# Set points of the plane on world coordinates
		self.points_plane[0] = np.array([-200/self.output_scale, 	800/self.output_scale])
		self.points_plane[1] = np.array([200/self.output_scale, 	800/self.output_scale])
		self.points_plane[2] = np.array([200/self.output_scale,		600/self.output_scale])
		self.points_plane[3] = np.array([-200/self.output_scale, 	600/self.output_scale])

		self.points_plane += np.float32([500,800])


		# Load old parameters if file exist
		if os.path.exists(file_path):

			filehandler = open(file_path, 'r')
			parameters_dict = pickle.load(filehandler)
			filehandler.close()	
			
			self.parameters.set(parameters_dict)

		self.create_track_bar(None)

	def create_track_bar(self, loaded_param):

		cv2.namedWindow('Parametros')
		cv2.createTrackbar('largura',			'Parametros', 0,		255, 	nothing)
		cv2.createTrackbar('profundidade',		'Parametros', 0,		255, 	nothing)
		cv2.createTrackbar('afastamento',		'Parametros', 0,		255, 	nothing)
		cv2.createTrackbar('offset',			'Parametros', 0,		255*2, 	nothing)
		cv2.createTrackbar('bird_scale',		'Parametros', 0,		100, 	nothing)
		cv2.createTrackbar('bird_offset_x',		'Parametros', 0,		1000, 	nothing)
		cv2.createTrackbar('bird_offset_y',		'Parametros', 0,		1000, 	nothing)
		cv2.createTrackbar('linha_largura_min',		'Parametros', 0,	50, 	nothing)
		cv2.createTrackbar('linha_largura_max',		'Parametros', 0,	50, 	nothing)
		cv2.createTrackbar('linha_altura_min',		'Parametros', 0,	50, 	nothing)
		cv2.createTrackbar('linha_altura_max',		'Parametros', 0,	50, 	nothing)

		cv2.setTrackbarPos('largura',			'Parametros', 	self.parameters.largura )
		cv2.setTrackbarPos('profundidade',		'Parametros', 	self.parameters.profundidade )
		cv2.setTrackbarPos('afastamento',		'Parametros', 	self.parameters.afastamento )
		cv2.setTrackbarPos('offset',			'Parametros', 	self.parameters.offset )
		cv2.setTrackbarPos('bird_scale',		'Parametros', 	self.parameters.bird_scale )
		cv2.setTrackbarPos('bird_offset_x',		'Parametros', 	self.parameters.bird_offset_x )
		cv2.setTrackbarPos('bird_offset_y',		'Parametros', 	self.parameters.bird_offset_y )
		cv2.setTrackbarPos('linha_largura_min',	'Parametros', 	self.parameters.linha_largura_min )
		cv2.setTrackbarPos('linha_largura_max',	'Parametros', 	self.parameters.linha_largura_max )
		cv2.setTrackbarPos('linha_altura_min',	'Parametros', 	self.parameters.linha_altura_min )
		cv2.setTrackbarPos('linha_altura_max',	'Parametros', 	self.parameters.linha_altura_max )


	def track_bar_update(self):
		self.parameters.largura 		= cv2.getTrackbarPos('largura',			'Parametros')
		self.parameters.profundidade 	= cv2.getTrackbarPos('profundidade', 	'Parametros')
		self.parameters.afastamento		= cv2.getTrackbarPos('afastamento', 	'Parametros')
		self.parameters.offset			= cv2.getTrackbarPos('offset', 			'Parametros')
		self.parameters.bird_scale		= cv2.getTrackbarPos('bird_scale', 		'Parametros')
		self.parameters.bird_offset_x	= cv2.getTrackbarPos('bird_offset_x', 	'Parametros')
		self.parameters.bird_offset_y	= cv2.getTrackbarPos('bird_offset_y', 	'Parametros')
		self.parameters.linha_largura_min	= cv2.getTrackbarPos('linha_largura_min', 	'Parametros')
		self.parameters.linha_largura_max	= cv2.getTrackbarPos('linha_largura_max', 	'Parametros')
		self.parameters.linha_altura_min	= cv2.getTrackbarPos('linha_altura_min', 	'Parametros')
		self.parameters.linha_altura_max	= cv2.getTrackbarPos('linha_altura_max', 	'Parametros')
		
	def routine(self):

		print("Press g to register wich point on the image")
		print("Press l to load homography matrix")
		print("Press SPACE to register the homography matrix")
		print("Press ESC/CTRL+C to save the parameters")

		# Use opencv to open camera
		if img_source == "cv2":
			cap = cv2.VideoCapture(0)
			cap.set(4,720)
			cap.set(3,1280)
			ret,frame = cap.read()

		# Wait for ros topic
		elif img_source == "ros":
			while received_img == False:
				pass

		while True:			
			self.track_bar_update()
			vision.set_parameters(self.parameters.__dict__)

			if img_source == "cv2":
				_, self.frame  = cap.read()
	
			else:
				self.frame = ros_image

			# Find homography matrix
			#self.h = vision.find_Hmatrix2(self.frame)

			if self.click_ind == 4:

				# Set points of the plane on world coordinates
				self.points_plane[0] = np.array([-200/self.parameters.bird_scale, 	800/self.parameters.bird_scale])
				self.points_plane[1] = np.array([200/self.parameters.bird_scale, 	800/self.parameters.bird_scale])
				self.points_plane[2] = np.array([200/self.parameters.bird_scale,	600/self.parameters.bird_scale])
				self.points_plane[3] = np.array([-200/self.parameters.bird_scale, 	600/self.parameters.bird_scale])

				self.points_plane += np.float32([self.parameters.bird_offset_x, self.parameters.bird_offset_y])
				
				self.homography_matrix = cv2.getPerspectiveTransform(self.points_camera, self.points_plane)
				self.inv_homography_matrix = np.linalg.inv(self.homography_matrix)

				self.homography_calculated = True


			if self.homography_calculated == True:
				screen_position = np.float32((self.mousePosX, self.mousePosY,1)).reshape(3,1)
				plane_position = np.matmul(self.homography_matrix, screen_position)
				plane_position[0] /= plane_position[2]
				plane_position[1] /= plane_position[2]

				#print("Camera (%d, %d) -> Plane (%f, %f)" % (screen_position[0], screen_position[1], plane_position[0], plane_position[1]))

				warped =cv2.warpPerspective(self.frame, self.homography_matrix, (self.parameters.warp_w, self.parameters.warp_h))

				cv2.imshow('warped', warped)

				vision.find_error(self.frame)


			self.draw()
			

	def draw(self):
								
		painted = self.frame.copy()
		# draw mouse cursor
		cv2.circle(painted, (int(self.mousePosX), int(self.mousePosY)), 5, (0,255,0), -1)			

		# draw 
		for i in range(4):
			cv2.circle(painted, (int(self.points_camera[i, 0]), int(self.points_camera[i, 1])), 5, (0,0,255), -1)	

		cv2.setMouseCallback('Painted', click_handler)
		cv2.imshow('Painted', painted)	
		c = cv2.waitKey(1) 

		if c == 32:
			print('Homography matrix registered')
			self.parameters.hmatrix = self.homography_matrix
		
		elif c == 27:
			print('Saving parameters!')
			self.save_parameters()
			exit(0)

		elif c == ord('l'):
			self.homography_matrix = self.parameters.hmatrix
			self.homography_calculated = True
		
		elif c == ord('g'):
			self.click_ind += 1

	def save_parameters(self):
		"""
		Save parameters adjusted in the track bar window in a pickle file.
		The parameters are stored as a dictionary
		"""
		filehandler = open(file_path, 'w')
		class_to_dict = self.parameters.__dict__
		pickle.dump(class_to_dict, filehandler, pickle.HIGHEST_PROTOCOL)		

	def on_click(self, event, x, y):
		if event == cv2.EVENT_MOUSEMOVE:
			self.mousePosX = x
			self.mousePosY = y
		
		if event == cv2.EVENT_LBUTTONUP:

			if self.click_ind < 4:
				self.points_camera[self.click_ind] = np.array([x, y])


def click_handler(event, x, y, flags, param):
	calibrator.on_click(event, x, y)	

def nothing(arg1):
	pass

def exit_handler(signal, frame):
	calibrator.save_parameters()
	exit(0)

def image_callback(data):
	global ros_image, received_img
    
	# Convert image from ROS format to cv2	
	try:
		cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
		
	
	except CvBridgeError as e:
		pass

	ros_image = cv_image
	received_img = True

if __name__ == '__main__':
	global file_path, calibrator, vision
	
	if len(sys.argv) < 2:
		print("Error!")
		print("Usage: python " + sys.argv[0] + " file.json" + " ros/cv2")
		exit(0)

	file_path = sys.argv[1]

	if len(sys.argv) > 2:
		img_source = sys.argv[2]


	vision = Vision()
	calibrator = Calibrator()

	# set function to save when exit de script
	signal.signal(signal.SIGINT, exit_handler)

	if img_source == "ros":
		rospy.init_node('calibrator', anonymous=True)

		# Subscribe to camera topic
		rospy.Subscriber('/usb_cam/image_raw', Image, image_callback)
		bridge = CvBridge()

	calibrator.routine()