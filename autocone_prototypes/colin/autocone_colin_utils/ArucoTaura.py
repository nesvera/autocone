import cv2
import numpy as np
import cv2.aruco as aruco

# Author: Eduardo Cattani
# TauraBots Team
# V 1.01 (Bugs corrected)

# Tutorial:
## Create the ArucoTaura object
## Feed the object with the image using feed(frame)
## After that, you are able to Draw the Arucos in the frame and find the distances
## The input vec -->> vec = [newcameraMatrix, dist, rvecs, tvecs]


class ArucoTaura:

	def __init__(self,dictionary,arucolen,vec=[0,0]): 
		# dictionary - the used dictionary, ex: aruco.DICT_6X6_250
		# vec - the vector returned by the calibration program
		# arucolen - the real size of aruco, ex: 0.1 (in meters, 10 cm)
		self.aruco_dict = aruco.Dictionary_get(dictionary)
		self.newcameraMatrix, self.dist = vec[0],vec[1]
		self.parameters = aruco.DetectorParameters_create()
		self.arucolen = arucolen


	def feed(self,frame):
		# This function feeds the class with the image to be used, just put the image as parameter
		# That has to be the firt thing before call another function
		# The image can be BGR or Gray
		self.frame = frame
		if len(frame.shape) > 2:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		self.corner, self.ids, __ = aruco.detectMarkers(frame, self.aruco_dict, parameters = self.parameters)

		try:
			self.ids!=None
			self.existID = True
		except: 
			self.existID = False
		
		self.estimated = False


	def drawAruco(self, axis = False):
		# Return the drawned image
		# Draw the marked Aruco in imageCD 
		# Take the second parameter as True to draw the axis
		if axis:
			self.drawArucoAxis()
		self.frame = aruco.drawDetectedMarkers(self.frame, self.corner,self.ids,255)
		return self.frame
	

	
	def findDistance(self, idselected):
		# Return a vector [x,y,z] with the distances from the camera
		# One Id at each time
		# There is no necessity to call the drawAruco() before this function
		if self.existID:
			if idselected in self.ids:
				if self.estimated == False:
					self.estimateArucoPose()
				indexofid = np.nonzero(self.ids == idselected)[0][0]
				x = (self.atvec[indexofid])[0][0][0]*100
				y = (self.atvec[indexofid])[0][0][1]*100
				z = (self.atvec[indexofid])[0][0][2]*100
				return [x,y,z]
	
	def findCorners(self, idselected):
		if self.existID:
			try:
				if idselected in self.ids:
					indexofid = np.nonzero(self.ids == idselected)[0][0]
					
					return self.corner[indexofid][0]
				return False
			except:
				return False
		return False

	def findM(self, idselected):
		if self.existID:
			if idselected in self.ids:
				if self.estimated == False:
					self.estimateArucoPose()
				indexofid = np.nonzero(self.ids == idselected)[0][0]
				matrix = [(self.avec[indexofid])[0][0],(self.atvec[indexofid])[0][0]]
				#print matrix[1]
				vector = []
				#vector, jacobian = cv2.Rodrigues(matrix,jacobian=0)
				
				return matrix
			return False
		return False

	def findPos(self, idselected):
		if self.existID:
			if idselected in self.ids:
				if self.estimated == False:
					self.estimateArucoPose()
				indexofid = np.nonzero(self.ids == idselected)[0][0]
				matrix = [(self.avec[indexofid])[0][0],(self.atvec[indexofid])[0][0]]
				#print matrix[1]
				vector = []
				#vector, jacobian = cv2.Rodrigues(matrix,jacobian=0)
				
				return matrix
			return False
		return False	

	def create(self, idaruco, size = 100):
		img = aruco.drawMarker(self.aruco_dict, idaruco, size)
		return img

	# the next function is called from other functions
	
	def drawArucoAxis(self): #Function used by other one
		if self.existID:
			self.estimateArucoPose()
			for x in xrange(self.n):
				self.frame = aruco.drawAxis(self.frame, self.newcameraMatrix, self.dist, self.avec[x], self.atvec[x], self.arucolen) 
			return self.frame
		
	def estimateArucoPose(self): #Function used by other one
		self.estimated = True
		if self.existID:
			self.n = len(self.ids)
			self.avec = [[] for x in xrange(self.n)]
			self.atvec = [[] for x in xrange(self.n)]
			for x in xrange(self.n):
				self.avec[x],self.atvec[x] = aruco.estimatePoseSingleMarkers(self.corner[x], self.arucolen, self.newcameraMatrix, self.dist)

