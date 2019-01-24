#!/usr/bin/env python

import cv2
import numpy as np
import time
import collections
import time

import cv2.aruco as aruco
import ArucoTaura as AT
import Charuco as ch
import matplotlib.pyplot as plt
import math

# adaptive threshold -> ganho, vizinhaca
# erode
# dilate
# max, min largura linha
# max, min altura linha
# min mean value

class Vision():

	def __init__(self):

		self.kernel = np.ones((3,3),np.uint8)
		#self.buffer_pontos = []
		#self.buffer_size = 5
		#self.lista_erros = []
		self.vec = [np.array([[1.40618132e+03,0.00000000e+00,5.55750039e+02],[0.00000000e+00,1.41511691e+03,3.70656616e+02],[0.00000000e+00,0.00000000e+00,1.00000000e+00]]),
					np.array([[-8.49002466e-02,1.46694765e+00,-4.45090764e-03,-6.48929635e-03,-5.49671429e+00]])]
		# parametros internos e distorcao = vec

		#self.w_image = 1280
		#self.h_image = 720

		self.image_size = 1 #tamanho em mp
		self.image_size *= 10**6 
		#self.lista_erros = []
		self.scale_factor = 0
		self.last_valid_error = 0
		#self.Hmatrix_ok = False
		#self.Hmatrix = None

		self.arucos = AT.ArucoTaura(aruco.DICT_6X6_250, 0.03, self.vec)
		self.charucos = ch.Charuco(5, 7, 0.04, 0.03, 0.20,self.vec)

		self.bird_src = np.zeros((4,2), dtype="float32")
		self.bird_dst = np.zeros((4,2), dtype="float32")

	def find_Hmatrix2(self, image):
		
		
		lista_pontos_imagem = []

		largura = self.largura/100.
		profundidade = self.profundidade/100.
		afastamento = self.afastamento/100.
		offset = self.offset/100.

		largura /= 2.

		self.charucos.feed(image)
		self.arucos.feed(image)

		try:
			
			self.charucos.estimate_pose()
			pos = self.charucos.get_position()
			# tudo em centimetro

			altura = 0
			projection_vector = np.array(	[[-largura+offset,	profundidade+afastamento,altura],
											[+largura+offset,	profundidade+afastamento,altura],
											[+largura+offset,	afastamento,altura],
											[-largura+offset,	afastamento,altura]])

			self.charucos.draw()

			pos[0] = cv2.Rodrigues(pos[0])[0]
			vetor_normal = np.dot(pos[0].T,np.array([0,0,1.0]))
			vetor_normal[-1] = 0
			vetor_normal /= np.linalg.norm(vetor_normal)
			
			C = np.vstack(((np.cross(vetor_normal,[0,0,1.]))/np.linalg.norm(np.cross(vetor_normal,[0,0,1.])),vetor_normal,(0,0,1)))
			C = C.T

			for ponto in projection_vector.reshape(4,-1):
				
				ponto = np.dot(C, ponto).reshape(3,1) - np.dot(pos[0].T,pos[1]).reshape(3,1)
				ponto[-1]=0

				point,_ = cv2.projectPoints(ponto.reshape(1,3),cv2.Rodrigues(pos[0])[0],pos[1],self.vec[0],self.vec[1])

				point = point.ravel()
				
				cv2.circle(image,(int(point[0]),int(point[1])),5,(255,0,0),-1)
				lista_pontos_imagem.append(point)
				

			self.scale_factor = (self.image_size/(largura*2*profundidade))**(.5)
			
			pontos_referencia = np.array([[0,0.],
										[largura*2*self.scale_factor,0],
										[largura*2*self.scale_factor,profundidade*self.scale_factor],
										[0,profundidade*self.scale_factor]])
			
			lista_pontos_imagem = np.array(lista_pontos_imagem).reshape(1,4,2)
						
			h, status = cv2.findHomography(lista_pontos_imagem, pontos_referencia)
			
			self.h_image = int(profundidade*self.scale_factor)
			self.w_image = int(largura*self.scale_factor*2)

			image2 = cv2.warpPerspective(image, h, (int(self.w_image),int(self.h_image)))
			

			cv2.imshow("oi",image)
			cv2.imshow("oi2",image2)

			return h

		except:
			pass

	def set_parameters(self, parameters):
		self.Hmatrix = np.array(parameters['hmatrix'])
		self.largura = parameters['largura'] + 0.01
		self.profundidade = parameters['profundidade'] + 0.01
		self.afastamento = parameters['afastamento'] + 0.01
		self.offset = parameters['offset']-255

		self.scale_factor = (self.image_size/(self.largura*2*self.profundidade))**(.5)
		self.h_image = int(self.profundidade*self.scale_factor)
		self.w_image = int(self.largura*self.scale_factor)

		self.warp_w = parameters['warp_w']
		self.warp_h = parameters['warp_h']

		self.linha_largura_min = parameters['linha_largura_min']
		self.linha_largura_max = parameters['linha_largura_max']
		self.linha_altura_min =  parameters['linha_altura_min']
		self.linha_altura_max =  parameters['linha_altura_max']
	
	def find_error(self, image):
		
		erro=0

		center_offset = 0
		
		image = cv2.cvtColor(image.copy(),cv2.COLOR_BGR2GRAY)
		image = cv2.warpPerspective(image, self.Hmatrix, (self.warp_w, self.warp_h))

		fator_reducao = 2.
		image = cv2.resize(image,(0,0),fx=1./fator_reducao,fy=1./fator_reducao)

		print_image = image.copy()
		
		curve = np.zeros(image.shape,np.uint8)
		mean_image = image.copy()
		original = image.copy()
	
		image = np.float32(image)
		image = cv2.medianBlur(image,5)
		#image = cv2.GaussianBlur(image,(5,5),0)
		image = cv2.adaptiveThreshold(image.astype(np.uint8),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,3)
		image = cv2.erode(image,self.kernel,iterations = 2)
		image = cv2.dilate(image,self.kernel,iterations = 1)
		total_area = image.shape[0]*image.shape[1]
		_,contours,_ = cv2.findContours(image, 1, 2)

		cv2.drawContours(print_image, contours, -1, (0, 0, 255), 3)

		lista_contornos = []
		for cnt in contours:

			area = cv2.contourArea(cnt)
			rect = cv2.minAreaRect(cnt)
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			#print box
			a = box[0]-box[1]
			b = box[1]-box[2]
			vector = (np.linalg.norm(a),np.linalg.norm(b))
			largura = min(vector)#/np.linalg.norm(b)
			altura = max(vector)
			#print a, np.linalg.norm(a)
			#if largura<8*self.scale_factor/fator_reducao and largura>3*self.scale_factor/fator_reducao and altura>20*self.scale_factor/fator_reducao:
			#	lista_contornos.append(box)	

			#print(largura, altura)
			#print(self.linha_altura_min, self.linha_altura_max, self.linha_largura_min, self.linha_largura_max)
			if largura<self.linha_largura_max and largura>self.linha_largura_min and altura>self.linha_altura_min:
				lista_contornos.append(box)

		contours = lista_contornos
		lista_contornos = []
		pontos_frame = []
		cv2.drawContours(original, contours, -1, (0,0,255), 3)
	
		#vencedor = (image.shape[1],None) #ycentro, x0
		for cnt in contours:
			#epsilon = 0.01*cv2.arcLength(cnt,True)
			#approx = cv2.approxPolyDP(cnt,epsilon,True)
			mask = np.zeros(image.shape,np.uint8)
			cv2.drawContours(mask,[cnt],0,255,-1)
			mean_val = cv2.mean(mean_image,mask = mask)

			if mean_val[0]>200: #and len(approx)>3: #and len(approx)<8:
				cv2.drawContours(print_image,[cnt],-1,(255,255,0),5)
				M = cv2.moments(cnt)
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])  
				pontos_frame.append([cx,cy])
				lista_contornos.append([cnt])
			
				mask_curve = np.zeros(image.shape,np.uint8)
				try:
					curve +=mask
				except:
					pass


		try:
			points = np.nonzero(curve)
			polinomio = np.poly1d(np.polyfit(points[0],points[1], 2))
			

			erro = 90-(np.arctan(float(image.shape[0])/float(polinomio(0)-image.shape[1]/2)))*180/np.pi
			if erro>90: erro = erro-180
				
			xp = np.linspace(0, image.shape[0], 30)    
			for x in xp:    
			   	cv2.circle(curve, (int(polinomio(x)),int(x)),2,(255,0,0),-1)
				pass

		except:
			pass
		
		
		cv2.imshow('image', print_image)
		
		image = cv2.resize(image,(0,0),fx=2,fy=2)
		cv2.imshow("original",image)		
		
		if erro>90: erro = erro-180
		if erro!=0: self.last_valid_error = erro
		if erro == 0: erro = self.last_valid_error
		try:
			deslocamento_x = int(np.sin(abs(erro)/180*np.pi)*300)#-imagem_copia_original.shape[1]/2
			deslocamento_y = int(np.cos(abs(erro)/180*np.pi)*300)
			if erro<0: deslocamento_x*=-1
			#print deslocamento,"--------"
			#if erro<0:
			xp = imagem_copia_original.shape
			cv2.line(imagem_copia_original, (xp[1]/2,xp[0]),(deslocamento_x+xp[1]/2,xp[0]-deslocamento_y),(0,0,255),4)
			cv2.imshow("final", cv2.resize(imagem_copia_original,(0,0),fx=0.5,fy=0.5))
		except:
			pass

		try:
			#return erro
			
			#calcular velocidade
			p1 = (polinomio(400), 400)
			#p2 = (image.shape[0]/2., polinomio(image.shape[0]/2.))
			p2 = (polinomio(200), 200)

			cv2.circle(curve, (int(p1[0]),int(p1[1])),10,(255,255,0),-1)
			cv2.circle(curve, (int(p2[0]),int(p2[1])),10,(255,255,0),-1)

			dx = p2[0] - p1[0]
			dy = p2[1] - p1[1]

			if dx == 0:
				angle_rad = math.pi/2.
				angle = 90

			else:
				angle_rad = np.arctan(dy/dx)
				angle = angle_rad*180/math.pi

				# direita
				if angle < 0:
					angle = -angle

				else:
					angle = 180 - angle


			# Offset x of the car related with the line
			center_offset = -(curve.shape[1]/2.)-p1[0]
			#upper_point = -(curve.shape[0]/np.tan(angle_rad))
			upper_point = polinomio(0)

			#print(center_offset, upper_point)

			cv2.imshow('Colin', curve)

			return center_offset, upper_point

		except:
			pass