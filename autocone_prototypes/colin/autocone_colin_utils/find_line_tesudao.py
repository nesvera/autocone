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

class vision():

	def __init__(self):
		
		self.kernel = np.ones((3,3),np.uint8)
		#self.cap = cv2.VideoCapture("/home/dudu/usb_camimage_raw.mp4")
		#self.cap = cv2.VideoCapture(1)
		#ret = self.cap.set(3,1280)
		#ret = self.cap.set(4,720)
		#self.cap = cv2.VideoCapture("/home/dudu/Dropbox/Programming/line_follower/3.mp4")
		#self.cap = cv2.VideoCapture("/home/dudu/Dropbox/Programming/line_follower/chao.webm")
		#for x in range(200):
		#	_ = self.cap.read()
		self.buffer_pontos = []
		self.buffer_size = 5
		self.lista_erros = []
		#self.lista_erros_ransac = []
		#self.Hmatrix = np.array([[-3.90976610e+00,-5.90540798e+00,1.65325824e+03],[-1.24055090e+00,-2.04359973e+01,3.41047888e+03],[-1.22651441e-03,-1.17716292e-02,1.00000000e+00]])
		#self.Hmatrix = np.array([[2.88501164e+01,4.90100565e+01,-9.66583626e+03],[-9.59281605e-01,1.49645379e+02,-2.18268875e+04],[2.27372670e-03,1.54124011e-01,1.00000000e+00]])
		self.vec = [np.array([[1.40618132e+03,0.00000000e+00,5.55750039e+02],[0.00000000e+00,1.41511691e+03,3.70656616e+02],[0.00000000e+00,0.00000000e+00,1.00000000e+00]]),
					np.array([[-8.49002466e-02,1.46694765e+00,-4.45090764e-03,-6.48929635e-03,-5.49671429e+00]])]

		#self.sf = 5

		#h_image = 120
		#w_image = 120
		#h_image*= self.sf
		#w_image*= self.sf

		self.image_size = 1
		self.image_size *= 10**6
		self.lista_erros = []
		#self.h_image = h_image
		#self.w_image = w_image
		self.scale_factor = 0
		self.last_valid_error = 0
		self.Hmatrix_ok = False
		self.arucos = AT.ArucoTaura(aruco.DICT_6X6_250, 0.03, self.vec)
		self.charucos = ch.Charuco(5, 7, 0.04, 0.03, 0.20,self.vec)
		#self.cap = cv2.VideoCapture("/home/dudu/Dropbox/Programming/line_follower/3.mp4")
		
		#self.ransac = linear_model.RANSACRegressor()
		self.im_ajuste = np.zeros((1,1,1),np.uint8)
		cv2.namedWindow('ajuste')
		cv2.createTrackbar('largura','ajuste',20,300,self.nothing)
		cv2.createTrackbar('profundidade','ajuste',20,300,self.nothing)
		cv2.createTrackbar('afastamento','ajuste',100,200,self.nothing)
		
	def teste(self,image):
		#_, image = self.cap.read()
		self.charucos.feed(image)
		self.arucos.feed(image)

		try:
			self.charucos.estimate_pose()
			pos = self.charucos.get_position()
			print pos[0]
			self.charucos.draw()
			
		except:
			pass
		
	def nothing(x):
		pass

	def find_Hmatrix2(self,image):
		print image
		im_ajuste = np.zeros((1,1,1),np.uint8)
		#cv2.imshow("nesvera",image)
		#plt.imshow(image)
		#plt.show()
		#cv2.imshow("oi3",np.zeros((100,100),np.uint8))
		#cv2.waitKey(0)
		#print "depois"
		#_, image = self.cap.read()
		#print image.shape
		lista_pontos_imagem = []
		largura = cv2.getTrackbarPos('largura','ajuste')
		profundidade = cv2.getTrackbarPos('profundidade','ajuste')
		afastamento = cv2.getTrackbarPos('afastamento','ajuste')

		largura /= 2.

		largura/=100.
		profundidade/=100.
		afastamento/=100.

		
		self.charucos.feed(image)
		self.arucos.feed(image)

		try:
			self.charucos.estimate_pose()
			pos = self.charucos.get_position()
			
			altura = 0
			projection_vector = np.array([[-largura,profundidade+afastamento,altura],
											[+largura,profundidade+afastamento,altura],
											[+largura,afastamento,altura],
											[-largura,afastamento,altura]])

			#print self.projection_vector
			

			self.charucos.draw()
			#copia_pos = pos[0].copy()
			#copia_pos[0] = 0
			#copia_pos[-1] = 0
			#Wcopia_pos = cv2.Rodrigues(copia_pos)[0]

			pos[0] = cv2.Rodrigues(pos[0])[0]
			vetor_normal = np.dot(pos[0].T,np.array([0,0,1.0]))
			vetor_normal[-1] = 0
			vetor_normal /= np.linalg.norm(vetor_normal)
			#print vetor_normal
			C = np.vstack(((np.cross(vetor_normal,[0,0,1.]))/np.linalg.norm(np.cross(vetor_normal,[0,0,1.])),vetor_normal,(0,0,1)))
			C = C.T

			for ponto in projection_vector.reshape(4,-1):
				
				#print ponto
				#R3i= -(Cicn)T*R1cn+ (Cicn)T*R2cn 		
				
				#print ponto.shape, pos[1].shapz
				#print pos[0].shape
				#print (np.dot(pos[0].T,-pos[1])).shape, np.dot(pos[0].T,ponto.reshape(3,1)).shape
				#print -np.dot(pos[0].T,pos[1]), ponto, -np.dot(pos[0].T,pos[1]) + ponto
				#print pos[1], ponto
				#print -np.dot(pos[0].T,pos[1]).reshape(3,1), np.dot(pos[0].T,ponto).reshape(3,1)
				
				#C /= np.linalg.norm(C)
				
				print  ponto
				#print vetor_normal
				#print -np.dot(pos[0].T,pos[1]).reshape(3,1), (vetor_normal*ponto).reshape(3,1), "-----"
				#ponto =  -np.dot(pos[0].T,pos[1]).reshape(3,1) + (vetor_normal*ponto).reshape(3,1)
				ponto = np.dot(C, ponto).reshape(3,1) - np.dot(pos[0].T,pos[1]).reshape(3,1)
				ponto[-1]=0


				#print ponto
				#ponto[-1] = 0
				#print pos[1]#,-np.dot(pos[0].T,pos[1]), np.dot(pos[0].T,ponto.reshape(3,1))
				#	print ponto
				point,_ = cv2.projectPoints(ponto.reshape(1,3),cv2.Rodrigues(pos[0])[0],pos[1],self.vec[0],self.vec[1])
				#point,_ = cv2.projectPoints(np.array([0,.1,0]).reshape(1,3),cv2.Rodrigues(pos[0])[0],pos[1],self.vec[0],self.vec[1])
				#point,_ = cv2.projectPoints(np.dot(copia_pos.T,ponto).reshape(1,3),np.zeros(3,np.float32).reshape(1,3),np.zeros(3,np.float32).reshape(1,3),self.vec[0],self.vec[1])
				point = point.ravel()
				print point
				#print point
				cv2.circle(image,(int(point[0]),int(point[1])),5,(255,0,0),-1)
				lista_pontos_imagem.append(point)
				

			self.scale_factor = (self.image_size/(largura*2*profundidade))**(.5)
			#print self.scale_factor, "--------------"
			pontos_referencia = np.array([[0,0.],
										[largura*self.scale_factor,0],
										[largura*self.scale_factor,profundidade*self.scale_factor],
										[0,profundidade*self.scale_factor]])
			print pontos_referencia
			#self.projection_vector = self.projection_vector - pos[:-1]
			#print self.projection_vector
			lista_pontos_imagem = np.array(lista_pontos_imagem).reshape(1,4,2)
			#print lista_pontos_imagem
			
			h, status = cv2.findHomography(lista_pontos_imagem, pontos_referencia)
			
			self.h_image = int(largura*self.scale_factor)
			self.w_image = int(profundidade*self.scale_factor)

			image2 = cv2.warpPerspective(image, h, (int(largura*self.scale_factor),int(profundidade*self.scale_factor)))
			print "a"
			print image2
			'''
			cv2.imshow("oi",image)
			cv2.imshow("oi2",image2)
			cv2.waitKey(1)
			print "b"
			if cv2.waitKey(1) == 27:
				
				self.Hmatrix_ok = True
				return h
			'''
		except:
			pass
			
		
	def find_Hmatrix(self):
		while True:
			#_, image = self.cap.read()
			print "erro"
			cv2.imshow("image",image)
			if cv2.waitKey(0) == 27: break

		lista_entrada = []
		for x in range(4):
			lista_entrada.append(cv2.selectROI('MultiTracker',image)[:4])
		
		lista_entrada_original = lista_entrada
		lista_entrada = np.array(lista_entrada_original).reshape(1,4,4).astype(np.float32)
		lista_entrada = lista_entrada[0,:,:2]+lista_entrada[0,:,2:]/2
		lista_entrada = np.array(lista_entrada).reshape(1,4,2).astype(np.float32)



		h_linha = 30.*self.sf
		w_linha = 5.*self.sf

		h_image = self.h_image 
		w_image = self.w_image 

		lista_referencia_linha = np.array([[(w_image-w_linha)/2,h_image-2*h_linha,0]
										   ,[(w_image-w_linha)/2+w_linha,h_image-2*h_linha,0]
										   ,[(w_image-w_linha)/2+w_linha,h_image-1*h_linha,0]
										   ,[(w_image-w_linha)/2,h_image-1*h_linha,0]
										  ]).reshape(1,4,3).astype(np.float32)

		h, status = cv2.findHomography(lista_entrada, lista_referencia_linha)
		print np.array(h)
		#self.cap = cv2.VideoCapture("/home/dudu/Dropbox/Programming/line_follower/teste.mp4")
		self.Hmatrix = h


	def find_error(self,image):
		
		if not self.Hmatrix_ok:
			self.find_Hmatrix2(image)
			return 0
		
		#print "oi"
		erro=0
		#ret, image = self.cap.read()
		imagem_copia_original = image.copy()
		time_1 = time.time()
		image = cv2.warpPerspective(image, self.Hmatrix, (self.w_image, self.h_image))
		
		fator_reducao = 4.
		image = cv2.resize(image,(0,0),fx=1./fator_reducao,fy=1./fator_reducao)
		
		print_image = image.copy()
		image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		curve = np.zeros(image.shape,np.uint8)
		mean_image = image.copy()
		original = image.copy()
	
		image = np.float32(image)
		#image = cv2.medianBlur(image,5)
		#image = cv2.GaussianBlur(image,(5,5),0)
		image = cv2.adaptiveThreshold(image.astype(np.uint8),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,3)
		image = cv2.erode(image,self.kernel,iterations = 2)
		image = cv2.dilate(image,self.kernel,iterations = 1)
		total_area = image.shape[0]*image.shape[1]
		_,contours,_ = cv2.findContours(image, 1, 2)

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
			if largura<8*self.scale_factor/fator_reducao and largura>3*self.scale_factor/fator_reducao and altura>20*self.scale_factor/fator_reducao:
				lista_contornos.append(box)	

			"""
			if ratio<1: ratio = ratio**(-1)
			if area>total_area/500 and area<total_area/20:# and ratio>1.5 and ratio <10:
				lista_contornos.append(cnt)
			"""
		
		
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
		
			"""
			try:
				,,angle = cv2.fitEllipse(cnt)
				if angle>90: angle=180-angle
			except:
				angle = 90
			"""
			
			#   len(approx)>3 and len(approx)<10 and 

			#print(mean_val[0])		

			if mean_val[0]>200: #and len(approx)>3: #and len(approx)<8:
				cv2.drawContours(print_image,[cnt],-1,(255,255,0),5)
				M = cv2.moments(cnt)
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])  
				pontos_frame.append([cx,cy])
				lista_contornos.append([cnt])
				rows,cols = image.shape[:2]
				[vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
				lefty = int((-x*vy/vx) + y)
				righty = int(((cols-x)*vy/vx)+y)
				#try:
				mask_curve = np.zeros(image.shape,np.uint8)
				try:
					#cv2.line(mask_curve,(cols-1,righty),(0,lefty),(255),2)
					#curve[mask==255] = mask_curve[mask==255]
					curve +=mask
				except:
					pass
		try:
			points = np.nonzero(curve)
			polinomio = np.poly1d(np.polyfit(points[0],points[1], 2))
			

			erro = 90-(np.arctan(float(image.shape[0])/float(polinomio(0)-image.shape[1]/2)))*180/np.pi
			if erro>90: erro = erro-180
			
			#self.lista_erros.append(erro)
			#if self.lis
			#print erro
			#print len(self.lista_erros)

			#if len(self.lista_erros)>300:
			#	plt.plot(self.lista_erros,"b")
				
			xp = np.linspace(0, image.shape[0], 300)    
			for x in xp:    
			   cv2.circle(curve, (int(polinomio(x)),int(x)),2,(255,0,0),-1)
		except:
			pass
		
		cv2.imshow('Colin', curve)
		cv2.imshow('image', print_image)
		
		image = cv2.resize(image,(0,0),fx=2,fy=2)
		cv2.imshow("original",image)
		#image = cv2.warpPerspective(original, self.Hmatrix.T, (image.shape[1], image.shape[0]))
		#cv2.imshow("perspectvive", image)

		
		
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
		#if cv2.waitKey(1)==27:
		cv2.waitKey(1)
		print 1000/((time.time()-time_1)*1000)

#classe = vision()
#while True:
#	#classe.teste()
#	classe.find_error()