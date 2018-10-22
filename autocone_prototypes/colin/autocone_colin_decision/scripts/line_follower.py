#!/usr/bin/env python

import rospy

from sensor_msgs.msg import Joy
from ackermann_msgs.msg import AckermannDrive

import sys
import argparse
from sklearn.cluster import DBSCAN
from sklearn import linear_model

fixed_speed = False
fixed_speed_value = 0
max_speed = 0
max_steering = 100

ackermann_cmd = AckermannDrive()
ackermann_pub = None

import cv2
import numpy as np
import time
import collections
import time

class vision():

    def __init__(self):
        self.kernel = np.ones((3,3),np.uint8)
        self.cap = cv2.VideoCapture(1)
        self.buffer_pontos = []
        self.buffer_size = 5
        self.lista_erros = []
        #self.ransac = linear_model.RANSACRegressor()




    def find_error(self):
		
	erro=0
        ret, image = self.cap.read()
#	print image
        time_1 = time.time()
        image = cv2.resize(image,(0,0),fx=0.5,fy=0.5)
	print_image = image.copy()
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        mean_image = image.copy()
        original = image.copy()
	
        image = np.float32(image)
        image = cv2.GaussianBlur(image,(5,5),0)
        image = cv2.adaptiveThreshold(image.astype(np.uint8),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,3)
        image = cv2.erode(image,self.kernel,iterations = 1)
        image = cv2.dilate(image,self.kernel,iterations = 1)
        total_area = image.shape[0]*image.shape[1]
        _,contours,_ = cv2.findContours(image, 1, 2)

        lista_contornos = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            a = box[0]-box[1]
            b = box[1]-box[2]
            ratio = np.linalg.norm(a)/np.linalg.norm(b)
            if ratio<1: ratio = ratio**(-1)
            if area>total_area/500 and area<total_area/20 and ratio>1.5 and ratio <10:
                lista_contornos.append(cnt)
		
		
        contours = lista_contornos
        lista_contornos = []
        pontos_frame = []
        #cv2.drawContours(original, contours, -1, (0,0,255), 3)
	
        #vencedor = (image.shape[1],None) #ycentro, x0
        for cnt in contours:
            epsilon = 0.01*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)
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

            if mean_val[0]>150 and len(approx)>3: #and len(approx)<8:
		cv2.drawContours(print_image,[	approx],-1,(255,255,0),5)
                M = cv2.moments(approx)
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])  
		pontos_frame.append([cx,cy])
		lista_contornos.append([approx])

                #if cy<vencedor[0]:
                    #cruzay = int(vx/abs(vy)*cy)
                    #vencedor = (cy,approx)
                #print righty
		

	if len(pontos_frame)>0:
		self.buffer_pontos.append(pontos_frame)
	elif len(self.buffer_pontos)>0:
		self.buffer_pontos.pop(0)
	
	if len(self.buffer_pontos)>self.buffer_size:
		self.buffer_pontos.pop(0)

	lista_pontos_buffer = []

	for frame in self.buffer_pontos:
		for ponto in frame:
			lista_pontos_buffer.append(ponto)
	
	pontos_frame = np.array(pontos_frame).reshape(-1,2)

	lista_pontos_buffer = np.array(lista_pontos_buffer).reshape(-1,2)
	if lista_pontos_buffer.shape[0]>0:
		db = DBSCAN(eps=20, min_samples=4).fit(lista_pontos_buffer)
		labels = db.labels_
		#print labels
		n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
		if len(pontos_frame)==0:
			labels_frame = []
		else:
			labels_frame = labels[-len(pontos_frame):]
		
		center_contours = []
		center_clusters = []
		for cluster in range(n_clusters_):
			#print pontos_frame.shape, labels_frame.shape, cluster
			pontos_centro = pontos_frame[labels_frame==cluster]
			if pontos_centro.shape[0]>0:
				center_clusters.append(pontos_centro[-1])#np.mean(pontos_frame,axis=0))
				#center_clusters = np.array(center_clusters).reshape(-1,2)
			
		
		for ponto in center_clusters:
			cv2.circle(original,(int(ponto[0]),int(ponto[1])),7,(0,0,255),-1)
		
		mais_longe = [0,image.shape[0]]
		for ponto in center_clusters:
		    if ponto[1]<mais_longe[1]:
		        mais_longe=ponto
		
		if mais_longe[1] < image.shape[0]:
		   
		    try:
		        erro = 90-(np.arctan(float(mais_longe[1])/float(mais_longe[0]-image.shape[1]/2.)))*180./np.pi
		    except:
		        erro = 0
		"""    
		    self.lista_erros.append(erro)
		if len(self.lista_erros)>5:
		    self.lista_erros.pop(0)
		
		if len(self.lista_erros)==5:
		    
		    	try:
			    ransac = linear_model.RANSACRegressor()
			    ransac.fit(np.arange(5).reshape(-1,1),np.array(self.lista_erros).reshape(-1,1))
		    	    erro = ransac.predict(5)
		    	    print "foi"
		    	except:
		    	    erro = 0
	    	
	    	#self.lista_erros(0)#	    return True
			#except:
			#	pass
		"""


 
				
        #   erro = 90-(np.arctan(float(image.shape[0])/float(value_x-image.shape[1]/2.)))*180./np.pi
        if erro ==0: x_value=image.shape[1]/2
	else: x_value = int(image.shape[1]/2+np.tan(erro/180.*np.pi)*image.shape[0])
        cv2.line(original,(int(image.shape[1]/2),image.shape[0]),(x_value,0),(255,0,0),3)
        cv2.imshow('Colin', image)
        cv2.imshow('image', print_image)
	cv2.imshow("original",original)
        cv2.waitKey(1)
        if erro>90: erro = erro-180
        #print 1000/((time.time()-time_1)*1000)
        return erro

def joy_callback(data):
    global new_data 

    axes = data.axes
    buttons = data.buttons

    left_trigger = data.axes[2]
    right_trigger = data.axes[5]
    left_x_stick = data.axes[0]

    forward = (-right_trigger+1)/2.
    backward = (-left_trigger+1)/2.

    speed = forward*max_speed
    #steering = left_x_stick*max_steering

    ackermann_cmd.speed = float(speed)

    new_data = True

def routine():
    global teste
    
    steer_list = collections.deque(maxlen=10)

    while not rospy.is_shutdown():

        #chama funcao do catani
        #ackermann_cmd.steering_angle = float(steering)

        erro = teste.find_error()
	#print "recebeu"
        #print("erro = "+str(erro))

        steer = 100*(-erro/30.)
        steer_list.append(steer)
	
        avg_steer = 0
        for i in steer_list:
            avg_steer += i 

        avg_steer = avg_steer / 10.

        ackermann_cmd.steering_angle = int(steer)

        ackermann_pub.publish(ackermann_cmd)

        rate.sleep()
                

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fixed_speed', action='store', dest='fix_speed',
                        default=0, required=False,
                        help="Value of fixed velocity.")
    parser.add_argument('-m ', '--max_speed', action='store', dest='max_speed',
                        default=50, required=False,
                        help="Limit of speed.")

    arguments = parser.parse_args(rospy.myargv()[1:])
    
    # car running on fixed speed
    fixed_speed_value = float(arguments.fix_speed)
    max_speed = float(arguments.max_speed)

    if fixed_speed_value > 0:
        fixed_speed = True

    rospy.init_node('line_follower', anonymous=True)

    rate = rospy.Rate(30)    

    rospy.Subscriber('/joy', Joy, joy_callback) 
    ackermann_pub = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=1)
    teste = vision()

    routine()

