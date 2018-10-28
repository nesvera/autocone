import cv2
import numpy as np
import time
import collections
import time
        
class vision():

    def __init__(self):
        self.kernel = np.ones((3,3),np.uint8)
        #self.cap = cv2.VideoCapture("/home/dudu/usb_camimage_raw.mp4")
        #self.cap = cv2.VideoCapture(0)

        #_, frame = self.cap.read()
        #print(frame)
        #raw_input()

        #self.cap = cv2.VideoCapture("/home/dudu/Dropbox/Programming/line_follower/3.mp4")
        #for x in range(200):
        
        self.buffer_pontos = []
        self.buffer_size = 5
        self.lista_erros = []
        self.lista_erros_ransac = []
        #self.Hmatrix = np.array([[-3.90976610e+00,-5.90540798e+00,1.65325824e+03],[-1.24055090e+00,-2.04359973e+01,3.41047888e+03],[-1.22651441e-03,-1.17716292e-02,1.00000000e+00]])

        #self.Hmatrix = np.array([[-2.94906603e+00,-5.51042566e+00,1.06802055e+03],[2.76648507e-01,-1.48718156e+01,2.81498555e+03],[6.89445335e-05,-1.88569010e-02,1.00000000e+00]])
        #
        #self.Hmatrix = np.array([[ -4.70371600e-01,  -1.89422268e+00,   6.55107687e+02],
        # [ -6.20593453e-03, -4.10899131e+00,   1.39520663e+03],
        # [  1.77982117e-05,  -4.80337451e-03,  1.00000000e+00]])

        #self.Hmatrix = np.array([[2.88501164e+01,4.90100565e+01,-9.66583626e+03],[-9.59281605e-01,1.49645379e+02,-2.18268875e+04],[2.27372670e-03,1.54124011e-01,1.00000000e+00]])
        #self.Hmatrix = np.array([[2.18077578e+00,4.48045099e+00,-2.48385255e+02],[-1.88141518e-01,1.22897414e+01,-8.29750111e+02],[-4.18092263e-04,1.15984417e-02,1.00000000e+00]])
        self.Hmatrix = None
        self.sf = 5

        h_image = 150
        w_image = 200
        h_image*= self.sf
        w_image*= self.sf

        self.lista_erros = []
        self.h_image = h_image
        self.w_image = w_image

        self.last_valid_error = 0
        #self.find_Hmatrix()
        #self.cap = cv2.VideoCapture("/home/dudu/Dropbox/Programming/line_follower/3.mp4")
        
        #self.ransac = linear_model.RANSACRegressor()


    def find_Hmatrix(self,image):
        '''
        while True:
            
            _, image = self.cap.read()
            
            cv2.imshow("image",image)
            if cv2.waitKey(1) == 27: break
        '''
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


    def find_error(self, image):
        try:
            if self.Hmatrix == None: self.find_Hmatrix(image)
        except:
            pass
        
        #print "oi"
        erro=0
        #ret, image = self.cap.read()
        imagem_copia_original = image.copy()
        time_1 = time.time()
        image = cv2.warpPerspective(image, self.Hmatrix, (self.w_image, self.h_image))
        
        image = cv2.resize(image,(0,0),fx=0.5,fy=0.5)
        
        print_image = image.copy()
        p_img = image.copy()

        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        curve = np.zeros(image.shape,np.uint8)
        mean_image = image.copy()
        original = image.copy()
        gray = image.copy()
    
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
            
            for p in cnt:
                cv2.circle(p_img, (p[0][0], p[0][1]), 2, (255,0,0), -1)

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
            if largura<8*self.sf/2 and largura>3*self.sf/2 and altura>20*self.sf/2:
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

            if mean_val[0]>50: #and len(approx)>3: #and len(approx)<8:
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
                
            xp = np.linspace(0, image.shape[0], 30)    
            for x in xp:    
               cv2.circle(curve, (int(polinomio(x)),int(x)),2,(255,0,0),-1)
        except:
            pass
        
        cv2.imshow('Colin', curve)
        cv2.imshow('image', print_image)
        cv2.imshow('gray', gray)
        cv2.imshow('p_img', p_img)
        
        #image = cv2.resize(image,(0,0),fx=2,fy=2)
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
        #print 1000/((time.time()-time_1)*1000)
        return erro*1.2