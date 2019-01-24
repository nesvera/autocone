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

class Charuco:

    def __init__(self,x,y,tw, ta,txlen, vec=[0,0], dictionary = aruco.DICT_6X6_250):

        self.dimension = [txlen, y*txlen/x]

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.x, self.y = x,y

        self.step = self.dimension[0] / self.x

        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary)
        
        #self.dictionary = aruco.Dictionary_get(dictionary)

        self.board = cv2.aruco.CharucoBoard_create(x,y,tw, ta,self.dictionary)

        self.parameters = aruco.DetectorParameters_create()

        self.vec = vec

    def feed(self, frame):

        self.frame = frame

        if len(frame.shape)==3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else: gray = frame

        self.aruco_corner, self.aruco_id, _  = cv2.aruco.detectMarkers(gray,self.dictionary)
        
        if len(self.aruco_corner)>0:

            

            self.n,self.ch_corner, self.ch_id = cv2.aruco.interpolateCornersCharuco(self.aruco_corner, self.aruco_id,gray,self.board)
            if self.n>0:
                self.ch_corner = cv2.cornerSubPix(gray,self.ch_corner ,(11,11),(-1,-1),self.criteria)

            cv2.aruco.drawDetectedCornersCharuco(gray, self.ch_corner, self.ch_id)

            self.estimate_pose()

            if self.n>6:

                return True
        else: self.n=0


        return False

    def estimate_pose(self):

        _,self.rvec, self.tvec = cv2.aruco.estimatePoseCharucoBoard(self.ch_corner,self.ch_id,self.board,self.vec[0], self.vec[1])

    def get_position(self):

        return [self.rvec, self.tvec]

    def get_charuco(self):

        if self.n==0: return [0,0,0]

        return [self.n,self.ch_corner, self.ch_id]

    def draw(self):

        return cv2.aruco.drawAxis(self.frame, self.vec[0],self.vec[1], self.rvec, self.tvec, 0.1)

    def compare(self, ids1, ids2):
        lista = []

        for ids in ids1:
            if ids in ids2: lista.append(ids)

        return lista

    def refine(self, idswanted, idstotal, corners):

        index2delete = []

        for i,ids in enumerate(idstotal):
            if ids not in idswanted:
                index2delete.append(i)

        for i, n in enumerate(index2delete):
            index = n-i
            corners = np.delete(corners, index, axis=1)
            

        return corners



    def list_calibration(self, totalids):

        lista = []

        maximo = self.x - 1

        for ids in totalids:

            linha = (ids - ids%maximo)/maximo

            coluna = ids - linha*maximo

            linha+=1

            coluna+=1

            pos = [(coluna*self.step)[0], (linha*self.step)[0],0]

            lista.append(pos)

        pattern_points = np.array(lista, dtype = np.float32).reshape(1,-1,3)

        

        return pattern_points