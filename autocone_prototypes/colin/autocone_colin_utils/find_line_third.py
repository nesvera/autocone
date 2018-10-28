import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def nothing(x):
    pass
    

def bgr2hsi(frame):
    B, G, R = cv2.split(frame)
    R = np.float32(R)
    G = np.float32(G)
    B = np.float32(B)

    r = np.zeros(R.shape)
    g = np.zeros(G.shape)
    b = np.zeros(B.shape)

    H = np.zeros(B.shape)
    S = np.zeros(B.shape)
    I = np.zeros(B.shape)

    print(r.shape)

    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            r[i,j] = R[i,j] / max((R[i,j] + G[i,j] + B[i,j]), 0.01)
            g[i,j] = G[i,j] / max((R[i,j] + G[i,j] + B[i,j]), 0.01)
            b[i,j] = B[i,j] / max((R[i,j] + G[i,j] + B[i,j]), 0.01)

            S[i,j] = 1 - 3*min(r[i,j], g[i,j], b[i,j])

            I[i,j] = (R[i,j] + G[i,j] + B[i,j])/(3*255)

    return R, S, I

if __name__ == "__main__":

    if(len(sys.argv) < 2):
        print("python " + sys.argv[0] + " video_path")
        exit(0)

    video_path = sys.argv[1]

    if(os.path.isfile(video_path) == False):
        print("File " + video_path + "not found!")
        exit(0)

    cap = cv2.VideoCapture(video_path)
    
    while False:

        # Get a frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #blurred = cv2.GaussianBlur(frame, (fSize, fSize), 0)
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)

        
        ret,th = cv2.threshold(blurred, 150, 220, cv2.THRESH_BINARY)
        th_adp1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        th_adp2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        ret3,th_adp3 = cv2.threshold(blurred,200 ,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        cv2.imshow('Original', gray_img)
        #cv2.imshow('Blurred', blurred)
        #cv2.imshow('Threshold', th)
        #cv2.imshow('Threshold mean', th_adp1)
        #cv2.imshow('Threshold gaussian', th_adp2)
        cv2.imshow('Threshold otsu', th_adp3)

        plt.hist(gray_img.ravel(),256,[0,256]); plt.show(); plt.pause(0.001)

        cv2.waitKey(1)


    video = list()
    ret, frame = cap.read()
    while ret == True:
        video.append(frame)
        ret, frame = cap.read()
    
    slide_img = np.zeros((1,1,1), np.uint8)
    cv2.namedWindow('Slider')
    cv2.namedWindow('Parameters')
    
    # create trackbars for frame selection
    cv2.createTrackbar('Frame','Slider',0, len(video),nothing)
    cv2.createTrackbar('Alpha','Slider',0, 255,nothing)
    cv2.createTrackbar('Beta','Slider',0, 255,nothing)

    frame_index = 0
    last_frame_index = -1

    while(1):
        cv2.imshow('Slider', slide_img)
        cv2.imshow('Parameters', slide_img)
        cv2.waitKey(1)

        #ret, frame = cap.read()
        frame_index = cv2.getTrackbarPos('Frame','Slider')
        new_frame = video[frame_index]

        alpha = cv2.getTrackbarPos('Alpha','Slider')
        beta = cv2.getTrackbarPos('Beta','Slider')
 
        if frame_index != last_frame_index:

            # Get a frame
            frame = cv2.resize(new_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

            # Convert to hsi
            #hsi_h, hsi_s, hsi_i = bgr2hsi(frame)
            hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
            hls_h, hls_l, hls_s = cv2.split(hls)


            binary = np.zeros(hls_l.shape)

            for i in range(hls_l.shape[0]):
                for j in range(hls_l.shape[1]):
                    
                    if hls_s[i, j] < alpha and hls_l[i, j] > beta:
                        binary[i, j] = 1

                    else:
                        binary[i, j] = 0


            last_frame_index = frame_index

        cv2.imshow('hsi_s', hls_s)
        cv2.imshow('hsi_i', hls_l)
        cv2.imshow('binary', binary)
        cv2.waitKey(1)

    
