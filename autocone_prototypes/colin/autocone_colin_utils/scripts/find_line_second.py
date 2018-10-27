import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def nothing(x):
    pass


def processing(frame):
    pass

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
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

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

    cv2.createTrackbar('Blur size', 'Parameters', 0, 5)
    cv2.createTrackbar('')

    frame_index = 0
    last_frame_index = -1

    while(1):
        cv2.imshow('Slider', slide_img)
        cv2.waitKey(1)

        #ret, frame = cap.read()
        frame_index = cv2.getTrackbarPos('Frame','Slider')
        new_frame = video[frame_index]

        if frame_index != last_frame_index:

            # Get a frame
            frame = cv2.resize(new_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

            # Convert to grayscale
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #image = cv2.GaussianBlur(image, (5, 5), 0)
            #cv2.imshow("Blurred", image)
            #cv2.imwrite("blurred.png", image)

            # When performing Canny edge detection we need two values
            # for hysteresis: threshold1 and threshold2. Any gradient
            # value larger than threshold2 are considered to be an
            # edge. Any value below threshold1 are considered not to
            # ben an edge. Values in between threshold1 and threshold2
            # are either classified as edges or non-edges based on how
            # the intensities are "connected". In this case, any gradient
            # values below 30 are considered non-edges whereas any value
            # above 150 are considered edges.
            canny = cv2.Canny(gray_img, 30, 150)


            last_frame_index = frame_index


        cv2.imshow('Original', gray_img)
        cv2.imshow('th4', canny)

        cv2.waitKey(1)

    
