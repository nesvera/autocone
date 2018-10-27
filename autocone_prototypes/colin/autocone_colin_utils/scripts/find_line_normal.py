import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def nothing(x):
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
    
    # create trackbars for frame selection
    cv2.createTrackbar('Frame','Slider',0, len(video),nothing)

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

            kernel = np.array(([[0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 1, 10, 1, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0]]))
            kernel = kernel/50.
        
            gray_img = cv2.filter2D(gray_img,-1,kernel)


            # Compute the Laplacian of the image
            lap = cv2.Laplacian(gray_img, cv2.CV_64F)
            lap = np.uint8(np.absolute(lap))
            

            #blurred = cv2.GaussianBlur(frame, (fSize, fSize), 0)
            #median = cv2.GaussianBlur(gray_img, (3, 3), 0)
            #median = cv2.medianBlur(gray_img, 9)
            #bilat = cv2.bilateralFilter(gray_img, 9, 75, 75)
            
            #ret,th1 = cv2.threshold(median.copy(), 80, 100, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            #ret,th2 = cv2.threshold(median.copy(), 180, 255, cv2.THRESH_BINARY)
            #ret,th4 = cv2.threshold(median.copy(), 200, 255, cv2.THRESH_BINARY)
            #ret,th5 = cv2.threshold(median.copy(), 210, 255, cv2.THRESH_BINARY)

            #ret,th = cv2.threshold(openning, 180, 220, cv2.THRESH_BINARY)
            #th_adp1 = cv2.adaptiveThreshold(median, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
            #th_adp2 = cv2.adaptiveThreshold(openning, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,5,2)
            #ret3,th_adp3 = cv2.threshold(openning,0 ,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            # Compute the Laplacian of the image
            lap = cv2.Laplacian(gray_img, cv2.CV_64F)
            lap = np.uint8(np.absolute(lap))

            # Compute gradients along the X and Y axis, respectively
            sobelX = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
            sobelY = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)

            # The sobelX and sobelY images are now of the floating
            # point data type -- we need to take care when converting
            # back to an 8-bit unsigned integer that we do not miss
            # any images due to clipping values outside the range
            # of [0, 255]. First, we take the absolute value of the
            # graident magnitude images, THEN we convert them back
            # to 8-bit unsigned integers
            sobelX = np.uint8(np.absolute(sobelX))
            sobelY = np.uint8(np.absolute(sobelY))

            # We can combine our Sobel gradient images using our
            # bitwise OR
            sobelCombined = cv2.bitwise_or(sobelX, sobelY)

            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(sobelCombined, cv2.MORPH_CLOSE, kernel)

            canny = cv2.Canny(sobelCombined, 30, 200)

            


            last_frame_index = frame_index

        # Show our Sobel images
        cv2.imshow("Sobel X", sobelX)
        cv2.imshow("Sobel Y", sobelY)
        cv2.imshow("Sobel Combined", sobelCombined)

        cv2.imshow("Laplacian", lap)
        cv2.imshow('Original', gray_img)
        #cv2.imshow('Median', median)
        #cv2.imshow('Threshold', th)
        #cv2.imshow('Threshold mean', th_adp1)
        #cv2.imshow('Threshold gaussian', th_adp2)
        #cv2.imshow('Threshold otsu', th_adp3)
        cv2.imshow('Openning', opening)
        cv2.imshow('canny', canny)
        #cv2.imshow('th1', th1)
        #cv2.imshow('th2', th2)
        #cv2.imshow('th4', th4)
        #cv2.imshow('th5', th5)

        cv2.waitKey(1)

    
