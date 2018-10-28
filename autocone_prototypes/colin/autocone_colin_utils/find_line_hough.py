import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def nothing(x):
    pass

def draw_lines(frame, lines):

    img = frame.copy()

    try:    
        for line in lines:

            coord = line[0]
            img = cv2.line(img, (coord[0], coord[1]), (coord[2], coord[3]), [0, 255, 0], 3)

    except:
        pass

    return img


if __name__ == "__main__":

    if(len(sys.argv) < 2):
        print("python " + sys.argv[0] + " video_path")
        exit(0)

    video_path = sys.argv[1]

    if(os.path.isfile(video_path) == False):
        print("File " + video_path + "not found!")
        exit(0)

    cap = cv2.VideoCapture(video_path)

    video = list()
    ret, frame = cap.read()
    while ret == True:
        video.append(frame)
        ret, frame = cap.read()
    
    slide_img = np.zeros((1,1,1), np.uint8)
    cv2.namedWindow('Parameters')
    
    # create trackbars for frame selection
    cv2.createTrackbar('Frame','Parameters',0, len(video),nothing)
    cv2.createTrackbar('Canny_thr1','Parameters',0, 300,nothing)
    cv2.createTrackbar('Canny_thr2','Parameters',0, 300,nothing)
    cv2.createTrackbar('minLineLength','Parameters',0, 1000,nothing)
    cv2.createTrackbar('maxLineGap','Parameters',0, 100,nothing)
    cv2.createTrackbar('thresh_low','Parameters',1, 20,nothing)
    cv2.createTrackbar('thresh_high','Parameters',1, 20,nothing)


    frame_index = 0
    last_frame_index = -1

    while(1):

        #ret, frame = cap.read()
        frame_index = cv2.getTrackbarPos('Frame','Parameters')
        new_frame = video[frame_index]

        alpha = cv2.getTrackbarPos('Alpha','Parameters')
        beta = cv2.getTrackbarPos('Beta','Parameters')
        canny_thr1 = cv2.getTrackbarPos('Canny_thr1','Parameters')
        canny_thr2 = cv2.getTrackbarPos('Canny_thr2','Parameters')
        minLineLength = cv2.getTrackbarPos('minLineLength','Parameters')
        maxLineGap = cv2.getTrackbarPos('maxLineGap','Parameters')
        thresh_low = (cv2.getTrackbarPos('thresh_low','Parameters')*2)+1
        thresh_high = (cv2.getTrackbarPos('thresh_high','Parameters')*2)+1

        if frame_index != last_frame_index:

            # Get a frame
            frame = cv2.resize(new_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            #hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
            #hls_h, hls_l, hls_s = cv2.split(hls)
#
            #gray = hls_l.copy()


            last_frame_index = frame_index
        

        img = frame.copy()
        adp = cv2.adaptiveThreshold(gray.copy(), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, thresh_low, thresh_high)
        th = cv2.Canny(adp, threshold1=canny_thr1, threshold2=canny_thr2)
        #ret,th = cv2.threshold(gray.copy(), thresh_low, thresh_high, cv2.THRESH_BINARY)
        #th = cv2.inRange(gray.copy(), thresh_low, thresh_high)

        #lines = cv2.HoughLinesP(canny, 1, np.pi/180., 180, minLineLength, maxLineGap)
        _, cnts, hierarchy = cv2.findContours(th.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # find the largest contour in the mask, then use it
        # to compute the minimum enclosing circle and centroid
        if len(cnts) > 0:
            
            # get the contour with the larger area
            for c in cnts:

                # compute the bounding box of the contour
                epsilon = 0.1*cv2.arcLength(c, False)
                approx = cv2.approxPolyDP(c, epsilon, False)

                #area = cv2.contourArea(approx)
                area = cv2.arcLength(c, False)

                print(len(approx))

                if area > minLineLength and len(approx) < maxLineGap:
                    for p in approx:

                        p = p[0]
                        cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
        
        

        cv2.imshow('Gray', gray)
        cv2.imshow('adp', adp)
        cv2.imshow('Canny', th)
        cv2.imshow('frame', img)
        cv2.waitKey(1)

    
