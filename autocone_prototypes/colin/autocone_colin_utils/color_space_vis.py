import cv2
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

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
    ret, frame = cap.read()

    video = list()
    while ret == True:
        video.append(frame)
        ret, frame = cap.read()

    plt.ion() ## Note this correction
    fig1=plt.figure()
    fig2=plt.figure()
    
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
        frame = video[frame_index]

        if frame_index != last_frame_index:

            frame = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)

            f = plt.figure(1)

            #plt.subplot(332), plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), plt.title('Original')
            #cv2.imshow('Original', frame)

            # RGB
            #rgb_b, rgb_g, rgb_r = cv2.split(frame.copy())
            #plt.subplot(331), plt.imshow(rgb_r, cmap="Greys"), plt.title('rgb_r'), plt.axis('off')
            #plt.subplot(332), plt.imshow(rgb_g, cmap="Greys"), plt.title('rgb_g'), plt.axis('off')
            #plt.subplot(333), plt.imshow(rgb_b, cmap="Greys"), plt.title('rgb_b'), plt.axis('off')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            plt.subplot(332), plt.imshow(gray), plt.title('grayscale'), plt.axis('off')

            # LAB
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab_l, lab_a, lab_b = cv2.split(lab)
            plt.subplot(334), plt.imshow(lab_l), plt.title('lab_l'), plt.axis('off')
            plt.subplot(335), plt.imshow(lab_a), plt.title('lab_a'), plt.axis('off')
            plt.subplot(336), plt.imshow(lab_b), plt.title('lab_b'), plt.axis('off')
            
            #  YCB
            ycb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            ycb_y, ycb_c, ycb_b = cv2.split(ycb)
            plt.subplot(337), plt.imshow(ycb_y), plt.title('ycb_y'), plt.axis('off')
            plt.subplot(338), plt.imshow(ycb_c), plt.title('ycb_c'), plt.axis('off')
            plt.subplot(339), plt.imshow(ycb_b), plt.title('ycb_b'), plt.axis('off')

            f.show()

            g = plt.figure(2)

            # HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_h, hsv_s, hsv_v = cv2.split(hsv)
            #plt.subplot(331), plt.imshow(hsv_h), plt.title('hsv_h'), plt.axis('off')
            #plt.subplot(332), plt.imshow(hsv_s), plt.title('hsv_s'), plt.axis('off')
            plt.subplot(333), plt.imshow(hsv_v), plt.title('hsv_v'), plt.axis('off')

            # HLS
            hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
            hls_h, hls_l, hls_s = cv2.split(hls)
            #plt.subplot(334), plt.imshow(hls_h), plt.title('hls_h'), plt.axis('off')
            plt.subplot(335), plt.imshow(hls_l), plt.title('hls_l'), plt.axis('off')
            #plt.subplot(336), plt.imshow(hls_s), plt.title('hls_s'), plt.axis('off')

            # LUV
            luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
            luv_l, luv_u, luv_v = cv2.split(luv)
            plt.subplot(337), plt.imshow(luv_l), plt.title('luv_l'), plt.axis('off')
            plt.subplot(338), plt.imshow(luv_u), plt.title('luv_u'), plt.axis('off')
            plt.subplot(339), plt.imshow(luv_v), plt.title('luv_v'), plt.axis('off')
            
            g.show()

            #h = plt.figure(3)

            # XYZ
            #xyz = cv2.cvtColor(frame, cv2.COLOR_BGR2XYZ)
            #xyz_x, xyz_y, xyz_z = cv2.split(xyz)
            #plt.subplot(331), plt.imshow(xyz_x), plt.title('xyz_x'), plt.axis('off')
            #plt.subplot(332), plt.imshow(xyz_y), plt.title('xyz_y'), plt.axis('off')
            #plt.subplot(333), plt.imshow(xyz_z), plt.title('xyz_z'), plt.axis('off')

            # YUV
            #yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            #yuv_y, yuv_u, yuv_v = cv2.split(yuv)
            #plt.subplot(334), plt.imshow(yuv_y), plt.title('yuv_y'), plt.axis('off')
            #plt.subplot(335), plt.imshow(yuv_u), plt.title('yuv_u'), plt.axis('off')
            #plt.subplot(336), plt.imshow(yuv_v), plt.title('yuv_v'), plt.axis('off')

            #h.show()            
            
            plt.pause(0.001)

            last_frame_index = frame_index
    
        #if cv2.waitKey(0) & 0xFF == ord('q') or ret==False :
        #    cap.release()
        #    cv2.destroyAllWindows()
        #    break