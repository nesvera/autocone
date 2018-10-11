#!/usr/bin/env python

import numpy as np
from glob import glob
import getpass
import os
import cv2
import random

from tensorflow.python import keras
from tensorflow.python.keras.layers import (
    Input, 
    Conv2D, 
    Conv2DTranspose, 
    Lambda, 
    Reshape, 
    Flatten, 
    Dense,
    MaxPooling2D,
    UpSampling2D,
)
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import rospy
import pickle

class VAE():

    input_dim = (240, 320, 1)
    z_dim = 16

    dense_size = 32

    # number of images load of the memory to train
    minibatch_size = 500

    # number of times of loading a minibatch and train
    iterations = 500

    epochs = 1
    batch_size = 32

    # Save weights after some train iterations
    trains_btw_saves = 10

    def __init__(self):
        
        self.models = self.build_model2()
        self.model = self.models[0]
        self.encoder = self.models[1]
        self.decoder = self.models[2]

   
    def build_model2(self):
        vae_input = Input( shape=self.input_dim)
        #print("vae_input shape " + str(vae_input.shape))

        vae_c1 = Conv2D( filters=32, kernel_size=3, padding='same', activation='relu', trainable=True)(vae_input)
        vae_m1 = MaxPooling2D((2, 2), padding='same', trainable=True)(vae_c1)
        vae_c2 = Conv2D( filters=16, kernel_size=3, padding='same', activation='relu', trainable=True)(vae_m1)
        vae_m2 = MaxPooling2D((2, 2), padding='same', trainable=True)(vae_c2)
        vae_c3 = Conv2D( filters=16, kernel_size=3, padding='same', activation='relu', trainable=True)(vae_m2)
        vae_m3 = MaxPooling2D((2, 2), padding='same', trainable=True)(vae_c3)
        vae_c4 = Conv2D( filters=16, kernel_size=3, padding='same', activation='relu', trainable=True)(vae_m3)
        vae_m4 = MaxPooling2D((2, 2), padding='same', trainable=True)(vae_c4)
        vae_c5 = Conv2D( filters=16, kernel_size=3, padding='same', activation='relu', trainable=True)(vae_m4)
        vae_m5 = MaxPooling2D((2, 2), padding='same', trainable=True)(vae_c5)
        vae_c6 = Conv2D( filters=4, kernel_size=3, padding='same', activation='relu', trainable=True)(vae_m5)
        vae_m6 = MaxPooling2D((2, 2), padding='same', trainable=True)(vae_c6)


        vae_z_in = Flatten()(vae_m6)
        print("vae_z_in shape " + str(vae_z_in.shape))

        vae_z = Dense(25, trainable=True)(vae_z_in)
        print("vae_z shape " + str(vae_z.shape))

        vae_z_input = Input( shape=(25,))
        print("vae_z_input shape " + str(vae_z_input.shape))

        vae_z_out = Reshape((5, 5, 1))
        vae_z_out_model = vae_z_out(vae_z)
        print("vae_z_out_model shape " + str(vae_z_out_model.shape))

        #vae_d1 = Conv2D( filters=8, kernel_size=(3, 3), padding='same', activation='relu')
        vae_u1 = UpSampling2D((3,4))
        vae_d2 = Conv2D( filters=32, kernel_size=(3, 3), padding='same', activation='relu')
        vae_u2 = UpSampling2D((2,2))
        vae_d3 = Conv2D( filters=32, kernel_size=(3, 3), padding='same', activation='relu')
        vae_u3 = UpSampling2D((2,2))
        vae_d4 = Conv2D( filters=16, kernel_size=(3, 3), padding='same', activation='relu')
        vae_u4 = UpSampling2D((2,2))
        vae_d5 = Conv2D( filters=8, kernel_size=(3, 3), padding='same', activation='relu')
        vae_u5 = UpSampling2D((2,2))
        vae_d6 = Conv2D( filters=1, kernel_size=(3, 3), padding='same', activation='sigmoid')

        # vae_d1_model = vae_d1(vae_z_out_model)
        vae_u1_model = vae_u1(vae_z_out_model)
        vae_d2_model = vae_d2(vae_u1_model)
        vae_u2_model = vae_u2(vae_d2_model)
        vae_d3_model = vae_d3(vae_u2_model)
        vae_u3_model = vae_u3(vae_d3_model)
        vae_d4_model = vae_d4(vae_u3_model)
        vae_u4_model = vae_u4(vae_d4_model)
        vae_d5_model = vae_d5(vae_u4_model)
        vae_u5_model = vae_u5(vae_d5_model)
        vae_d6_model = vae_d6(vae_u5_model)
        #print("vae_d1_model shape " + str(vae_d1_model.shape))
        #print("vae_u1_model shape " + str(vae_u1_model.shape))
        #print("vae_d2_model shape " + str(vae_d2_model.shape))
        #print("vae_u2_model shape " + str(vae_u2_model.shape))
        #print("vae_d3_model shape " + str(vae_d3_model.shape))
        #print("vae_u3_model shape " + str(vae_u3_model.shape))
        #print("vae_d4_model shape " + str(vae_d4_model.shape))
        #print("vae_u4_model shape " + str(vae_u4_model.shape))
        #print("vae_d5_model shape " + str(vae_d5_model.shape))

        #240 120 60 30 15
        #320 160 80 40 20

        vae_dense_decoder = vae_z_input
        vae_z_out_decoder = vae_z_out(vae_dense_decoder)

        #vae_d1_decoder = vae_d1(vae_z_out_decoder)
        vae_u1_decoder = vae_u1(vae_z_out_decoder)
        vae_d2_decoder = vae_d2(vae_u1_decoder)
        vae_u2_decoder = vae_u2(vae_d2_decoder)
        vae_d3_decoder = vae_d3(vae_u2_decoder)
        vae_u3_decoder = vae_u3(vae_d3_decoder)
        vae_d4_decoder = vae_d4(vae_u3_decoder)
        vae_u4_decoder = vae_u4(vae_d4_decoder)
        vae_d5_decoder = vae_d5(vae_u4_decoder)
        vae_u5_decoder = vae_u5(vae_d5_decoder)
        vae_d6_decoder = vae_d6(vae_u5_decoder)
        print("vae_d1_decoder shape " + str(vae_u1_decoder.shape))
        print("vae_d2_decoder shape " + str(vae_d2_decoder.shape))
        print("vae_d3_decoder shape " + str(vae_d3_decoder.shape))
        print("vae_d4_decoder shape " + str(vae_d4_decoder.shape))
        print("vae_d5_decoder shape " + str(vae_d5_decoder.shape))

        # Models
        vae = Model(vae_input, vae_d6_model)
        vae_encoder = Model(vae_input, vae_z)
        vae_decoder = Model(vae_z_input, vae_d6_decoder)

        #vae.compile(optimizer='rmsprop', loss = vae_loss,  metrics = [vae_r_loss, vae_kl_loss])
        #vae.compile(optimizer='rmsprop', loss='binary_crossentropy')
        #optimizer = Adam(lr=0.001)
        vae.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy')
        vae.summary()

        return (vae, vae_encoder, vae_decoder)

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal( shape=(K.shape(z_mean)[0], self.z_dim), mean=0., stddev=1.)
        return ( z_mean + K.exp(z_log_var/2.)*epsilon)

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def freeze(self):

        for layer in self.models.layers[0:5]:
            print(layer)
	

    def generate_image(self, z_input):
        img = self.decoder.predict(z_input)

        return img

    def encode_image(self, data):
        
        z = self.encoder.predict(data)

        return z


if __name__ == "__main__":

    rospy.init_node('ae_camera_test', anonymous=True)

    cam = cv2.VideoCapture(1)


    i1 = 1
    i2 = 1
    minH = 0
    minS = 180
    minV = 100
    maxH = 255
    maxS = 255
    maxV = 255
    fSize = 5	

    username = getpass.getuser()
    vae_weight = '/home/' + username + '/Documents/autocone_ae_weights/' + "498_1000_1500.h5"

    vae = VAE()

    vae.load_weights(vae_weight)


    while False:
        ret, frame = cam.read()
        cv2.imshow("Webcam", frame)
	
	    # up and low color boundaries
        colorLower = (minH, minS, minV)
        colorUpper = (maxH, maxS, maxV)

        # is possible to define image size direct in usb_cam package
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        
        
        # processing the image
        # apply blur, converto to HSV color space, create mask based on color, erode and dilate
        blurred = cv2.GaussianBlur(frame, (fSize, fSize), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, colorLower, colorUpper)
        mask = cv2.erode(mask, None, iterations=i1)
        mask = cv2.dilate(mask, None, iterations=i2)

        img = mask.astype('float32')/255.

        cv2.imshow('input', img)

        img = np.reshape(img, (1, img.shape[0], img.shape[1], 1))                

        z = vae.encode_image(img)

        img_out = vae.generate_image(z)
        img_out = np.reshape(img_out, (img_out.shape[1], img_out.shape[2]))

        cv2.imshow('output', img_out)


        cv2.waitKey(1)



    vae_dataset_folder = '/home/' + username + '/Documents/autocone_vae_dataset/'

    # get all images files inside the folder
    dataset = glob(vae_dataset_folder + "*.jpg")

    while True:
        img_file = random.choice(dataset)
        img = cv2.imread(img_file, 0)
        print(img)

        img = img.astype('float32')/255.
        print(img)

        cv2.imshow('input', img)

        img = np.reshape(img, (1, img.shape[0], img.shape[1], 1))                

        z = vae.encode_image(img)
        print(z)

        img_out = vae.generate_image(z)
        img_out = np.reshape(img_out, (img_out.shape[1], img_out.shape[2]))

        cv2.imshow('output', img_out)
        cv2.waitKey(0)
    


