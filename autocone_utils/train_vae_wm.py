import numpy as np
from glob import glob
import getpass
import os
import cv2
import random

from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping

INPUT_DIM = (240,320,1)

CONV_FILTERS = [32,64,64,128]
CONV_KERNEL_SIZES = [4,4,4,4]
CONV_STRIDES = [2,2,2,2]
CONV_ACTIVATIONS = ['relu','relu','relu','relu']

DENSE_SIZE = 1024

CONV_T_FILTERS = [64,64,32,1]
CONV_T_KERNEL_SIZES = [(6,8),(5,6),4,4]
CONV_T_STRIDES = [2,2,4,4]
CONV_T_ACTIVATIONS = ['relu','relu','relu','sigmoid']

#240   120   60   30   15   8    4    
#320   160   80   40   20   10   5

Z_DIM = 32

EPOCHS = 1
BATCH_SIZE = 32

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], Z_DIM), mean=0.,stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

class VAE():

    # number of images load of the memory to train
    minibatch_size = 500

    # number of times of loading a minibatch and train
    iterations = 500

    epochs = 1
    batch_size = 32

    # Save weights after some train iterations
    trains_btw_saves = 10


    def __init__(self):
        self.models = self._build()
        self.model = self.models[0]
        self.encoder = self.models[1]
        self.decoder = self.models[2]

        self.input_dim = INPUT_DIM
        self.z_dim = Z_DIM

    def _build(self):
        vae_x = Input(shape=INPUT_DIM)
        vae_c1 = Conv2D(filters = CONV_FILTERS[0], kernel_size = CONV_KERNEL_SIZES[0], strides = CONV_STRIDES[0], activation=CONV_ACTIVATIONS[0])(vae_x)
        vae_c2 = Conv2D(filters = CONV_FILTERS[1], kernel_size = CONV_KERNEL_SIZES[1], strides = CONV_STRIDES[1], activation=CONV_ACTIVATIONS[0])(vae_c1)
        vae_c3= Conv2D(filters = CONV_FILTERS[2], kernel_size = CONV_KERNEL_SIZES[2], strides = CONV_STRIDES[2], activation=CONV_ACTIVATIONS[0])(vae_c2)
        vae_c4= Conv2D(filters = CONV_FILTERS[3], kernel_size = CONV_KERNEL_SIZES[3], strides = CONV_STRIDES[3], activation=CONV_ACTIVATIONS[0])(vae_c3)

        vae_z_in = Flatten()(vae_c4)

        vae_z_mean = Dense(Z_DIM)(vae_z_in)
        vae_z_log_var = Dense(Z_DIM)(vae_z_in)

        vae_z = Lambda(sampling)([vae_z_mean, vae_z_log_var])
        vae_z_input = Input(shape=(Z_DIM,))

        # we instantiate these layers separately so as to reuse them later
        vae_dense = Dense(1024)
        vae_dense_model = vae_dense(vae_z)

        vae_z_out = Reshape((1,1,DENSE_SIZE))
        vae_z_out_model = vae_z_out(vae_dense_model)

        vae_d1 = Conv2DTranspose(filters = CONV_T_FILTERS[0], kernel_size = CONV_T_KERNEL_SIZES[0] , strides = CONV_T_STRIDES[0], activation=CONV_T_ACTIVATIONS[0])
        vae_d1_model = vae_d1(vae_z_out_model)
        vae_d2 = Conv2DTranspose(filters = CONV_T_FILTERS[1], kernel_size = CONV_T_KERNEL_SIZES[1] , strides = CONV_T_STRIDES[1], activation=CONV_T_ACTIVATIONS[1])
        vae_d2_model = vae_d2(vae_d1_model)
        vae_d3 = Conv2DTranspose(filters = CONV_T_FILTERS[2], kernel_size = CONV_T_KERNEL_SIZES[2] , strides = CONV_T_STRIDES[2], activation=CONV_T_ACTIVATIONS[2])
        vae_d3_model = vae_d3(vae_d2_model)
        vae_d4 = Conv2DTranspose(filters = CONV_T_FILTERS[3], kernel_size = CONV_T_KERNEL_SIZES[3] , strides = CONV_T_STRIDES[3], activation=CONV_T_ACTIVATIONS[3])
        vae_d4_model = vae_d4(vae_d3_model)

        #### DECODER ONLY

        vae_dense_decoder = vae_dense(vae_z_input)
        vae_z_out_decoder = vae_z_out(vae_dense_decoder)

        vae_d1_decoder = vae_d1(vae_z_out_decoder)
        vae_d2_decoder = vae_d2(vae_d1_decoder)
        vae_d3_decoder = vae_d3(vae_d2_decoder)
        vae_d4_decoder = vae_d4(vae_d3_decoder)

        #### MODELS

        vae = Model(vae_x, vae_d4_model)
        vae_encoder = Model(vae_x, vae_z)
        vae_decoder = Model(vae_z_input, vae_d4_decoder)

        def vae_r_loss(y_true, y_pred):

            y_true_flat = K.flatten(y_true)
            y_pred_flat = K.flatten(y_pred)

            return 10 * K.mean(K.square(y_true_flat - y_pred_flat), axis = -1)

        def vae_kl_loss(y_true, y_pred):
            return - 0.5 * K.mean(1 + vae_z_log_var - K.square(vae_z_mean) - K.exp(vae_z_log_var), axis = -1)

        def vae_loss(y_true, y_pred):
            return vae_r_loss(y_true, y_pred) + vae_kl_loss(y_true, y_pred)
            
        vae.compile(optimizer='rmsprop', loss = vae_loss,  metrics = [vae_r_loss, vae_kl_loss])
        vae.summary()

        return (vae,vae_encoder, vae_decoder)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self):

        username = getpass.getuser()
        vae_dataset_folder = '/home/' + username + '/Documents/autocone_vae_dataset/'
        vae_weights_folder = '/home/' + username + '/Documents/autocone_vae_weights/'

        # get all images files inside the folder
        dataset = glob(vae_dataset_folder + "*.jpg")
        dataset_total = len(dataset)

        minibatch_begin = 0
        minibatch_end = self.minibatch_size

        n_trains = 0

        i = 0
        while i < self.iterations:

            print("Iteration " + str(i) + " of " + str(self.iterations))
            print("Data from " + str(minibatch_begin) + " to " + str(minibatch_end) + " of Total: " + str(dataset_total))
            print("")

            # load minibatch
            minibatch = dataset[minibatch_begin: minibatch_end]

            data = np.zeros((self.minibatch_size, self.input_dim[0], self.input_dim[1], self.input_dim[2]))
            for j, img_file in enumerate(minibatch):
                img = cv2.imread(img_file, 0)
                img = img.astype('float32')/255.
                #img = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
                
                data[j, :, :, 0] = img

            #for f in range(len(minibatch)):
            #    imagem = data[f, :, :, 0]
            #    cv2.imshow('image', imagem)
            #    cv2.waitKey(0)

            #self.model.fit( x=data, y=data, shuffle=True, epochs=self.epochs, batch_size=self.batch_size)

            earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
            callbacks_list = [earlystop]

            self.model.fit( x=data, y=data,
                    shuffle=True,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    callbacks=callbacks_list)

            # save weights after some trains sections
            if n_trains % self.trains_btw_saves == 0:
                filename = vae_weights_folder + str(i) + "_" + str(minibatch_begin) + "_" + str(minibatch_end) + ".h5"
                print("Saving " + filename)
                self.save_weights(filename)

            n_trains += 1

            # increase indexes of dataset
            minibatch_begin += self.minibatch_size
            minibatch_end += self.minibatch_size

            # after pass over all dataset, go for other iteration
            if minibatch_begin >= len(dataset):
                minibatch_begin = 0
                minibatch_end = self.minibatch_size

                i += 1

            elif minibatch_end >= len(dataset):
                minibatch_end = len(dataset)

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def generate_image(self, z_input):
        img = self.decoder.predict(z_input)

        return img

    def encode_image(self, data):
        
        z = self.encoder.predict(data)

        return z

    def vae_predict(self, data):

        img = self.model.predict(data)

        return img


if __name__ == "__main__":

    username = getpass.getuser()
    vae_weight = '/home/' + username + '/Documents/autocone_vae_weights/' + "84_1000_1500.h5"

    vae = VAE()
    #vae.train()

    vae.load_weights(vae_weight)


    #vae.train()

    vae_dataset_folder = '/home/' + username + '/Documents/autocone_vae_dataset/'

    # get all images files inside the folder
    dataset = glob(vae_dataset_folder + "*.jpg")

    
    while False:
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
    
    
    
    z = np.zeros((1, 32))

    while True:

        for i in range(32):
            z[0, i] = random.gauss(0, 1)
            #z[0, i] = random.uniform(-2, 2)
       
        img_out = vae.generate_image(z)


        img_out = np.reshape(img_out, (240, 320))
        #img_out = img_out.astype('float32')*50.
        print(np.amin(img_out))
        print(np.amax(img_out))
        
        cv2.imshow('carai', img_out)
        cv2.waitKey(0)
