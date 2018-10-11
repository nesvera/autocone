import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import getpass
from glob import glob
import cv2

from keras import backend as K

from keras.layers import (
    Input, 
    Dense, 
    Lambda, 
    Layer, 
    Add, 
    Multiply, 
    Conv2D,
    Flatten,
    Reshape,
    UpSampling2D
)
from keras.models import Model, Sequential
from keras.datasets import mnist


original_dim = (240, 320, 1)
intermediate_dim = 256
latent_dim = 300
batch_size = 100
epochs = 50
epsilon_std = 1.0

def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs

encoder = Sequential([
    Conv2D(input_shape=original_dim,
           filters=32, 
           kernel_size=3, 
           padding='same',
           strides=(2, 2),
           activation='relu'),
    Conv2D(filters=32, 
           kernel_size=3, 
           padding='same',
           strides=(2, 2),
           activation='relu'),
    Conv2D(filters=32, 
           kernel_size=3, 
           padding='same',
           strides=(2, 2), 
           activation='relu'),
    Conv2D(filters=32, 
           kernel_size=3, 
           padding='same',
           strides=(2, 2), 
           activation='relu')
    Conv2D(filters=32, 
           kernel_size=3, 
           padding='same',
           strides=(2, 2), 
           activation='relu')
])

decoder = Sequential([
    Conv2D(input_shape=(25,25,1),
            filters=128, 
            kernel_size=(3, 3), 
            padding='same', 
            activation='relu'),
    UpSampling2D((2,2)),
    Conv2D(filters=64, 
            kernel_size=(3, 3), 
            padding='same', 
            activation='relu'),
    UpSampling2D((2,2)),
    Conv2D(filters=32, 
            kernel_size=(3, 3), 
            padding='same', 
            activation='relu'),
    UpSampling2D((2,2)),
    Conv2D(filters=16, 
            kernel_size=(3, 3), 
            padding='same', 
            activation='relu'),
    UpSampling2D((2,2)),
    Conv2D(filters=1, 
            kernel_size=(3, 3), 
            padding='same', 
            activation='sigmoid')
])

x = Input(shape=original_dim)
h = encoder(x)

print(h.shape)

h_f = Flatten()(h)

z_mu = Dense(latent_dim)(h_f)
z_log_var = Dense(latent_dim)(h_f)

z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

eps = Input(tensor=K.random_normal(stddev=epsilon_std,
                                   shape=(K.shape(x)[0], latent_dim)))
z_eps = Multiply()([z_sigma, eps])
z = Add()([z_mu, z_eps])

z_r = Reshape((15, 20, 1))(z)
x_pred = decoder(z_r)

vae = Model(inputs=[x, eps], outputs=x_pred)
vae.compile(optimizer='rmsprop', loss=nll)

vae.summary()

username = getpass.getuser()
vae_dataset_folder = '/home/' + username + '/Documents/autocone_vae_dataset/'
vae_weights_folder = '/home/' + username + '/Documents/autocone_vae_weights/'

# get all images files inside the folder
dataset = glob(vae_dataset_folder + "*.jpg")
dataset_total = len(dataset)

minibatch_begin = 0
minibatch_end = 500

n_trains = 0

i = 0

while i < 1:

    # load minibatch
    minibatch = dataset[minibatch_begin: minibatch_end]

    data = np.zeros((500, 240, 320, 1))
    for j, img_file in enumerate(minibatch):
        img = cv2.imread(img_file, 0)
        img = img.astype('float32')/255.
        #img = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
        
        data[j, :, :, 0] = img

    vae.fit( x=data, y=data,
            shuffle=True,
            epochs=1,
            batch_size=8)

    i += 1

encoder = Model(x, z_mu)

z = np.zeros((1, 300))
while True:

    for i in range(300):
        z[0, i] = np.random.normal(0, 1)

    z = z.reshape((1,15,20,1))


    x_decoded = decoder.predict(z)
    x_decoded = x_decoded.reshape(240, 360)

    print(np.amin(x_decoded))
    print(np.amax(x_decoded))

    cv2.imshow('carai', x_decoded)
    cv2.waitKey(0)