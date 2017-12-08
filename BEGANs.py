
# coding: utf-8

# In[43]:
import keras
from keras.preprocessing import image
from keras.datasets import cifar10
from keras.optimizers import Adam
import numpy as np
import sys 
import keras.backend as K
from keras.models import Model
from keras.layers import Input,Multiply
from keras.layers.core import Flatten, Dense, Reshape
from keras.layers.convolutional import Convolution2D, UpSampling2D,Conv2DTranspose


import trainer
import utils
import sporco
import cv2
# In[44]:


epochs = 5
batches_per_epoch = 150
batch_size = 16
gamma = .5 #between 0 and 1y


# In[45]:

img_size = 32 
channels = 3 

z = 100 
h = 128 



# epochs = 1
# batches_per_epoch = 2
# batch_size = 2
# gamma = .5 #between 0 and 1y


# # In[45]:

# img_size = 32 
# channels = 3 

# z = 100 
# h = 128 
adam = Adam(lr=0.00005) 
tf_session = K.get_session()

# In[46]:

def shape(depth, row, col):
    if(K.image_dim_ordering()=='th'):
        return (depth, row, col)
    else:
        return (row, col, depth)


# In[47]:

def l1(y_true,y_pred):
    # print y_true
    return K.mean(K.abs(y_true-y_pred))

prewittFilter = K.variable([[[[-1.,  -1.]], [[0.,  -1.]],[[1.,  -1.]]],
                      [[[-1.,  0.]], [[0.,  0.]],[[1.,  0.]]],
                      [[[-1., 1.]], [[0., 1.]],[[1., 1.]]]])
def expandedprewitt(inputTensor):

    inputChannels = K.reshape(K.ones_like(inputTensor[0,0,0,:]),(1,1,-1,1))
    return prewittFilter * inputChannels

sobelFilter = K.variable([[[[-1.,  -1.]], [[0.,  -1.]],[[1.,  -1.]]],
                      [[[-1.,  0.]], [[0.,  0.]],[[1.,  0.]]],
                      [[[-1., 1.]], [[0., 1.]],[[1., 1.]]]])
def expandedsobel(inputTensor):

    inputChannels = K.reshape(K.ones_like(inputTensor[0,0,0,:]),(1,1,-1,1))
    return sobelFilter * inputChannels

# def cqsLoss(yTrue,yPred):


def gmsmLoss(yTrue,yPred):
    # print yTrue[:,:,:,2]
    # print yTrue[:,:,:,1]
    # print yTrue[:,:,:,0]



    filt = expandedprewitt(yTrue)

    prewittTrue = K.depthwise_conv2d(yTrue,filt)
    prewittPred = K.depthwise_conv2d(yPred,filt)
    filter_index=0
    
    C=0.02
    return (2*K.mean(prewittTrue)*K.mean(prewittPred)+C)/(K.mean(prewittTrue)+K.mean(prewittPred)+C)
    # return K.mean(K.square(prewittTrue - prewittPred))

def hyb(yTrue,yPred):
    a=gmsmLoss(yTrue,yPred)
    b=l1(yTrue,yPred)
    return (a+b)/2
# In[48]:

def decoder(h,img_dim,channels,n=128):
    init_dim=8
    n_layers=int(np.log2(img_dim)-3)
    in_input=Input(shape=(h,))
    x=Dense(n*init_dim*init_dim)(in_input)
    x=Reshape(shape(n,init_dim,init_dim))(x)
    x=Convolution2D(n,3,3,activation="elu",border_mode="same")(x)
    x=Convolution2D(n,3,3,activation="elu",border_mode="same")(x)
    for i in xrange(n_layers):
        x=UpSampling2D(size=(2,2))(x)
        x=Convolution2D(n,3,3,activation="elu",border_mode="same")(x)
        x=Convolution2D(n,3,3,activation="elu",border_mode="same")(x)
    x=Convolution2D(channels,3,3,activation="elu",border_mode="same")(x)
    return Model(in_input,x)


# In[49]:

def encoder(h,img_dim,channels,n=128):
    init_dim=8
    n_layers=int(np.log2(img_dim)-3+1)
    in_input=Input(shape=shape(channels,img_dim,img_dim))
    x=Convolution2D(n,3,3,activation="elu",border_mode="same")(in_input)
    for i in xrange(1,n_layers):
        x=Convolution2D(i*n,3,3,activation="elu",border_mode="same")(x)
        x=Convolution2D(i*n,3,3,activation="elu",border_mode="same",subsample=(2,2))(x)
    x=Convolution2D(n_layers*n,3,3,activation="elu",border_mode="same")(x)
    x=Convolution2D(n_layers*n,3,3,activation="elu",border_mode="same")(x)

    x=Reshape((n_layers*n*init_dim**2,))(x)
    x=Dense(h)(x)
    return Model(in_input,x)


# In[50]:

def autoencoder(h,img_dim,channels,n=128):
    in_input=Input(shape=shape(channels,img_dim,img_dim))
    x=encoder(h,img_dim,channels,n)(in_input)
    x=decoder(h,img_dim,channels,n)(x)
    return Model(in_input,x)

def GAN(gen,discrim):
    in_input=gen.input
    x=gen(in_input)
    x=discrim(x)
    return Model(in_input,x)


# In[51]:

generator = decoder(z, img_size, channels)
discriminator = autoencoder(h, img_size, channels)
GAN = GAN(generator, discriminator)

# generator1 = decoder(z, img_size, channels)
# discriminator1 = autoencoder(h, img_size, channels)
# GAN1 = GAN(generator1, discriminator1)

# generator2 = decoder(z, img_size, channels)
# discriminator2 = autoencoder(h, img_size, channels)
# GAN2 = GAN(generator2, discriminator2)




# In[52]:
l=sys.argv[1]

if l=="l1":    
    generator.compile(loss=l1, optimizer=adam)
    discriminator.compile(loss=l1, optimizer=adam)
    GAN.compile(loss=l1, optimizer=adam)

if l=="gmsmLoss":    
    generator.compile(loss=gmsmLoss, optimizer=adam)
    discriminator.compile(loss=gmsmLoss, optimizer=adam)
    GAN.compile(loss=gmsmLoss, optimizer=adam)

if l=="hyb":    
    generator.compile(loss=hyb, optimizer=adam)
    discriminator.compile(loss=hyb, optimizer=adam)
    GAN.compile(loss=hyb, optimizer=adam)



# generator.compile(loss=l1, optimizer=adam)
# discriminator.compile(loss=l1, optimizer=adam)
# GAN.compile(loss=l1, optimizer=adam)

# generator.compile(loss=hyb, optimizer=adam)
# discriminator.compile(loss=hyb, optimizer=adam)
# GAN.compile(loss=hyb, optimizer=adam)



# In[ ]:


#Load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
dataGenerator = image.ImageDataGenerator(preprocessing_function = utils.dataRescale)
batchIterator = dataGenerator.flow(X_train, batch_size = batch_size)


# In[ ]:



# In[ ]:


trainer = trainer.GANTrainer(generator, discriminator, GAN, batchIterator,l)
trainer.train(epochs, batches_per_epoch, batch_size, gamma)


# In[ ]:



