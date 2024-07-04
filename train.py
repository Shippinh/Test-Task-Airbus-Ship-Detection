# This file dedicated to training the model.
# Chosen model: U-net
# Model grading: dice score 
# Modules: os, keras, numpy #

import os
import keras
import matplotlib.pyplot
import numpy
import matplotlib

#Seeding
os.environ["PYTHONHASHSEED"] = str(38)
numpy.random.seed(38)
keras.utils.set_random_seed(38)

#Parameters
batch_size = 2
learning_rate = 0.0001
epochs = 100
img_height = 768
img_width = 768

#Path
dataset_path = os.path.join("airbus-ship-detection")

out_dir = os.path.join("output", "files")
model_file = os.path.join(out_dir, "unet.h5")
log_file = os.path.join(out_dir, "log.csv")

#Creating folder
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

create_dir(out_dir)

#Building UNET
#Convolution
def convolute(inputs, num_filters):
    x = keras.layers.Conv2D(num_filters, 3, padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.Conv2D(num_filters, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    return x

#Encoder
def encode(inputs, num_filters):
    x = convolute(inputs, num_filters)
    p = keras.layers.MaxPool2D((2,2))(x)
    return x, p

#Decoder
def decode(inputs, skip, num_filters):
    x = keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = keras.layers.Concatenate()([x, skip])
    x = convolute(x, num_filters)
    return x

#UNET
def build_unet(input_shape):
    inputs = keras.layers.Input(input_shape)

    #Encoding
    s1, p1 = encode(inputs, 64)
    s2, p2 = encode(p1, 128)
    s3, p3 = encode(p2, 256)
    s4, p4 = encode(p3, 512)

    #The bottleneck
    b1 = convolute(p4, 1024)

    #Decoding
    d1 = decode(b1, s4, 512)
    d2 = decode(b1, s3, 256)
    d3 = decode(b1, s2, 128)
    d4 = decode(b1, s1, 64)

    outputs = keras.layers.Conv2D(1,1, padding="same", activation="sigmoid")(d4)

    model = keras.models.Model(inputs, outputs, name="UNET")
    return model

#Decoding bounding box
def rle_decode(mask_rle, shape):

    s = mask_rle.split()
    starts, lengths = [numpy.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = numpy.zeros(shape[0] * shape[1], dtype=numpy.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

#matplotlib.pyplot.imsave('filename.png', numpy.array(mask), cmap=matplotlib.cm.gray)

#Next is 09:36, picking validation images and masks from total train set 