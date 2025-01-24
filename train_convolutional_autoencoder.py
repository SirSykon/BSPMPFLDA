#Code to train an autoencoder.

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten
from keras.models import Model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import numpy as np
import os
import cv2
import math
import shutil
from random import shuffle
import tensorflow as tf
import time
from glob import glob
import random

import noise_utils
import dataset_utils
import train_utils
import file_utils
import image_utils

###################################
os.environ["CUDA_VISIBLE_DEVICES"]="0" #To force tensorflow to only see one GPU.
# TensorFlow wizardry
config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.9
 
# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))
###################################

##########################################################################
############################## Input data ################################
##########################################################################

AUTOENCODER_NUMERATION = 3
VALIDATION_SPLIT = 0.2
TRAIN_PATCH_HEIGHT = 16
TRAIN_PATCH_WIDTH = 16
TRAIN_DATASET_FOLDER = "../data/training_data/"+str(TRAIN_PATCH_HEIGHT)+"x"+str(TRAIN_PATCH_WIDTH)+"/"
OUTPUT_MODELS_FOLDER = "../output/models/"
OUTPUT_MODEL_PATH = os.path.join(OUTPUT_MODELS_FOLDER, "autoencoder_" + str(AUTOENCODER_NUMERATION) + "_t_" + str(TRAIN_PATCH_HEIGHT) + "x" + str(TRAIN_PATCH_WIDTH))
BATCH_SIZE = 32                                         # We define batch size.
TRAINING_EPOCHS = 35                                    # We define training epochs.

NUMBER_OF_CHANNELS = 3
CONVOLUIONAL_FILTER_SIZE = 5
MAX_POOLING_SIZE = 2
AVERAGE_FILTER_SIZE = None
MEDIAN_FILTER_SIZE = None
PRELOAD_IMAGES_IN_RAM = True
MAX_NUMBER_OF_IMAGES_USED_FOR_TRAINING_AND_VALIDATION = None
IMG_RESIZE_SHAPE = None

INITIAL_RANDOM_SEED = 1234

random.seed(INITIAL_RANDOM_SEED)

initial_time = time.time()

##########################################################################
############################# Preparation ################################
##########################################################################

if not os.path.isdir(OUTPUT_MODEL_PATH):                                                             # We check if output path exists.
    os.makedirs(OUTPUT_MODEL_PATH)                                                                   # We create path if it does not exist.

# We will copy all code to /src inside output_model_path.

file_utils.copy_files_to_folder(glob("./*.py"), os.path.join(OUTPUT_MODEL_PATH, "src"))

##########################################################################
######################### Auxiliar Functions #############################
##########################################################################

def apply_random_transformation(image):
    rand = random.randint(0,2)
    if rand == 0:
        return noise_utils.add_low_gaussian_noise(image)
    if rand == 1:
        return noise_utils.add_medium_gaussian_noise(image)
    if rand == 2:
        return noise_utils.add_high_gaussian_noise(image)

##########################################################################
########################## Network Structure #############################
##########################################################################

input_img = Input(shape=(TRAIN_PATCH_HEIGHT, TRAIN_PATCH_WIDTH, NUMBER_OF_CHANNELS))  # adapt this if using `channels_first` image data format

x = Conv2D(32, (CONVOLUIONAL_FILTER_SIZE, CONVOLUIONAL_FILTER_SIZE), activation='relu', padding='same')(input_img)
x = MaxPooling2D((MAX_POOLING_SIZE, MAX_POOLING_SIZE), padding='same')(x)
x = Conv2D(64, (CONVOLUIONAL_FILTER_SIZE, CONVOLUIONAL_FILTER_SIZE), activation='relu', padding='same')(x)
last_encoder_layer = MaxPooling2D((MAX_POOLING_SIZE, MAX_POOLING_SIZE), padding='same', name="last_encoder_layer")(x)

first_decoder_layer = Conv2D(64, (CONVOLUIONAL_FILTER_SIZE, CONVOLUIONAL_FILTER_SIZE), activation='relu', padding='same', name="first_decoder_layer")(last_encoder_layer)
x = UpSampling2D((MAX_POOLING_SIZE, MAX_POOLING_SIZE))(first_decoder_layer)
x = Conv2D(32, (CONVOLUIONAL_FILTER_SIZE, CONVOLUIONAL_FILTER_SIZE), activation='relu', padding = 'same')(x)
x = UpSampling2D((MAX_POOLING_SIZE, MAX_POOLING_SIZE))(x)
last_decoder_layer = Conv2D(NUMBER_OF_CHANNELS, (CONVOLUIONAL_FILTER_SIZE, CONVOLUIONAL_FILTER_SIZE), activation='relu', padding='same')(x)

autoencoder = Model(input_img, last_decoder_layer)
autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics = ["accuracy"])

##########################################################################
################################ Data ####################################
##########################################################################

if PRELOAD_IMAGES_IN_RAM:

    train_data, validation_data = image_utils.get_normalized_images_from_folder(TRAIN_DATASET_FOLDER, validation_split = VALIDATION_SPLIT, max_number_of_images_used_for_training_and_validation=MAX_NUMBER_OF_IMAGES_USED_FOR_TRAINING_AND_VALIDATION)

else:
    print("Reading data from folder " + OUTPUT_MODEL_PATH)

    print("Loading data.")
    images_path = glob(os.path.join(OUTPUT_MODEL_PATH,'*'))

    shuffle(images_path)

    train_data = images_path[:-int(len(images_path)*VALIDATION_SPLIT)]
    validation_data = images_path[-int(len(images_path)*VALIDATION_SPLIT):]

#python3 -m tensorboard.main --logdir=../tmp/autoencoder_train

##########################################################################
################################ Train ###################################
##########################################################################

image_transform_function = apply_random_transformation

#datagen.fit() # There is no need of this function since "Only required if `featurewise_center` or`featurewise_std_normalization` or `zca_whitening` are set to True."

#train_generator = datagen.flow(x_train, y=x_train, batch_size=batch_size, shuffle=True) # There is no need of this since we generate data on the fly.
#test_generator = datagen.flow(x_test, y=x_test, batch_size=batch_size, shuffle=True) # There is no need of this since we generate data on the fly.

history = train_utils.train_model_with_generator(autoencoder, train_data, validation_data, TRAINING_EPOCHS, BATCH_SIZE, image_transform_function, PRELOAD_IMAGES_IN_RAM)

train_utils.save_model(autoencoder, OUTPUT_MODEL_PATH, "autoencoder", plot_graph_model=False)

autoencoder.summary()

train_utils.save_training_and_validation_loss_from_history(history, OUTPUT_MODEL_PATH)

train_utils.save_training_and_validation_accuracy_from_history(history, OUTPUT_MODEL_PATH)

encoder = Model(input_img, last_encoder_layer)

encoder.summary()

train_utils.save_model(encoder, OUTPUT_MODEL_PATH, "encoder", plot_graph_model=False)

decoder_input = None
deco = None
for layer in autoencoder.layers:
    print(layer.name)
    print(layer.output_shape)
    if not decoder_input is None:

        if deco is None:
            deco = layer(decoder_input)
        else:
            deco = layer(deco)

    if layer.name == "last_encoder_layer":
        decoder_input = Input(shape=(layer.output_shape[1:]))

decoder = Model(decoder_input, deco)

decoder.summary()

train_utils.save_model(decoder, OUTPUT_MODEL_PATH, "decoder", plot_graph_model=False)
