#Command to test on server: python3.6 caepia_with_voting.py -d "../../data/changeDetection/*" -o "../output/test1"

import os
import sys
import random
import math
import cv2
import argparse
import numpy as np
from scipy import misc, ndimage
from keras import backend as KB
import keras
import tensorflow as tf
from glob import glob
import time
from multiprocessing import Process

# Root directory
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)
import Background_model_iwinac_parallel                            # pylint: disable=import-error
import Encoder
import image_utils                                         # pylint: disable=import-error
import file_utils
import dataset_utils


ENCODER_PATH = '../output/models/autoencoder_4_t_16x16/encoder.h5py'
DATASET_FOLDER = "../data/data_to_segment/"
OUTPUT_FOLDER = "../output/segmentation/test5/"
CATEGORY_TO_SEARCH = ["dynamicBackground"]
VIDEO_TO_SEARCH = ["boats","canoe","fountain01","fountain02","overpass"]
MAX_NUMBER_OF_PARALLEL_PROCESSES = 1
NOISES_LIST = ["gaussian_noise_0_1","gaussian_noise_0_2","gaussian_noise_0_3","gaussian_noise_0_4"]
NOISES_LIST = ["salt_and_pepper", "gaussian_noise_0_4"]
NOISES_LIST = ["no_noise","gaussian_noise_0_3"]

JOIN_TIME = 30

IMAGE_INTERVAL_BETWEEN_LOG = 100

parameters = {
    "canoe":([4,5,6],[2,3,4],[0.001]),
    "boats":([4],[3],[0.001]),
    "fountain02":([6],[15],[0.001]),
    "overpass":([6],[3],[0.001]),
    "pedestrians":([7],[15],[0.001]),
    "fountain01":([2,3,4,5,6,7,8],[3,6,9,12,15],[0.001,0.005,0.01,0.05]),
    "fall":([2,3,4,5,6,7,8],[3,6,9,12,15],[0.001,0.005,0.01,0.05])
}

T=16

initial_time = time.time()

def segmentation(images_path, output_path_in, encoder_model_path, category, dataset, n_img_to_train, K= -1, C = -1, ALPHA = -1, T = -1, prefix = ""):

    ###################################
    os.environ["CUDA_VISIBLE_DEVICES"]="1"                      # To force tensorflow to only see one GPU.
    # TensorFlow wizardry
    config = tf.ConfigProto()
     
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True                      # pylint: disable=import-error
     
    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 0.95/MAX_NUMBER_OF_PARALLEL_PROCESSES    # pylint: disable=import-serror
     
    # Create a session with the above options specified.
    KB.tensorflow_backend.set_session(tf.Session(config=config))
    ###################################
    
    print ("Loading encoder model from " + encoder_model_path)
    encoder = Encoder.Encoder(encoder_model_path=encoder_model_path)
                                                                                                    # Code to execute semantic segmentation.                  
    output_path = os.path.join(output_path_in, category, dataset)                                   # We generate output path.
    if not os.path.isdir(output_path):                                                              # We check if output path exists.
        os.makedirs(output_path)                                                                    # We create path if it does not exist.

    total_image_number = len(images_path)                                                           # We get total number of images path.
    segmented_images_count = 0                                                                      # We set segmented images counter to 0.

    images_path = sorted(images_path)                                                               # We ensure images are ordered by name.
    for i, img_path in enumerate(images_path):                                                      # For each image path...
        if (os.path.isfile(img_path) and (img_path[-4:]==".png" or img_path[-4:]==".jpg")):         # If it is a file and has image extensión...
            segmented_images_count += 1                                                             # We update counter.
            if segmented_images_count%IMAGE_INTERVAL_BETWEEN_LOG==0:
                print((prefix+" Processing image " + img_path + "\nProcessing {} / {} from " 
                    + category + "/" + dataset).format(segmented_images_count,total_image_number))      # Print progress info.
            img = cv2.imread(img_path)                                                              # We load the image.    
            
            output_filename = os.path.join(output_path, 
                file_utils.generate_file_name_by_number(segmented_images_count))                      # We generate the segmented image path.

            if segmented_images_count == 1:                                                         # If this is the first image...
                back_model = Background_model_iwinac_parallel.Background_model(encoder=encoder, 
                patch_side = 16, number_of_images_for_training = n_img_to_train, 
                K = K, C = C, ALPHA = ALPHA, T = T)                                                 # We initialize the background model from the first image.

            back_model.next_image(img, training = segmented_images_count <= n_img_to_train)         # We process the image with background model.
            
            if segmented_images_count > n_img_to_train:
                segmented_img = back_model.get_last_segmented_image()                               # We get segmented image from background model.
                cv2.imwrite(output_filename, segmented_img)                                         # We save the segmented image.
                
                """
                tesseras_images = back_model.get_last_segmented_tesseras_adjusted_to_image_shape()
                for tes in range(tesseras_images.shape[0]):
                    tesseras_path = os.path.join(output_path, "tesseras_cut")                    
                    if not os.path.isdir(tesseras_path):                                                              # We check if output path exists.
                        os.makedirs(tesseras_path)                                                                    # We create path if it does not exist.
                    output_filename = os.path.join(tesseras_path, 
                        my_utils.generate_file_name_by_number(segmented_images_count) + "_Tessera=" + str(tes))   # We generate the segmented tessera image path.
                    cv2.imwrite(output_filename, tesseras_images[tes])                                          # We write the image
                
                tesseras_images = back_model.get_last_segmented_tesseras()
                for tes in range(tesseras_images.shape[0]):
                    tesseras_path = os.path.join(output_path, "tesseras")                    
                    if not os.path.isdir(tesseras_path):                                                              # We check if output path exists.
                        os.makedirs(tesseras_path)                                                                    # We create path if it does not exist.
                    output_filename = os.path.join(tesseras_path, 
                        my_utils.generate_file_name_by_number(segmented_images_count) + "_Tessera=" + str(tes))   # We generate the segmented tessera image path.
                    cv2.imwrite(output_filename, tesseras_images[tes])                                          # We write the image
                """

        else:                                                                                       # This path is not a file or the file is not an image...
            total_image_number -= 1                                                                 # We update total image number.
            print(("This element is not an image : " + img_path + "\nProcessed {} / {} from " 
                + category + "/" + dataset).format(segmented_images_count,total_image_number))      # Print progress info.
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--encoder_model', type=str, default=ENCODER_PATH,
                        help='Encoder model path')
    parser.add_argument('-d', '--datasets_dir', type=str, default=DATASET_FOLDER, 
                        help="Path to the input datasets folder.")
    parser.add_argument('-o', '--output_path', type=str, default=OUTPUT_FOLDER,
                        help='Path to output')
    parser.add_argument('--id', default="0")
    
    args = parser.parse_args()

    # Handle output args
    if args.datasets_dir:
        fn, ext = os.path.splitext(args.output_path)
        print("Output folder: " + fn)
        if ext:
            parser.error("output_path should be a folder for multiple file input")
        if not os.path.isdir(args.output_path):
            os.makedirs(args.output_path)

    processes_list = []
    print(args)        
    if args.datasets_dir:
        print (args.datasets_dir)

        for noise in NOISES_LIST:
            dataset_dir = os.path.join(args.datasets_dir, noise)
            noise_output_path = os.path.join(args.output_path, noise)
            for (video_path, video_name, video_category) in dataset_utils.get_changeDetection_dirs_and_info(dataset_dir, category_to_search = CATEGORY_TO_SEARCH, video_to_search = VIDEO_TO_SEARCH, do_print = True):
                video_images = glob(os.path.join(video_path, '*'))

                _, n_img_to_train = dataset_utils.obtain_information_about_dataset(video_name)

                K_list,C_list,ALPHA_list = parameters[video_name]
                for K in K_list:
                    for C in C_list:
                        for ALPHA in ALPHA_list:
                            print("K="+str(K)+"_C="+str(C)+"_ALPHA="+str(ALPHA))
                            output_path = os.path.join(noise_output_path, "T="+str(T), "K="+str(K)+"_C="+str(C)+"_ALPHA="+str(ALPHA))
                            prefix = "K="+str(K)+"_C="+str(C)+"_ALPHA="+str(ALPHA)+":"
                            keywords = {'K':K, 'C':C, 'ALPHA':ALPHA, 'T':T, 'prefix':prefix}                                                         # Preparamos los argumentos nominales.
                            p = Process(target=segmentation, args=(video_images, output_path, args.encoder_model,
                                video_category, video_name, n_img_to_train), kwargs=keywords)                                       # Creamos el proceso.
                            p.start()                                                                                               # Iniciamos el proceso.
                            processes_list.append(p)                                                                                # Almacenamos el proceso en la lista.
                            index = 0
                            while len(processes_list) >= MAX_NUMBER_OF_PARALLEL_PROCESSES:
                                p = processes_list[index]
                                p.join(JOIN_TIME)               
                                if not p.exitcode is None:                                                                           # This process has finished.
                                    processes_list.pop(index)
                                index += 1
                                if index == len(processes_list):
                                    index = 0

    final_time = time.time()

    print("Total time: " + final_time - initial_time)
                                    
