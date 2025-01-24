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
from datetime import datetime
from multiprocessing import Process

# Root directory
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)
import Background_model_parallel_online_update                            # pylint: disable=import-error
import Encoder
import image_utils                                         # pylint: disable=import-error
import file_utils
import dataset_utils

RO = None
load_encoder_from_json = True

sigmoid_autoencoder_list = [6,9,10]


AUTOENCODER_NUMERATION = 8
MORPHOLOGICAL_OPENING_KERNEL = None
INITIAL_PI_FORE = 0.5
INITIAL_PI_BACK = 0.5
UPDATE_PI = False


min_encode_value = 0
max_encode_value = 1

if MORPHOLOGICAL_OPENING_KERNEL is None:
    OUTPUT_FOLDER = "../output/segmentation/new_model/no_update_pi_test"+str(AUTOENCODER_NUMERATION)+"_INITIAL_PI_FORE_"+str(INITIAL_PI_FORE)+"_INITIAL_PI_BACK_"+str(INITIAL_PI_BACK)+"/"
else:    
    OUTPUT_FOLDER = "../output/segmentation/new_model/no_update_pi_test"+str(AUTOENCODER_NUMERATION)+"_closing/"
    
ENCODER_PATH = '../output/models/autoencoder_'+str(AUTOENCODER_NUMERATION)+'_t_16x16/encoder'
DATASET_FOLDER = "../data/data_to_segment/"
DATASET_FOLDER = "/usr/share/Data1/Datasets/changeDetection_noise"
CATEGORY_TO_SEARCH = ["shadow", "badWeather", "baseline", "nightVideos"]
CATEGORY_TO_SEARCH = ["dynamicBackground", "shadow", "badWeather", "baseline", "nightVideos"]
CATEGORY_TO_SEARCH = ["dynamicBackground"]
#CATEGORY_TO_SEARCH = ["lowFramerate", "intermittentObjectMotion", "turbulence", "thermal"]
VIDEO_TO_SEARCH = ["ALL"]
#VIDEO_TO_SEARCH = ["canoe"]
MAX_NUMBER_OF_PARALLEL_PROCESSES = 10
NOISES_LIST = ["gaussian_noise_0_1","gaussian_noise_0_2","gaussian_noise_0_3","gaussian_noise_0_4"]
NOISES_LIST = ["salt_and_pepper", "gaussian_noise_0_4"]
NOISES_LIST = ["no_noise","gaussian_noise_0_1","gaussian_noise_0_2","gaussian_noise_0_3","gaussian_noise_0_4", "uniform", "salt_and_pepper","jpeg_compression_10","jpeg_compression_1"]
NOISES_LIST = ["no_noise", "gaussian_noise_0_2","gaussian_noise_0_4", "uniform", "salt_and_pepper","jpeg_compression_10","jpeg_compression_1"]
NOISES_LIST = ["gaussian_noise_0_3","gaussian_noise_0_1","medium_mask_noise"]
NOISES_LIST = ["no_noise", "gaussian_noise_0_2","gaussian_noise_0_4", "uniform", "salt_and_pepper","jpeg_compression_10","jpeg_compression_1","gaussian_noise_0_3","gaussian_noise_0_1","medium_mask_noise","big_mask_noise","medium_mask_noise","small_mask_noise"]
NOISES_LIST = ["uniform","big_mask_noise", "small_mask_noise", "medium_mask_noise"]
NOISES_LIST = ["no_noise", "gaussian_noise_0_2","gaussian_noise_0_4", "uniform", "uniform_2", "salt_and_pepper","jpeg_compression_10","jpeg_compression_1","gaussian_noise_0_3","gaussian_noise_0_1","small_mask_noise"]
NOISES_LIST = ["small_mask_noise", "medium_mask_noise", "big_mask_noise"]

JOIN_TIME = 30
IMAGE_INTERVAL_BETWEEN_LOG = 50

parameters = {
    "canoe":([0.005]),
    "boats":([0.005]),
    "fountain02":([0.005]),
    "overpass":([0.005]),
    "pedestrians":([0.005]),
    "fountain01":([0.005]),
    "fall":([0.005])
}

GENERAL_ALPHA_LIST = [0.005]

T=16

def segmentation(images_path, output_path_in, encoder_model_path, category, dataset, n_img_to_train, ALPHA = -1, T = -1, RO = None, prefix = "", log = None):
    initial_time = time.time()
    print(prefix + " Start.") 

    ###################################
    os.environ["CUDA_VISIBLE_DEVICES"]="0"                      # To force tensorflow to only see one GPU.
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
    encoder = Encoder.Encoder(encoder_model_path=encoder_model_path, load_from_json = load_encoder_from_json)
                                                                                                    # Code to execute semantic segmentation.                  
    output_path = os.path.join(output_path_in, category, dataset)                                   # We generate output path.
    if not os.path.isdir(output_path):                                                              # We check if output path exists.
        os.makedirs(output_path)                                                                    # We create path if it does not exist.

    total_image_number = len(images_path)                                                           # We get total number of images path.
    segmented_images_count = 0                                                                      # We set segmented images counter to 0.

    images_path = sorted(images_path)                                                               # We ensure images are ordered by name.
    for i, img_path in enumerate(images_path):                                                      # For each image path...
        if (os.path.isfile(img_path) and (img_path[-4:]==".png" or img_path[-4:]==".jpg" or img_path[-5:]==".jpeg")):         # If it is a file and has image extensión...
            img = cv2.imread(img_path)                                                              # We load the image.    

            if segmented_images_count == 0:                                                         # If this is the first image...
                back_model = Background_model_parallel_online_update.Background_model(encoder=encoder, 
                patch_side = 16, number_of_images_for_training = n_img_to_train, ALPHA = ALPHA, T = T,
                RO = RO, min_encode_value = min_encode_value, max_encode_value = max_encode_value,
                initial_pi_Back = INITIAL_PI_BACK, initial_pi_Fore = INITIAL_PI_FORE, update_pi = UPDATE_PI)  # We initialize the background model from the first image.
            #print(segmented_images_count+1)
            back_model.next_image(img, training = segmented_images_count < n_img_to_train)          # We process the image with background model.           
            
            segmented_images_count += 1                                                             # We update counter.
            #if (segmented_images_count == 890):
            #    quit()
            output_filename = os.path.join(output_path, 
                file_utils.generate_file_name_by_number(segmented_images_count))                    # We generate the segmented image path.
                
            if segmented_images_count == 1 or segmented_images_count%IMAGE_INTERVAL_BETWEEN_LOG==0:
                print((prefix+" Processing image " + img_path + "\nProcessing {} / {} from " 
                    + category + "/" + dataset).format(segmented_images_count,total_image_number))  # Print progress info.
            
            if segmented_images_count > n_img_to_train:
                segmented_img = back_model.get_last_segmented_image()                               # We get segmented image from background model.
                
                if not MORPHOLOGICAL_OPENING_KERNEL is None:
                    normalized_segmented_img = segmented_img/255.
                    closing = cv2.morphologyEx(normalized_segmented_img, cv2.MORPH_CLOSE, MORPHOLOGICAL_OPENING_KERNEL)
                    segmented_img = closing*255.
                    
                cv2.imwrite(output_filename, segmented_img)                                         # We save the segmented image.

        else:                                                                                       # This path is not a file or the file is not an image...
            total_image_number -= 1                                                                 # We update total image number.
            print(("This element is not an image : " + img_path + "\nProcessed {} / {} from " 
                + category + "/" + dataset).format(segmented_images_count,total_image_number))      # Print progress info.
    time_to_finish = time.time()-initial_time
    print(prefix + " Finish. Time: " + str(time_to_finish))

    if not log is None:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("date and time =", dt_string)	
        with open(log, 'a') as f:
            f.write(prefix + "  " + "OK" + " date: " + dt_string + " time to finish " +str(time_to_finish)+"\n")

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
            
    initial_time = time.time()

    processes_list = []
    log = os.path.join(args.output_path,"log.txt")
    print(args)
    if args.datasets_dir:
        print (args.datasets_dir)

        for noise in NOISES_LIST:
            dataset_dir = os.path.join(args.datasets_dir, noise)
            for (video_path, video_name, video_category) in dataset_utils.get_changeDetection_dirs_and_info(dataset_dir, category_to_search = CATEGORY_TO_SEARCH, video_to_search = VIDEO_TO_SEARCH, do_print = True):
                video_images = glob(os.path.join(video_path, '*'))

                #_, n_img_to_train = dataset_utils.obtain_information_about_dataset(video_name)
                n_img_to_train = dataset_utils.get_CDNET_number_of_training_images(video_category, video_name)

                ALPHA_list = GENERAL_ALPHA_LIST
                for ALPHA in ALPHA_list:
                    output_path = os.path.join(args.output_path, "T="+str(T), "ALPHA="+str(ALPHA) + "_RO="+str(RO) + "_INITIAL_PI_BACK="+str(INITIAL_PI_BACK), noise)
                    prefix = video_name+"_"+noise+"_"+ "T="+str(T) +"_ALPHA="+str(ALPHA) + "_RO="+str(RO)+":"
                    if ( MAX_NUMBER_OF_PARALLEL_PROCESSES > 1):
                        keywords = {'RO':RO, 'ALPHA':ALPHA, 'T':T, 'prefix':prefix, "log":log}                                                         # Preparamos los argumentos nominales.
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
                    else:
                        segmentation(video_images, output_path, args.encoder_model, video_category, video_name, n_img_to_train, RO=RO, ALPHA=ALPHA, T=T, prefix=prefix)

#We will ensure all proceses has finished.
    index = 0
    while len(processes_list) > 0:
        p = processes_list[index]
        p.join(JOIN_TIME)               
        if not p.exitcode is None:                                                                           # This process has finished.
            processes_list.pop(index)
        index += 1
        if index >= len(processes_list):
            index = 0
    final_time = time.time()

    print(log)
    print("Total time: " + str(final_time - initial_time))
    
                                    
