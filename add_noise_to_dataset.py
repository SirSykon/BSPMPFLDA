#Command to test on server: 
#python3.6 add_noise_to_dataset.py -d "../../data/changeDetection/*" -o "../output/data/autoencoded/_/noise_0/"
#python3.6 add_noise_to_dataset.py -d "../../NeurocomputingPatches/datasets_with_noise/0.1/*" -o "../output/data/autoencoded/_/gaussian_0.1/"
#python3.6 add_noise_to_dataset.py -d "../../NeurocomputingPatches/datasets_with_noise/0.2/*" -o "../output/data/autoencoded/_/gaussian_0.2/"

#Command to use on icai24:
#python3.6 add_noise_to_dataset.py -d "../../data/dataset2014/dataset/*" -o "../output/"

import numpy as np
import sys
import cv2
import os
import math
import time
import argparse
from glob import glob
import shutil
import tensorflow as tf
from keras import backend as K

import image_utils
import dataset_utils
import file_utils

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)

import Autoencoder
import noise_utils

DATASET_FOLDER = "/usr/share/Data1/Datasets/changeDetection/"
OUTPUT_FOLDER = "../data/data_to_segment/uniform_2/"
CATEGORY_TO_SEARCH = ["dynamicBackground"]
CATEGORY_TO_SEARCH = ["dynamicBackground", "baseline", "badWeather", "intermittentObjectMotion", "lowFramerate", "cameraJitter", "nightVideos", "shadow", "thermal", "turbulence"]
noise_function_to_apply = noise_utils.add_uniform_noise

#jpg_quality = 10

def add_noise_to_dataset(images_path, output_path_in, category, dataset):

    processed_images = 0
    output_path = os.path.join(output_path_in, category, dataset)                                   # We generate output path.
    if not os.path.isdir(output_path):                                                              # We check if output path exists.
        os.makedirs(output_path)                                                                    # We create path if it does not exist.
    
    images_path = sorted(images_path)                                                               # We ensure images are ordered by name.
    total_image_number = len(images_path)
    for i, img_path in enumerate(images_path):                                                      # For each image path...
        if (os.path.isfile(img_path) and (img_path[-4:]==".png" or img_path[-4:]==".jpg")):         # If it is a file and has image extensión...
            processed_images += 1                                                                   # We update counter.
            print(("Processing image " + img_path + "\nProcessing {} / {} from " 
                + category + "/" + dataset).format(processed_images,total_image_number))            # Print progress info.
            img = cv2.imread(img_path)                                                              # We load the image.
            
            output_filename = os.path.join(output_path, 
                file_utils.generate_file_name_by_number(processed_images, prefix="in", suffix=".png"))             # We generate the segmented image path.
            
            normalized_img = img/255.
            normalized_image_with_noise = noise_function_to_apply(normalized_img)

            normalized_image_with_noise = np.clip(normalized_image_with_noise, 0, 1, out=normalized_image_with_noise)

            denormalize_image_matrix = np.round(normalized_image_with_noise*255, decimals=0)

            cv2.imwrite(output_filename, denormalize_image_matrix)
            #cv2.imwrite(output_filename, img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datasets_dir', type=str, default=DATASET_FOLDER, 
                        help="Path to the input datasets folder.")
    parser.add_argument('-o', '--output_folder', type=str, default=OUTPUT_FOLDER,
                        help='Path to output')

    file_utils.copy_files_to_folder(glob("./*.py"), os.path.join(OUTPUT_FOLDER, "src"))
    
    args = parser.parse_args()

    # Handle output args
    output_path = args.output_folder

    print(args)        
    if args.datasets_dir:
        print (args.datasets_dir)
        for (video_path, video_name, video_category) in dataset_utils.get_changeDetection_dirs_and_info(args.datasets_dir, category_to_search = CATEGORY_TO_SEARCH, do_print = True):
            video_images = glob(os.path.join(video_path, '*'))
            add_noise_to_dataset(video_images, OUTPUT_FOLDER, video_category, video_name)
            
