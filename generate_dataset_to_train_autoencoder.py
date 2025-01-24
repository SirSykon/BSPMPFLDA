import numpy as np
import sys
import cv2
import os
import math
import time
import argparse
import random
from random import randint
from glob import glob

import file_utils
import dataset_utils

category_to_search = ["all"]
category_to_ignore = ["dynamicBackground"]
video_to_search = ["all"]
video_to_ignore = []

DATASET_DIR = "/usr/share/Data1/Datasets/COCO-dataset/images/train2017"
OUTPUT_IMAGES_PATCH_HEIGHT = 16
OUTPUT_IMAGES_PATCH_WIDTH = 16
PATCH_IMAGES_TO_OBTAIN = 400000
PATCH_IMAGES_OUTPUT_PATH = "../data/training_data/" + str(OUTPUT_IMAGES_PATCH_HEIGHT) + "x" + str(OUTPUT_IMAGES_PATCH_WIDTH)
INITIAL_RANDOM_SEED = 1234

random.seed(INITIAL_RANDOM_SEED)

def get_images_from_dataset(images_path, output_path_in, patch_height, patch_width, images_to_obtain):

    output_path = os.path.join(output_path_in)                                                      # We generate output path.
    if not os.path.isdir(output_path):                                                              # We check if output path exists.
        os.makedirs(output_path)                                                                    # We create path if it does not exist.
    
    images_path = sorted(images_path)                                                               # We ensure images are ordered by name.
    total_image_number = len(images_path)
    print(total_image_number)
    extracted_datas = 0
    while extracted_datas < images_to_obtain:
        img_path = images_path[randint(0,total_image_number-1)]                                       # We select an image.

        if (os.path.isfile(img_path) and (img_path[-4:]==".png" or img_path[-4:]==".jpg")):         # If it is a file and has image extensión...

            img = cv2.imread(img_path)                                                              # We load the image.
            shape = img.shape

            if (shape[0]-patch_height-1 >= 0 and shape[1]-patch_width-1 >= 0):                       # If there is space  in the image to obtain the patch
                initial_height = randint(0, shape[0]-patch_height-1)
                initial_width = randint(0, shape[1]-patch_width-1)
                data = img[initial_height:initial_height + patch_height, initial_width:initial_width + patch_width, :]
                extracted_datas += 1         
                output_filename = os.path.join(output_path, 
                    file_utils.generate_file_name_by_number(extracted_datas, prefix = "train"))         # We generate data image path.

                print("Processing image " + img_path + " to " + output_filename +
                    "\nProcessing {} / {}".format(extracted_datas,images_to_obtain))                    # Print progress info.
                cv2.imwrite(output_filename, data)


            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str, default=DATASET_DIR, 
                        help="Path to the input datasets folder.")
    parser.add_argument('-o', '--output_path', type=str, default=PATCH_IMAGES_OUTPUT_PATH,
                        help='Path to output')
    parser.add_argument('-i', '--images_to_obtain', type=str, default=PATCH_IMAGES_TO_OBTAIN,
                        help='Number of images taken.')
    parser.add_argument('-hght', '--height', type=str, default=OUTPUT_IMAGES_PATCH_HEIGHT,
                        help='Data height.')
    parser.add_argument('-wdth', '--width', type=str, default=OUTPUT_IMAGES_PATCH_WIDTH,
                        help='Data width.')
    
    args = parser.parse_args()

    # Handle output args
    print("Input folder:" + args.dataset_dir)
    if args.dataset_dir:
        fn, ext = os.path.splitext(args.output_path)
        print("Output folder: " + fn)
        if ext:
            parser.error("output_path should be a folder for multiple file input")
        if not os.path.isdir(args.output_path):
            os.makedirs(args.output_path)

    dataset_images_list = glob(os.path.join(args.dataset_dir,"*"))
    get_images_from_dataset(dataset_images_list, args.output_path, args.height, args.width, args.images_to_obtain)
