import cv2
import numpy as np
import os
from glob import glob
import shutil

def generate_file_name_by_number(file_number, number_of_digits = 6, prefix = "bin", suffix = ".png"):
    file_name = "error"
    zeroes_to_add = number_of_digits - len(str(file_number))
    
    if zeroes_to_add < 0:
        raise("Error: number of digits lesser than file_number length.")
    else:
        file_name = prefix + zeroes_to_add * "0" + str(file_number) + suffix
        
    return file_name

def copy_files_to_folder(file_list, folder_path):

    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    for file in file_list:
        _, file_name = os.path.split(file)
        shutil.copyfile("./"+file_name, os.path.join(folder_path,file_name))          # We save the files used to train to know how the model has been trained.



