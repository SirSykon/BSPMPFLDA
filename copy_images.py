import shutil
import os

dst_folder = "../article/images/"
dataset_folder = "/home/jorgegarcia//Documents/Work/bgs_handler/output/ICAE2019/"
dataset_folder = "/usr/share/Data1/Datasets/changeDetection_noise/"
#dataset_folder = "/home/jorgegarcia/Documents/experiments/icae_2019/output/segmentation/new_model/no_update_pi_test8_INITIAL_PI_FORE_0.5_INITIAL_PI_BACK_0.5/T=16/ALPHA=0.005_RO=None_INITIAL_PI_BACK=0.5/"
METHODS_LIST = ["PAWCS", "LOBSTER", "KDE","SUBSENSE", "fuzzy_adaptative_som", "adaptative_som", "zivkovic", "wren"]
NOISES_LIST = ["jpeg_compression_10","jpeg_compression_1"]
NOISES_LIST = ["no_noise"]
#NOISES_LIST = ["gaussian_noise_0_1","gaussian_noise_0_2","gaussian_noise_0_3","gaussian_noise_0_4", "uniform", "uniform_2", "salt_and_pepper", "small_mask_noise"]
#NOISES_LIST = ["no_noise","jpeg_compression_10","jpeg_compression_1","gaussian_noise_0_1","gaussian_noise_0_2","gaussian_noise_0_3","gaussian_noise_0_4", "uniform", "uniform_2", "salt_and_pepper", "small_mask_noise"]
category = "dynamicBackground"
video = "fountain02"
image_name = "in000658.jpeg"
image_name = "bin001275.png"
image_name = "in001275.jpg"

def copy_rival_image(dataset_folder,method, noise, category, video, image_name, dst_folder):
    image_path = os.path.join(dataset_folder, method, noise, category, video, image_name)
    image_path_to_copy = os.path.join(dst_folder, method+"_"+noise+"_"+video+"_"+image_name)
    
    shutil.copy(image_path, image_path_to_copy)
   
def copy_original_image(dataset_folder, noise, category, video, image_name, dst_folder):
    image_path = os.path.join(dataset_folder, noise, category, video, image_name)
    image_path_to_copy = os.path.join(dst_folder, "original_"+noise+"_"+video+"_"+image_name)
    
    shutil.copy(image_path, image_path_to_copy)
    
def copy_ours8_image(dataset_folder, noise, category, video, image_name, dst_folder):
    image_path = os.path.join(dataset_folder, noise, category, video, image_name)
    image_path_to_copy = os.path.join(dst_folder, "ours8_"+noise+"_"+video+"_"+image_name)
    
    shutil.copy(image_path, image_path_to_copy)
    
#for method in METHODS_LIST:
#    for noise in NOISES_LIST:    
#        copy_rival_image(dataset_folder, method, noise, category, video, image_name, dst_folder)

for noise in NOISES_LIST:
    copy_original_image(dataset_folder, noise, category, video, image_name, dst_folder)
