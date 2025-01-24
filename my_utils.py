import cv2
import numpy as np
import os
from glob import glob
import keras
import tensorflow as tf
from keras import backend as k

def generate_file_name_by_number(file_number, number_of_digits = 6, prefix = "bin", suffix = ".png"):
    file_name = "error"
    zeroes_to_add = number_of_digits - len(str(file_number))
    
    if zeroes_to_add < 0:
        raise("Error: number of digits lesser than file_number length.")
    else:
        file_name = prefix + zeroes_to_add * "0" + str(file_number) + suffix
        
    return file_name

def mixture_of_two_images(img1_dir, img2_dir, output_dir):
    img1 = cv2.imread(img1_dir)
    img2 = cv2.imread(img2_dir)

    new_img = np.zeros(img1.shape)
    new_img[:,:,0] = img1[:,:,0] # ours blue
    new_img[:,:,2] = img2[:,:,0] # other red

    cv2.imwrite(output_dir, new_img)

def mixture_of_images(imageset1_dir, imageset2_dir, output_dir):
    imageset1 = sorted(glob(os.path.join(imageset1_dir,"*")))
    imageset2 = sorted(glob(os.path.join(imageset2_dir,"*")))

    for img1_dir in imageset1:
        _, img_name = os.path.split(img1_dir)
        output_img_dir = os.path.join(output_dir, img_name)
        img2_dir = os.path.join(imageset2_dir, img_name)
        if img2_dir in imageset2:
            mixture_of_two_images(img1_dir, img2_dir, output_img_dir)

def obtain_information_about_dataset(dataset_name):

    if (dataset_name == "highway"):
        number_of_training_images = 469 #Number of images to obtain initial gaussian models.
        category ="baseline"
        
    if (dataset_name == "office"):
        number_of_training_images = 569 #Number of images to obtain initial gaussian models.
        category ="baseline"
        
    if (dataset_name == "pedestrians"):
        number_of_training_images = 299 #Number of images to obtain initial gaussian models.
        category ="baseline"
        
    if (dataset_name == "PETS2006"):
        number_of_training_images = 299 #Number of images to obtain initial gaussian models.
        category ="baseline"
        
    if (dataset_name == "canoe"):
        number_of_training_images = 799 #Number of images to obtain initial gaussian models.
        print_list = [[2,20], [9,20], [11,20], [15,20], [18,20], [24,20]]
        category ="dynamicBackground"
        
    if (dataset_name == "boats"):
        number_of_training_images = 1899 #Number of images to obtain initial gaussian models.
        print_list = [[4,5], [5,20], [11,36], [19,32], [18,13], [28,13]]
        category ="dynamicBackground"
        
    if (dataset_name == "port_0_17fps"):
        number_of_training_images = 999 #Number of images to obtain initial gaussian models.
        #print_list = [[4,5], [5,20], [11,36], [19,32], [18,13], [28,13]]
        category ="lowFramerate"
        
    if (dataset_name == "overpass"):
        number_of_training_images = 999 #Number of images to obtain initial gaussian models.
        print_list = [[16,23], [18,23], [20,23], [25,13], [25,20], [26,15]]
        category ="dynamicBackground"
        
    if (dataset_name == "streetCornerAtNight"):
        number_of_training_images = 799 #Number of images to obtain initial gaussian models.
        category ="nightVideos"
        
    if (dataset_name == "tramStation"):
        number_of_training_images = 499 #Number of images to obtain initial gaussian models.
        category = "nightVideos"
        
    if (dataset_name == "blizzard"):
        number_of_training_images = 899 #Number of images to obtain initial gaussian models.
        category ="badWeather"
        
    if (dataset_name == "wetSnow"):
        number_of_training_images = 499 #Number of images to obtain initial gaussian models.
        category ="badWeather"
        
    if (dataset_name == "skating"):
        number_of_training_images = 799 #Number of images to obtain initial gaussian models.
        category ="badWeather"
        
    if (dataset_name == "snowFall"):
        number_of_training_images = 799 #Number of images to obtain initial gaussian models.
        category ="badWeather"
        
    if (dataset_name == "fountain01"):
        number_of_training_images = 399 #Number of images to obtain initial gaussian models.
        category ="dynamicBackground"
        
    if (dataset_name == "fountain02"):
        number_of_training_images = 499 #Number of images to obtain initial gaussian models.
        category ="dynamicBackground"
     
    return category, number_of_training_images
    
def cartesian_product2D(a,b):
    return np.array(np.meshgrid(a,b)).T.reshape(-1,2)

def split_in_patches_one_image(image_matrix, patch_height, patch_width):
    return_value = None
    
    if len (image_matrix.shape) == 2: #It is a simple image with one channel.
        image_matrix_height = image_matrix.shape[0]
        image_matrix_width = image_matrix.shape[1]
        return_value = np.zeros([int(image_matrix_height / patch_height) * int(image_matrix_width / patch_width), patch_height, patch_width, 1]) #We create the numpy array to be returned.
        for ii in range(int(image_matrix_height / patch_height)):
            for jj in range(int(image_matrix_width / patch_width)):
                return_value[ii*int(image_matrix_width / patch_width) + jj, :, :, 0] = image_matrix[ii*patch_height:(ii+1)*patch_height, jj*patch_width:(jj+1)*patch_width]
    
    if len (image_matrix.shape) == 3: #It is a image with more than one channel.
        image_matrix_height = image_matrix.shape[0]
        image_matrix_width = image_matrix.shape[1]
        image_matrix_channels = image_matrix.shape[2]

        if (image_matrix_channels == 1): #There is only one channel.
            image_matrix = np.reshape(image_matrix, [image_matrix.shape[0], image_mtrix.shape[1]]) #We reshape.
            return_value = split_in_patches_one_image(image_matrix, patch_height, patch_width)

        if (image_matrix_channels > 1): #There is various channels
            return_value = np.zeros([int(image_matrix_height / patch_height) * int(image_matrix_width / patch_width), patch_height, patch_width, image_matrix_channels]) #We create the numpy array to be returned.
            for ii in range(int(image_matrix_height / patch_height)):
                for jj in range(int(image_matrix_width / patch_width)):
                    return_value[ii*int(image_matrix_width / patch_width) + jj] = image_matrix[ii*patch_height:(ii+1)*patch_height, jj*patch_width:(jj+1)*patch_width, :]

    return return_value
    
def split_in_patches_one_image_with_tessera(image_matrix, patch_side, tessera_initial_point, patches_per_height, patches_per_width):
    return_value = None
    
    if len (image_matrix.shape) == 2: #It is a simple image with one channel.
        return_value = "ERROR, image with only 1 channel."
    
    if len (image_matrix.shape) == 3: #It is a image with more than one channel.
        image_matrix_height = image_matrix.shape[0]
        image_matrix_width = image_matrix.shape[1]
        image_matrix_channels = image_matrix.shape[2]

        if (image_matrix_channels == 1): #There is only one channel.
            return_value = "ERROR, image with only 1 channel."

        if (image_matrix_channels > 1): #There are various channels
            return_value = np.zeros([patches_per_height * patches_per_width, patch_side, patch_side, image_matrix_channels]) #We create the numpy array to be returned.
            for ii in range(patches_per_height):
                for jj in range(patches_per_width):
                    return_value[ii*patches_per_width + jj] = image_matrix[tessera_initial_point[0] + ii*patch_side:tessera_initial_point[0] + (ii+1)*patch_side, tessera_initial_point[1] + jj*patch_side: tessera_initial_point[1] + (jj+1)*patch_side, :]

    return return_value

def split_in_patches_with_windows_one_image(image_matrix, patch_height, patch_width, slicing_height, slicing_width):
    return_value = None
    
    if len (image_matrix.shape) == 2: #It is a simple image with one channel.
        image_matrix_height = image_matrix.shape[0]
        image_matrix_width = image_matrix.shape[1]
        return_value = np.zeros([int((image_matrix_height - patch_height) / slicing_height + 1) * int((image_matrix_width - patch_width) / slicing_width + 1), patch_height, patch_width, 1]) #We create the numpy array to be returned.
        for ii in range(int((image_matrix_height - patch_height) / slicing_height + 1)):
            for jj in range(int((image_matrix_width - patch_width) / slicing_width + 1)):
                return_value[ii*int((image_matrix_width - patch_width) / slicing_width + 1) + jj, :, :, 0] = image_matrix[ii*slicing_height:ii*slicing_height + patch_height, jj*slicing_width:jj*slicing_width + patch_width]
    
    if len (image_matrix.shape) == 3: #It is a image with more than one channel.
        image_matrix_height = image_matrix.shape[0]
        image_matrix_width = image_matrix.shape[1]
        image_matrix_channels = image_matrix.shape[2]

        if (image_matrix_channels == 1): #There is only one channel.
            image_matrix = np.reshape(image_matrix, [image_matrix.shape[0], image_mtrix.shape[1]]) #We reshape.
            return_value = split_in_patches_with_windows_one_image(image_matrix, patch_height, patch_width, slicing_height, slicing_width)

        if (image_matrix_channels > 1): #There is various channels
            return_value = np.zeros([int((image_matrix_height - patch_height) / slicing_height + 1) * int((image_matrix_width - patch_width) / slicing_width + 1), patch_height, patch_width, image_matrix_channels]) #We create the numpy array to be returned.
            for ii in range(int((image_matrix_height - patch_height) / slicing_height + 1)):
                for jj in range(int((image_matrix_width - patch_width) / slicing_width + 1)):
                    return_value[ii*int((image_matrix_width - patch_width) / slicing_width + 1) + jj] = image_matrix[ii*slicing_height:ii*slicing_height + patch_height, jj*slicing_width:jj*slicing_width + patch_width, :]

    return return_value


def split_in_patches_various_images(images_matrix, patch_height, patch_width):
    images_matrix_size = images_matrix.shape[0]
    images_matrix_height = images_matrix.shape[1]
    images_matrix_width = images_matrix.shape[2]
    if len (images_matrix.shape) > 3:
        images_matrix_channel = images_matrix.shape[3]
    else:
        images_matrix_channel = 1
    return_value = np.zeros([images_matrix_size * int(images_matrix_height / patch_height) * int(images_matrix_width / patch_width), patch_height, patch_width, images_matrix_channel])
    for ii in range(images_matrix_size):
        return_value[ii * int(images_matrix_height / patch_height) * int(images_matrix_width / patch_width): (ii+1) * int(images_matrix_height / patch_height) * int(images_matrix_width / patch_width)] = split_in_patches_one_image(images_matrix[ii], patch_height, patch_width)

    return return_value

def split_in_patches_with_windows_various_images(images_matrix, patch_height, patch_width, slicing_height, slicing_width):
    images_matrix_size = images_matrix.shape[0]
    images_matrix_height = images_matrix.shape[1]
    images_matrix_width = images_matrix.shape[2]
    if len (images_matrix.shape) > 3:
        images_matrix_channel = images_matrix.shape[3]
    else:
        images_matrix_channel = 1
    return_value = np.zeros([images_matrix_size * int((images_matrix_height - patch_height) / slicing_height + 1) * int((images_matrix_width - patch_width) / slicing_width + 1), patch_height, patch_width, images_matrix_channel])
    for ii in range(images_matrix_size):
        return_value[ii * int((images_matrix_height - patch_height) / slicing_height + 1) * int((images_matrix_width - patch_width) / slicing_width + 1): (ii + 1) * int((images_matrix_height - patch_height) / slicing_height + 1) * int((images_matrix_width - patch_width) / slicing_width + 1)] = split_in_patches_with_windows_one_image(images_matrix[ii], patch_height, patch_width, slicing_height, slicing_width)

    return return_value

def reconstruct_from_patches(images_patch_matrix, original_images_height, original_images_width):
    return_value = None

    images_patch_matrix_size = images_patch_matrix.shape[0]
    images_patch_matrix_height = images_patch_matrix.shape[1]
    images_patch_matrix_width = images_patch_matrix.shape[2]

    patches_in_row = int(original_images_width / images_patch_matrix_width)
    patches_in_column = int(original_images_height / images_patch_matrix_height)
    original_images_number = images_patch_matrix_size/(patches_in_column * patches_in_row)

    if len (images_patch_matrix.shape) == 3: #There is only one channel.

        return_value = np.zeros([original_images_number, patches_in_column * images_patch_matrix_height, patches_in_row * images_patch_matrix_width])
        
        for ii in range(original_images_number):
            for jj in range(patches_in_column):
                for kk in range(patches_in_row):
                    return_value[ii, jj * images_patch_matrix_height : (jj+1) * images_patch_matrix_height, kk * images_patch_matrix_width : (kk+1) * images_patch_matrix_width] = images_patch_matrix[patches_in_column * patches_in_row * ii + patches_in_row * jj + kk, :, :]

    if len (images_patch_matrix.shape) == 4: #It could be more than one channel.
        
        if images_patch_matrix.shape[3] == 1: #There is only one channel
            images_patch_matrix = np.reshape(images_patch_matrix, [images_patch_matrix_size, images_patch_matrix_height, images_patch_matrix_width])
            return_value = reconstruct_from_patches(images_patch_matrix. original_images_height, original_images_width)

        if images_patch_matrix.shape[3] > 1: #There is more than one channel.
            return_value = np.zeros([original_images_number, patches_in_column * images_patch_matrix_height, patches_in_row * images_patch_matrix_width, images_patch_matrix.shape[3]])

            for ii in range(original_images_number):
                for jj in range(patches_in_column):
                    for kk in range(patches_in_row):
                        return_value[ii, jj * images_patch_matrix_height : (jj+1) * images_patch_matrix_height, kk * images_patch_matrix_width : (kk+1) * images_patch_matrix_width, :] = images_patch_matrix[patches_in_column * patches_in_row * ii + patches_in_row * jj + kk, :, :, :]

    return return_value

def draw_rectangles_over_image(image_matrix, matrix_to_draw, value_to_draw, padding_shape=[0,0,0], color_to_draw = np.array([255.,0,0])):
    #Function to draw rectangles over image_matrix where matrix_to_draw indicates.
    output_image = image_matrix
    patches_height = image_matrix.shape[0]/matrix_to_draw.shape[0]
    patches_width = image_matrix.shape[1]/matrix_to_draw.shape[1]
    
    for ii in range(matrix_to_draw.shape[0]):
        for jj in range(matrix_to_draw.shape[1]):
            if (matrix_to_draw[ii][jj] >= value_to_draw):
                output_image = draw_a_rectangle_over_image(output_image, patches_height, patches_width, ii, jj, padding_shape)

    return output_image

def draw_a_rectangle_over_image(image_matrix, patches_height, patches_width, height, width, padding_shape, color_to_draw = np.array([255.,0,0])):
    output_matrix = image_matrix

    output_matrix[padding_shape[0]/2 + patches_height*height:padding_shape[0]/2 + patches_height*(height+1), padding_shape[1]/2 + patches_width*width, :] = color_to_draw
    output_matrix[padding_shape[0]/2 + patches_height*height:padding_shape[0]/2 + patches_height*(height+1), padding_shape[1]/2 + patches_width*(width+1)-1, :] = color_to_draw
    output_matrix[padding_shape[0]/2 + patches_height*height, padding_shape[1]/2 + patches_width*width:padding_shape[1]/2 + patches_width*(width+1), :] = color_to_draw
    output_matrix[padding_shape[0]/2 + patches_height*(height+1)-1, padding_shape[1]/2 + patches_width*width:padding_shape[1]/2 + patches_width*(width+1), :] = color_to_draw            
    
    return output_matrix

def to_output_array_file(matrix, output_path, mode):
    
    print(matrix)
    output_file = open(output_path, "w")
    text_matrix = ""
    print(matrix.shape)
    if (mode == "matlab"):
        if (len(matrix.shape) == 3): #Matrix has three dimensions.
            for ii in range(matrix.shape[0]):
                if (ii > 0): #This is not the first element to introduce.
                    text_matrix = text_matrix + ";"
                for jj in range(matrix.shape[1]):
                    if (jj > 0): #This is not the first element to introduce.
                        text_matrix = text_matrix + ","
                    for kk in range(matrix.shape[2]):
                        if (kk > 0): #This is not the first element to introduce.
                            text_matrix = text_matrix + " "
                        text_matrix = text_matrix + str(matrix[ii][jj][kk])                   

        if (len(matrix.shape) == 4): #Matrix has four dimensions.
            for ii in range(matrix.shape[0]):
                if (ii > 0): #This is not the first element to introduce.
                    text_matrix = text_matrix + "|"
                for jj in range(matrix.shape[1]):
                    if (jj > 0): #This is not the first element to introduce.
                        text_matrix = text_matrix + ";"
                    for kk in range(matrix.shape[2]):
                        if (kk > 0): #This is not the first element to introduce.
                           text_matrix = text_matrix + ","
                        for mm in range(matrix.shape[3]):
                           if (mm > 0): #This is not the first element to introduce.
                               text_matrix = text_matrix + " " 
                           text_matrix = text_matrix + str(matrix[ii][jj][kk][mm])

    if (mode == "python"):
        if (len(matrix.shape) == 3): #Matrix has three dimensions.
            text_matrix = text_matrix + "["
            for ii in range(matrix.shape[0]):
                text_matrix = text_matrix + "["
                for jj in range(matrix.shape[1]):
                    if (jj > 0): #This is not the first element to introduce.
                        text_matrix = text_matrix + " "
                    text_matrix = text_matrix + str(matrix[ii][jj])
                text_matrix = text_matrix + "]"
            text_matrix = text_matrix + "]"                    

        if (len(matrix.shape) == 4): #Matrix has four dimensions.
            text_matrix = text_matrix + "["
            for ii in range(matrix.shape[0]):
                text_matrix = text_matrix + "["
                for jj in range(matrix.shape[2]):
                    text_matrix = text_matrix + "["
                    for kk in range(matrix.shape[2]):
                        if (kk > 0): #This is not the first element to introduce.
                            text_matrix = text_matrix + " "
                        text_matrix = text_matrix + str(matrix[ii][jj][kk])
                    text_matrix = text_matrix + "]"
                text_matrix = text_matrix + "]"
            text_matrix = text_matrix + "]"

    print(text_matrix)
    output_file.write(text_matrix)
    output_file.close()

def create_segmented_image_from_foreground_matrix(foreground_matrix, patch_height, patch_width, minimum_value_to_draw, original_image_height = None, original_image_width = None):

    if (original_image_height is None or original_image_width is None):
        original_image_height = foreground_matrix.shape[0]
        original_image_width = foreground_matrix.shape[1]
        segmented_image_matrix = np.zeros([original_image_height*patch_height, original_image_width*patch_width])
    else:
        segmented_image_matrix = np.zeros([original_image_height, original_image_width])
    
    for ii in range(foreground_matrix.shape[0]):
        for jj in range(foreground_matrix.shape[1]):
            if (foreground_matrix[ii,jj] >= minimum_value_to_draw):
                segmented_image_matrix[patch_height * ii:patch_height * (ii+1), patch_width * jj:patch_width * (jj+1)] = 255.

    return segmented_image_matrix

def create_segmented_grayscale_image_from_foreground_matrix(foreground_matrix, patch_height, patch_width, original_image_height = None, original_image_width = None):

    if (original_image_height is None or original_image_width is None):
        original_image_height = foreground_matrix.shape[0]
        original_image_width = foreground_matrix.shape[1]
        segmented_image_matrix = np.zeros([original_image_height*patch_height, original_image_width*patch_width])
    else:
        segmented_image_matrix = np.zeros([original_image_height, original_image_width])
    
    for ii in range(foreground_matrix.shape[0]):
        for jj in range(foreground_matrix.shape[1]):
            segmented_image_matrix[patch_height * ii:patch_height * (ii+1), patch_width * jj:patch_width * (jj+1)] = 255. * foreground_matrix[ii,jj]

    return segmented_image_matrix


def convert_foreground_image_matrix_from_pixel_to_region(input_matrix_image, patch_height, patch_width, minimum_percentage_to_foreground):

    original_image_height = input_matrix_image.shape[0]
    original_image_width = input_matrix_image.shape[1]
    patches_per_height = int(original_image_height/patch_height)
    patches_per_width = int(original_image_width/patch_width)

    foreground_matrix = np.zeros([patches_per_height, patches_per_width])

    for ii in range(0, patches_per_height):
        for jj in range(0, patches_per_width):
            patch_matrix = input_matrix_image[ii*patch_height:(ii+1)*patch_height, jj*patch_width:(jj+1)*patch_width]
            total_foreground_pixels_in_patch = np.sum(np.all(patch_matrix==255., axis=2))
            if ((total_foreground_pixels_in_patch/(patch_height * patch_width * 1.)) >= minimum_percentage_to_foreground):
                foreground_matrix[ii, jj] = 1
    
    return create_segmented_image_from_foreground_matrix(foreground_matrix, patch_height, patch_width, 1, original_image_height, original_image_width)

def encode_images_from_folder(encoder_model_dir, dataset_dir, patch_height=16, patch_width=16, L=64):


    ###################################
    #os.environ["CUDA_VISIBLE_DEVICES"]="0" #To force tensorflow to only see one GPU.
    # TensorFlow wizardry
    #config = tf.ConfigProto()
     
    # Don't pre-allocate memory; allocate as-needed
    #config.gpu_options.allow_growth = True
     
    # Only allow a total of half the GPU memory to be allocated
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5
     
    # Create a session with the above options specified.
    #k.tensorflow_backend.set_session(tf.Session(config=config))
    ###################################

    print ("Loading encoder model...")
    encoder_model = keras.models.load_model(encoder_model_dir)

    print ("Loading data...")
    dataset_dirs = os.listdir(dataset_dir)
    dataset_dirs = sorted(dataset_dirs)
    original_dataset_size = len(dataset_dirs)

    #We get an image from the dataset to obtain images height and width.
    if original_dataset_size > 0:
        img_name = dataset_dirs[0]
        img_dir = os.path.join(dataset_dir, img_name)
        img = cv2.imread( img_dir )
        original_image_height = img.shape[0]
        original_image_width = img.shape[1]

    original_dataset = np.zeros([original_dataset_size, original_image_height, original_image_width, 3])

    index = 0
    for img_name in dataset_dirs:
        img_dir = os.path.join(dataset_dir, img_name)
        img = cv2.imread( img_dir )
        original_dataset[index] = img.astype( float )
        index += 1

    original_dataset /= 255.

    print ("Transforming original data to patches...")
    patched_dataset = split_in_patches_various_images(original_dataset, patch_height, patch_width)

    patched_dataset_reshaped = np.reshape(patched_dataset, [patched_dataset.shape[0], patch_height*patch_width*3])

    print ("Encoding...")
    predicted_patches = encoder_model.predict(patched_dataset_reshaped)

    return predicted_patches

def extract_gaussian_parameters_from_encoded_images(encoder_model_dir, dataset_dir, L):

    predicted_patches = encode_images_from_folder(encoder_model_dir, dataset_dir, 16, 16, L)
    R = predicted_patches.shape[0]
    SIGMA2 = np.zeros([L])
    MU = np.zeros([L])

    for kk in range(L):   #For each component...
        MU[kk] = np.median(predicted_patches[:, kk])
        SIGMA2[kk] = np.sum(np.power(predicted_patches[:, kk] - MU[kk], 2)) / (R - 1)


    return MU, SIGMA2
    
def reorder_dataset_to_input_subfolder(dataset_folder):
    
    dataset_dirs = os.listdir(dataset_dir)
    dataset_dirs = sorted(dataset_dirs)
    
    index = 0
    for img_name in dataset_Dirs:
        img_dir = os.path.join(dataset_dir, img_name)
        img = cv2.imread(img_dir)
        cv2.imwrite(os.path.join(dataset_dir,"input", img_name), img)

def reorder_all_dataset_to_input_subfolder(dataset_folder):

    dataset_list = ["pedestrians", "canoe", "boats", "port_0_17fps", "fountain01", "fountain02", "overpass"]

    for dataset_name in dataset_list:
	    if (dataset_name == "level_crossing"):
	        N = 150 #Number of images to obtain initial gaussian models.
	        dataset_dir = os.path.join(dataset_source_folder, dataset_name)
	        dataset_dir = os.path.join(dataset_dir, "all")
	        dataset_subfolder = ""
	    if (dataset_name == "water_surface"):
	        N = 30 #Number of images to obtain initial gaussian models.
	        dataset_dir = os.path.join(dataset_source_folder, dataset_name)
	        dataset_dir = os.path.join(dataset_dir, "all")
	        dataset_subfolder = ""
	    if (dataset_name == "highway"):
	        N = 469 #Number of images to obtain initial gaussian models.
	        dataset_dir = os.path.join(dataset_source_folder, "baseline", dataset_name)
	        dataset_dir = os.path.join(dataset_dir, "input")
	        dataset_subfolder = "baseline"
	    if (dataset_name == "office"):
	        N = 569 #Number of images to obtain initial gaussian models.
	        dataset_dir = os.path.join(dataset_source_folder, "baseline", dataset_name)
	        dataset_dir = os.path.join(dataset_dir, "input")
	        dataset_subfolder = "baseline"
	    if (dataset_name == "pedestrians"):
	        N = 299 #Number of images to obtain initial gaussian models.
	        dataset_dir = os.path.join(dataset_source_folder, "baseline", dataset_name)
	        dataset_dir = os.path.join(dataset_dir, "input")
	        dataset_subfolder = "baseline"
	    if (dataset_name == "PETS2006"):
	        N = 299 #Number of images to obtain initial gaussian models.
	        dataset_dir = os.path.join(dataset_source_folder, "baseline", dataset_name)
	        dataset_dir = os.path.join(dataset_dir, "input")
	        dataset_subfolder = "baseline"
	    if (dataset_name == "canoe"):
	        N = 799 #Number of images to obtain initial gaussian models.
	        print_list = [[2,20], [9,20], [11,20], [15,20], [18,20], [24,20]]
	        dataset_dir = os.path.join(dataset_source_folder, "dynamicBackground", dataset_name)
	        dataset_dir = os.path.join(dataset_dir, "input")
	        dataset_subfolder = "dynamicBackground"
	    if (dataset_name == "boats"):
	        N = 1899 #Number of images to obtain initial gaussian models.
	        print_list = [[4,5], [5,20], [11,36], [19,32], [18,13], [28,13]]
	        dataset_dir = os.path.join(dataset_source_folder, "dynamicBackground", dataset_name)
	        dataset_dir = os.path.join(dataset_dir, "input")
	        dataset_subfolder = "dynamicBackground"
	    if (dataset_name == "port_0_17fps"):
	        N = 999 #Number of images to obtain initial gaussian models.
	        #print_list = [[4,5], [5,20], [11,36], [19,32], [18,13], [28,13]]
	        dataset_dir = os.path.join(dataset_source_folder, "lowFramerate", dataset_name)
	        dataset_dir = os.path.join(dataset_dir, "input")
	        dataset_subfolder = "lowFramerate"
	    if (dataset_name == "overpass"):
	        N = 999 #Number of images to obtain initial gaussian models.
	        print_list = [[16,23], [18,23], [20,23], [25,13], [25,20], [26,15]]
	        dataset_dir = os.path.join(dataset_source_folder, "dynamicBackground", dataset_name)
	        dataset_dir = os.path.join(dataset_dir, "input")
	        dataset_subfolder = "dynamicBackground"
	    if (dataset_name == "streetCornerAtNight"):
	        N = 799 #Number of images to obtain initial gaussian models.
	        dataset_dir = os.path.join(dataset_source_folder, "nightVideos", dataset_name)
	        dataset_dir = os.path.join(dataset_dir, "input")
	        dataset_subfolder = "nightVideos"
	    if (dataset_name == "tramStation"):
	        N = 499 #Number of images to obtain initial gaussian models.
	        dataset_dir = os.path.join(dataset_source_folder, "nightVideos", dataset_name)
	        dataset_dir = os.path.join(dataset_dir, "input")
	        dataset_subfolder = "nightVideos"
	    if (dataset_name == "blizzard"):
	        N = 899 #Number of images to obtain initial gaussian models.
	        dataset_dir = os.path.join(dataset_source_folder, "badWeather", dataset_name)
	        dataset_dir = os.path.join(dataset_dir, "input")
	        dataset_subfolder = "badWeather"
	    if (dataset_name == "wetSnow"):
	        N = 499 #Number of images to obtain initial gaussian models.
	        dataset_dir = os.path.join(dataset_source_folder, "badWeather", dataset_name)
	        dataset_dir = os.path.join(dataset_dir, "input")
	        dataset_subfolder = "badWeather"
	    if (dataset_name == "fountain01"):
	        N = 399 #Number of images to obtain initial gaussian models.
	        dataset_dir = os.path.join(dataset_source_folder, "dynamicBackground", dataset_name)
	        dataset_dir = os.path.join(dataset_dir, "input")
	        dataset_subfolder = "dynamicBackground"
	    if (dataset_name == "fountain02"):
	        N = 499 #Number of images to obtain initial gaussian models.
	        dataset_dir = os.path.join(dataset_source_folder, "dynamicBackground", dataset_name)
	        dataset_dir = os.path.join(dataset_dir, "input")
	        dataset_subfolder = "dynamicBackground"
    
        
    
