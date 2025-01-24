import cv2
import numpy as np
import os
from glob import glob
from PIL import Image
from random import shuffle

def mixture_of_two_images(img1_dir, img2_dir, output_dir):
    img1 = cv2.imread(img1_dir)
    img2 = cv2.imread(img2_dir)

    new_img = np.zeros(img1.shape)
    new_img[:,:,0] = img1[:,:,0] # first blue
    new_img[:,:,2] = img2[:,:,0] # second red

    cv2.imwrite(output_dir, new_img)

#Function to create a mixture of two foreground segmentation images set. 
def mixture_of_images(imageset1_dir, imageset2_dir, output_dir):
    imageset1 = sorted(glob(os.path.join(imageset1_dir,"*")))
    imageset2 = sorted(glob(os.path.join(imageset2_dir,"*")))

    for img1_dir in imageset1:
        _, img_name = os.path.split(img1_dir)
        output_img_dir = os.path.join(output_dir, img_name)
        img2_dir = os.path.join(imageset2_dir, img_name)
        if img2_dir in imageset2:
            mixture_of_two_images(img1_dir, img2_dir, output_img_dir)
            
def split_in_patches_one_image(image_matrix, patch_height, patch_width, check_fitting = True):
    return_value = None
    
    if len (image_matrix.shape) == 2: #It is a simple image with one channel.
        print("Error. No images with only 1 channel allowed.")
        quit()
    if len (image_matrix.shape) == 3: #It is a image with more than one channel.
        image_matrix_height = image_matrix.shape[0]
        image_matrix_width = image_matrix.shape[1]
        image_matrix_channels = image_matrix.shape[2]

        if (image_matrix_channels == 1): #There is only one channel.
            print("Error. No images with only 1 channel allowed.")
            quit()

        if (image_matrix_channels > 1): #There are various channels
            patches_in_column = int(image_matrix_height / patch_height)
            patches_in_row = int(image_matrix_width / patch_width)

            if (check_fitting):
                if patches_in_column != float(image_matrix_height) / patch_height:
                    patches_in_column += 1
                if patches_in_row != float(image_matrix_width) / patch_width:
                    patches_in_row += 1
            
            return_value = np.zeros([patches_in_column * patches_in_row, patch_height, patch_width, image_matrix_channels]) #We create the numpy array to be returned.
            for ii in range(patches_in_column):
                for jj in range(patches_in_row):
                    if (ii < patches_in_column-1) and (jj < patches_in_row-1):
                        return_value[ii*patches_in_row + jj] = image_matrix[ii*patch_height:(ii+1)*patch_height, jj*patch_width:(jj+1)*patch_width, :]
                    if (ii == patches_in_column-1) and (jj < patches_in_row-1):
                        return_value[ii*patches_in_row + jj] = image_matrix[-1*patch_height:, jj*patch_width:(jj+1)*patch_width, :]
                    if (ii < patches_in_column-1) and (jj == patches_in_row-1):
                        return_value[ii*patches_in_row + jj] = image_matrix[ii*patch_height:(ii+1)*patch_height, -1*patch_width:, :]
                    if (ii == patches_in_column-1) and (jj == patches_in_row-1):
                        return_value[ii*patches_in_row + jj] = image_matrix[-1*patch_height:, -1*patch_width:, :]

    return return_value

def split_in_patches_with_windows_one_image(image_matrix, patch_height, patch_width, slicing_height, slicing_width):
    return_value = None
    
    if len (image_matrix.shape) == 2: #It is a simple image with one channel.
        print("Error. No images with only 1 channel allowed.")
        quit()

    if len (image_matrix.shape) == 3: #It is a image with more than one channel.
        image_matrix_height = image_matrix.shape[0]
        image_matrix_width = image_matrix.shape[1]
        image_matrix_channels = image_matrix.shape[2]

        if (image_matrix_channels == 1): #There is only one channel.
            print("1-channel images are not allowed.")
            quit()

        if (image_matrix_channels > 1): #There is various channels
            return_value = np.zeros([int((image_matrix_height - patch_height) / slicing_height + 1) * int((image_matrix_width - patch_width) / slicing_width + 1), patch_height, patch_width, image_matrix_channels]) #We create the numpy array to be returned.
            for ii in range(int((image_matrix_height - patch_height) / slicing_height + 1)):
                for jj in range(int((image_matrix_width - patch_width) / slicing_width + 1)):
                    return_value[ii*int((image_matrix_width - patch_width) / slicing_width + 1) + jj] = image_matrix[ii*slicing_height:ii*slicing_height + patch_height, jj*slicing_width:jj*slicing_width + patch_width, :]

    return return_value

def split_in_patches_one_image_with_tessera(image_matrix, patch_side, tessera_initial_point, patches_per_height, patches_per_width):
    return_value = None
    
    if len (image_matrix.shape) == 2: #It is a simple image with one channel.
        print("Error. No images with only 1 channel allowed.")
        quit()
    
    if len (image_matrix.shape) == 3: #It is a image with more than one channel.
        image_matrix_height = image_matrix.shape[0]
        image_matrix_width = image_matrix.shape[1]
        image_matrix_channels = image_matrix.shape[2]

        if (image_matrix_channels == 1): #There is only one channel.
            print("Error. No images with only 1 channel allowed.")
            quit()

        if (image_matrix_channels > 1): #There are various channels
            return_value = np.zeros([patches_per_height * patches_per_width, patch_side, patch_side, image_matrix_channels]) #We create the numpy array to be returned.
            for ii in range(patches_per_height):
                for jj in range(patches_per_width):
                    return_value[ii*patches_per_width + jj] = image_matrix[tessera_initial_point[0] + ii*patch_side:tessera_initial_point[0] + (ii+1)*patch_side, tessera_initial_point[1] + jj*patch_side: tessera_initial_point[1] + (jj+1)*patch_side, :]

    return return_value
    
def split_in_patches_one_image_with_tesseras(image_matrix, patch_side, tesseras_initial_points, patches_per_height, patches_per_width):
    return_value = None
    for tessera_initial_point in tesseras_initial_points:
        if return_value is None:
            return_value = split_in_patches_one_image_with_tessera(image_matrix, patch_side, tessera_initial_point, patches_per_height, patches_per_width)
        else:
            return_value = np.concatenate((return_value, split_in_patches_one_image_with_tessera(image_matrix, patch_side, tessera_initial_point, patches_per_height, patches_per_width)))
    return return_value

def split_in_patches_various_images(images_matrix, patch_height, patch_width, check_fitting = True):
    images_matrix_size = images_matrix.shape[0]
    images_matrix_height = images_matrix.shape[1]
    images_matrix_width = images_matrix.shape[2]

    patches_in_column = int(images_matrix_height / patch_height)
    patches_in_row = int(images_matrix_width / patch_width)

    if (check_fitting):
        if patches_in_column != float(images_matrix_height) / patch_height:
            patches_in_column += 1
        if patches_in_row != float(images_matrix_width) / patch_width:
            patches_in_row += 1

    if len (images_matrix.shape) > 3:
        images_matrix_channel = images_matrix.shape[3]
    else:
        print("Error. No images with only 1 channel allowed.")
        quit()
    return_value = np.zeros([images_matrix_size * patches_in_column * patches_in_row, patch_height, patch_width, images_matrix_channel])
    for ii in range(images_matrix_size):
        return_value[ii * patches_in_column * patches_in_row: (ii+1) * patches_in_column * patches_in_row] = split_in_patches_one_image(images_matrix[ii], patch_height, patch_width, check_fitting=check_fitting)

    return return_value

def split_in_patches_with_windows_various_images(images_matrix, patch_height, patch_width, slicing_height, slicing_width):
    images_matrix_size = images_matrix.shape[0]
    images_matrix_height = images_matrix.shape[1]
    images_matrix_width = images_matrix.shape[2]
    if len (images_matrix.shape) > 3:
        images_matrix_channel = images_matrix.shape[3]
    else:
        print("Error. No images with only 1 channel allowed.")
        quit()
    return_value = np.zeros([images_matrix_size * int((images_matrix_height - patch_height) / slicing_height + 1) * int((images_matrix_width - patch_width) / slicing_width + 1), patch_height, patch_width, images_matrix_channel])
    for ii in range(images_matrix_size):
        return_value[ii * int((images_matrix_height - patch_height) / slicing_height + 1) * int((images_matrix_width - patch_width) / slicing_width + 1): (ii + 1) * int((images_matrix_height - patch_height) / slicing_height + 1) * int((images_matrix_width - patch_width) / slicing_width + 1)] = split_in_patches_with_windows_one_image(images_matrix[ii], patch_height, patch_width, slicing_height, slicing_width)

    return return_value
    
#Function to reconstruct a set of images from patches.
#images_patch_matrix : numpy matrix with size LxNxM where Lis the number of patches and NxM represents each patch.
#original_images_height : int to represent original image height
#original_images_height : int to represent original image width
def reconstruct_from_patches(images_patch_matrix, original_images_height, original_images_width, check_fitting = True):
    return_value = None

    images_patch_matrix_size = images_patch_matrix.shape[0]
    images_patch_matrix_height = images_patch_matrix.shape[1]
    images_patch_matrix_width = images_patch_matrix.shape[2]

    patches_in_row = int(original_images_width / images_patch_matrix_width)
    patches_in_column = int(original_images_height / images_patch_matrix_height)
    original_images_number = int(images_patch_matrix_size/(patches_in_column * patches_in_row))

    if check_fitting:
        width_fit = patches_in_row == (float(original_images_width) / images_patch_matrix_width)         # To know if original width is divisble by patch width
        height_fit = patches_in_column == (float(original_images_height) / images_patch_matrix_height)   # # To know if original height is divisble by patch width
    else:
        width_fit = True
        height_fit = True

    if len (images_patch_matrix.shape) == 3: #There is only one channel.

        print("Error: Undefined reconstructo from patches for one image.")
        quit()

    if len (images_patch_matrix.shape) == 4: #It could be more than one channel.
        
        if images_patch_matrix.shape[3] == 1: #There is only one channel
            print("Error: Undefined reconstructo from patches for one image.")
            quit()

        if images_patch_matrix.shape[3] > 1: #There is more than one channel.
            return_value = np.zeros([original_images_number, original_images_height, original_images_width, images_patch_matrix.shape[3]])
            patches_in_column_range = patches_in_column
            patches_in_row_range = patches_in_row
            if not height_fit:
                patches_in_column_range += 1
            if not width_fit:
                patches_in_row_range += 1
            
            for ii in range(original_images_number):
                for jj in range(patches_in_column_range):
                    for kk in range(patches_in_row_range):
                        if jj < patches_in_column and kk < patches_in_row:
                            return_value[ii, jj * images_patch_matrix_height : (jj+1) * images_patch_matrix_height, kk * images_patch_matrix_width : (kk+1) * images_patch_matrix_width, :] = images_patch_matrix[patches_in_column_range * patches_in_row_range * ii + patches_in_row_range * jj + kk, :, :, :]
                        if jj < patches_in_column and kk == patches_in_row:
                            return_value[ii, jj * images_patch_matrix_height : (jj+1) * images_patch_matrix_height, -1*images_patch_matrix_width: , :] = images_patch_matrix[patches_in_column_range * patches_in_row_range * ii + patches_in_row_range * jj + kk, :, :, :]
                        if jj == patches_in_column and kk < patches_in_row:
                            return_value[ii, -1*images_patch_matrix_height:, kk * images_patch_matrix_width : (kk+1) * images_patch_matrix_width, :] = images_patch_matrix[patches_in_column_range * patches_in_row_range * ii + patches_in_row_range * jj + kk, :, :, :]
                        if jj == patches_in_column and kk == patches_in_row:
                            return_value[ii, -1*images_patch_matrix_height:, -1*images_patch_matrix_width:, :] = images_patch_matrix[patches_in_column_range * patches_in_row_range * ii + patches_in_row_range * jj + kk, :, :, :]
    return return_value
    
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
    
def grayscale_all_images_in_folder(input_path, output_path):
    images_path = sorted(glob(input_path))
    if not os.path.isdir(output_path):                                                              # We check if output path exists.
        os.makedirs(output_path)                                                                    # We create path if it does not exist.
    
    for image_path in images_path:
        print(image_path)
        image = cv2.imread(image_path)                                                              # We read the image
        image_path, image_name = os.path.split(image_path)                                          # We obtain the image name.
        output_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                                      # We turn image to grayscale.
        output_image_path = os.path.join(output_path, image_name)                                   # We generate output image path.
        cv2.imwrite(output_image_path, output_image)                                                # We write the image.

def resize_image(numpy_array_image, new_shape):                             # Function to resize an image array.
    im = Image.fromarray(numpy_array_image)
    im.resize(new_shape)
    resized_image_numpy_array = np.array(im)
    return resized_image_numpy_array

def get_normalized_images_from_folder(folder_path, max_number_of_images_used_for_training_and_validation = None, validation_split = 0.2, img_resize_shape = None):
    print("Reading data from folder " + folder_path)

    print("Loading data.")
    images_path = glob(os.path.join(folder_path,'*'))

    if not max_number_of_images_used_for_training_and_validation is None:
        images_path = images_path[:max_number_of_images_used_for_training_and_validation]
    
    shuffle(images_path)

    train_paths = images_path[:-int(len(images_path)*validation_split)]
    validation_paths = images_path[-int(len(images_path)*validation_split):]

    print("Loading training data.") 

    train_images_in_RAM = []
    share_loaded = -1
    total_len_train = len(train_paths)
    for i in range(total_len_train):
        if share_loaded < int(100*i/total_len_train):
            share_loaded=int(100*i/total_len_train)
            print(str(share_loaded)+"% loaded.")
        if img_resize_shape is None:
            normalized_img = cv2.imread(train_paths[i])/255.                        # We load de image and normalize it.
        else:
            img = cv2.imread(train_paths[i])                                        # We load the image.
            resized_image = resize_image(img, img_resize_shape)             # We resize the image.
            normalized_img = resized_image/255.                                 # We normalize it.
        train_images_in_RAM.append(normalized_img)                                  # We append the normalized image to the list.

    print("Loading validation data.")

    validation_images_in_RAM = []
    share_loaded = -1
    total_len_validation = len(validation_paths)
    for i in range(total_len_validation):
        if share_loaded < int(100*i/total_len_validation):
            share_loaded=int(100*i/total_len_validation)
            print(str(share_loaded)+"% loaded.")
        if img_resize_shape is None:
            normalized_img = cv2.imread(validation_paths[i])/255.                   # We load de image and normalize it.
        else:
            img = cv2.imread(validation_paths[i])                                   # We load the image.
            resized_image = resize_image(img, img_resize_shape)         # We resize the image.
            normalized_img = resized_image/255.                                     # We normalize it.
        validation_images_in_RAM.append(normalized_img)                             # We append the normalized image to the list.

    print("Train images:")
    print(len(train_images_in_RAM))
    print("Validation images:")
    print(len(validation_images_in_RAM))

    return train_images_in_RAM, validation_images_in_RAM
