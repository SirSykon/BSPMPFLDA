import numpy as np
import sys
import cv2
import os
import math
import time

import image_utils                                     # pylint: disable=import-error
import noise_utils
import math_utils

#import matplotlib.pyplot as plt

class Background_model(object):
    C = -1                                                  # foreground threshold.
    K = -1                                                  # SIGMA2 multiplier to serve as threshold.
    T = -1                                                  # Number of tesseras. This number should be 2^n form.
    ALPHA = -1                                              # Background model update step size.
    
    encoder = None                                          # encoder object.
    narrowest_layer_size = -1                               # narrowest layer size. It must bee qual to encoder output layer.
    padding_value = -1                                      # Value to serve as paddding to increase image size in order to have margin to get patches (example: we have a 16x16 patch but we segmented with 8x8 resolution, it is needed 8 as padding).
    SIGMA2 = None                                           # variance values for each component for each patch.
    MU = None                                               # average values for each component for each patch.
    SAMPLEVARIANCE = None                                   # Sample variance
    
    MU2 = None                                              # Auciliar value for online calculation of variance and average.
    
    last_segmented_image = None                             # last segmented image.
    tesseras_images = None                                  # last tesseras image set.

    original_image_height = -1                              # Image height.
    original_image_width = -1                               # Image Width.
    original_image_height_with_padding = -1                 # Image height with padding.
    original_image_width_with_padding = -1                  # Image width with padding.
    tessera_height = -1                                     # Tessera width.
    tessera_width = -1                                      # Tessera height.
    patch_side = -1                                         # Patch side size.
    patches_per_image = -1                                  # Number of patches per image.
    patches_per_tessera = -1                                # Number of patches per tessera.
    patches_per_width = -1                                  # Number of patches along width.
    patches_per_height = -1                                 # Number of patches along height.
    number_of_images_for_training = -1                      # Number of images that will be used to initialize the background model.
    extra_padding_height_to_cover_the_last_patch = -1       # Extra padding to height that must be added to fill the last patch if needed.
    extra_padding_width_to_cover_the_last_patch = -1        # Extra padding to width that must be added to fill the last patch if needed.
    
    tesseras_initial_points = None                          # Vector of initial tesseras point.
    
    training_done = False                                   # Control value to know if training is ended.
    data_initialized = False                                # Control value to know if data has been initialized.

    processed_images = -1

    def __init__(self, encoder=None, patch_side = -1, number_of_images_for_training = -1, C = -1, K = -1, ALPHA = -1, T = -1):            # Class to represent the background model base class for Foreground detection
        self.encoder = encoder
        self.narrowest_layer_size = self.encoder.get_narroest_layer_size()
        self.patch_side = patch_side
        self.padding_value = self.patch_side
        self.training_done = False
        self.data_initialized = False
        self.number_of_images_for_training = number_of_images_for_training
        self.processed_images = 0
        self.C=C
        self.K=K
        self.ALPHA = ALPHA
        self.T=T     
        
        n_tesseras_side = int(math.sqrt(self.T))                                                  # T must match 2^n so we can obtain n.
        point_step = patch_side/n_tesseras_side
        if point_step % 2 == 0:
            point_step = int(point_step)
        else:
            print("ERROR: POINT STEP IS NOT NATURAL. ROUND.")
            point_step = int(point_step)
        aux_points = patch_side-np.array(range(0,patch_side,point_step))                                    
        self.tesseras_initial_points = math_utils.cartesian_product2D(aux_points, aux_points)         # We obtain all tesseras initial points.

        print("T =" + str(self.T))
        print("ALPHA =" + str(self.ALPHA))
        print("K =" + str(self.K))
        print("C =" + str(self.C))
        print("Tesseras initial points:")
        print(self.tesseras_initial_points)       
        print("L (Encoder Output Layer) = " + str(self.narrowest_layer_size) + " " + str(type(self.narrowest_layer_size)))
        print("Patch Side size = " + str(self.patch_side) + " " + str(type(self.patch_side)))
        print("Padding value = " + str(self.padding_value) + " " + str(type(self.padding_value)))
        print("Number of Images for Training = " + str(self.number_of_images_for_training) + " " + str(type(self.number_of_images_for_training)))
        
    def initialize_data(self, image):                                                                                                                   # Method to initialize data.
        self.original_image_height = image.shape[0]                                                                                                     # We set images height.  
        self.original_image_width = image.shape[1]                                                                                                      # We set images width.
        self.extra_padding_height_to_cover_the_last_patch = ((self.original_image_height % self.patch_side)!=0)*(self.patch_side - (self.original_image_height % self.patch_side))                            # We obtain the extra padding to height that must be added to fill the last patch if needed.
        self.extra_padding_width_to_cover_the_last_patch = ((self.original_image_width % self.patch_side)!=0)*(self.patch_side - (self.original_image_width % self.patch_side))                              # We obtain the extra padding to width that must be added to fill the last patch if needed.
        self.original_image_height_with_padding = self.original_image_height + 2*self.patch_side + self.extra_padding_height_to_cover_the_last_patch    # We obtain heigth with padding.
        self.original_image_width_with_padding = self.original_image_width + 2*self.patch_side + self.extra_padding_width_to_cover_the_last_patch       # We obtain width with padding.        
        self.tessera_height = int(math.ceil(float(self.original_image_height/self.patch_side)))*self.patch_side + self.patch_side                       # We obtain each tessera height, we ensure it has the correct size.
        self.tessera_width = int(math.ceil(float(self.original_image_width/self.patch_side)))*self.patch_side + self.patch_side                         # We obtain each tessera width, we ensure it has the correct size.
        self.patches_per_height = int(self.tessera_height / self.patch_side)                                                                            # We set patches per height.
        self.patches_per_width = int(self.tessera_width / self.patch_side)                                                                              # We set patches per width.
        self.patches_per_tessera = self.patches_per_width*self.patches_per_height                                                                       # We set total number of patches per tessera.
        self.patches_per_image = self.patches_per_tessera * self.T                                                                                      # We set total number of patches per image. 
        
        print("Tessera Height = " + str(self.tessera_height) + " " + str(type(self.tessera_height)))
        print("Tessera Width = " + str(self.tessera_width) + " " + str(type(self.tessera_width)))
        print("Patches per height = " + str(self.patches_per_height) + " " + str(type(self.patches_per_height)))
        print("Patches per width = " + str(self.patches_per_width) + " " + str(type(self.patches_per_width)))
        print("Patches per tessera = " + str(self.patches_per_tessera) + " " + str(type(self.patches_per_tessera)))
        print("Patches per image = " + str(self.patches_per_image) + " " + str(type(self.patches_per_image)))
        print("Original Image Height = " + str(self.original_image_height) + " " + str(type(self.original_image_height)))
        print("Original Image Width = " + str(self.original_image_width) + " " + str(type(self.original_image_width)))
        print("Extra padding to height to fill last patch = " + str(self.extra_padding_height_to_cover_the_last_patch) + " " + str(type(self.extra_padding_height_to_cover_the_last_patch)))
        print("Extra padding to width to fill last patch = " + str(self.extra_padding_width_to_cover_the_last_patch) + " " + str(type(self.extra_padding_width_to_cover_the_last_patch)))
        print("Original Image Height with Padding = " + str(self.original_image_height_with_padding) + " " + str(type(self.original_image_height_with_padding)))
        print("Original Image Width with Padding = " + str(self.original_image_width_with_padding) + " " + str(type(self.original_image_width_with_padding)))

        #There will be a gaussian distribution for each encoded patch component for each tessera.
        self.SIGMA2 = np.zeros(
            [self.T, self.patches_per_height, self.patches_per_width, self.narrowest_layer_size])   # We generate a matrix full of zeroes to initialize sigma2.
        self.MU = np.zeros(
            [self.T, self.patches_per_height, self.patches_per_width, self.narrowest_layer_size])   # We generate a matrix full of zeroes to initialize mu.
        self.SAMPLEVARIANCE = np.zeros(
            [self.T, self.patches_per_height, self.patches_per_width, self.narrowest_layer_size])   # We generate a matrix full of zeroes to initialize sample variance.
        
        #Online mean and sigma2 matrix utilities.            
        self.M2 = np.zeros(
            [self.T, self.patches_per_height, self.patches_per_width, self.narrowest_layer_size])   # We generate a matrix full of zeroes to initialize Onlines aux value.
            
        self.data_initialized = True                                                                # Initialization done.

    def get_initial_background_model_from_training_image_set(self):                                 # Method to get the initial background model.
        print ("Obtaining initial median and standard desviation for each component.")
        print("Patches per image:")
        print(self.patches_per_image)
        for T_index in range(self.T):                                                                           # For each tessera...
            for ii in range(self.patches_per_height):                                                           # For each patch...
                for jj in range(self.patches_per_width):
                    absolute_index = T_index * self.patches_per_tessera + ii * self.patches_per_width + jj      # We get the position of patch from the tessera from the first image we are going to process.
                    for kk in range(self.narrowest_layer_size):                                                 # For each component...
                        # Average value already is in the array.  
                        self.SIGMA2[T_index, ii, jj, kk] = self.M2[T_index, ii, jj, kk]/self.processed_images                   # We calculate "std^2" value                
                        self.SAMPLEVARIANCE[T_index, ii, jj, kk] = self.M2[T_index, ii, jj, kk]/(self.processed_images - 1)    # We calculate "std^2" value 
        
        
        #print("Training done.")
        self.training_done = True                                                                               # We declare training as done.
        
        #print("Total time for training: " + str(time.time()-start_time))
        # print(np.mean(self.MU, axis = 2))
        # print(np.mean(self.MU, axis = 2).shape)
        # print(np.mean(self.SIGMA2, axis = 2))
        # print(np.mean(self.SIGMA2, axis = 2).shape)
        # quit()

    def next_image(self, image, training=False):                                                                                    # Method to deal with next image.
        # image -> the image to turn into patch and encode.
        # training -> are we on training frames?
        if training:                                                                                                                # This image will be used for training.
            if self.training_done or self.processed_images >= self.number_of_images_for_training:                                   # If training has been already done.
                print("Error: background model training already done.")                                                             # Error we do nothing.

            else:                                                                                                                   # If training has not be done yet.
                if not self.data_initialized:                                                                                       # If model data has not been initialized
                    self.initialize_data(image)                                                                                     # We initialize them.
                #We normalize the image and put it into matrix to be used to initialize the background model.
                #print(self.normalized_training_dataset.shape)
                #print(self.original_image_height_with_padding)
                
                start_time=time.time()
                
                image= image.astype(float) / 255.                                                                                   #We normalize the image.
                
                normalized_image = np.random.uniform(0, 1, 
                [self.original_image_height_with_padding, self.original_image_width_with_padding, 3])                                                       # We generate uniform noise image.
                normalized_image[self.patch_side:self.original_image_height_with_padding-(self.patch_side + self.extra_padding_height_to_cover_the_last_patch), 
                self.patch_side:self.original_image_width_with_padding-(self.patch_side + self.extra_padding_width_to_cover_the_last_patch)] = image        # We set normalized image.
                
                for T_index in range(self.T):                                                                                                                   # For each tessera...
                    tessera_initial_point = self.tesseras_initial_points[T_index]
                    patched_tessera = image_utils.split_in_patches_one_image_with_tessera(
                        normalized_image, self.patch_side, 
                        tessera_initial_point, self.patches_per_height, self.patches_per_width)     
                        
                    predicted_patches = None
                    while predicted_patches is None:
                        predicted_patches=self.encoder.predict(patched_tessera)                                                                                 # We encode the patches.
                        
                    for ii in range(self.patches_per_height):                                                                                                   # For each patch...
                        for jj in range(self.patches_per_width):
                            absolute_index = ii * self.patches_per_width + jj                                                                                   # We get the absolute index for the patch in the tessera we are fetching.

                            encoded_patch_value = predicted_patches[absolute_index]                                                                             # We get the encoded patch.
                            
                            if self.processed_images == 0:
                                self.MU[T_index,ii,jj] = encoded_patch_value
                                
                            delta = encoded_patch_value - self.MU[T_index,ii,jj]
                            self.MU[T_index,ii,jj] += delta/(self.processed_images + 1)
                            delta2 = encoded_patch_value - self.MU[T_index,ii,jj]
                            self.M2[T_index,ii,jj] += np.multiply(delta, delta2)
                           
            
            self.processed_images += 1
            return None

        else:                                                                                                                # This image will be used as 
            if not self.training_done:                                                                                       # If training process has not been finalized.
                self.get_initial_background_model_from_training_image_set()                                                  # We apply final calculus over training data and finalize it.
            
            start_time = time.time()
            #We add padding and normalize the image.
            image= image.astype(float) / 255.                                                                                                               # We normalize the image.
                
            normalized_image = np.random.uniform(0, 1, 
                [self.original_image_height_with_padding, self.original_image_width_with_padding, 3])                                                       # We generate uniform noise image.
            normalized_image[self.patch_side:self.original_image_height_with_padding-(self.patch_side + self.extra_padding_height_to_cover_the_last_patch), 
                self.patch_side:self.original_image_width_with_padding-(self.patch_side + self.extra_padding_width_to_cover_the_last_patch)] = image        # We set normalized image.
            
            foreground_probability_matrix = np.zeros((self.T, self.patches_per_height, self.patches_per_width))
            foreground_grayscale_image = np.zeros((self.T, self.original_image_height_with_padding, self.original_image_width_with_padding))
            # print(np.mean(self.MU, axis = 2))
            # print(np.mean(self.SIGMA2, axis = 2))
            
            #print("Time to initialize structures: " + str(time.time() - aux_time))
            aux=time.time()
            
            for T_index in range(self.T):                                                                                                                   # For each tessera...
                
                tessera_initial_time = time.time()
                
                tessera_initial_point = self.tesseras_initial_points[T_index]
                patched_tessera = image_utils.split_in_patches_one_image_with_tessera(
                    normalized_image, self.patch_side, 
                    tessera_initial_point, self.patches_per_height, self.patches_per_width)                                                                 # We split the tessera into patches
                    
                #print("Time to split tessera into patches: " + str(time.time()-tessera_initial_time))
                aux_time=time.time()

                predicted_patches = None
                while predicted_patches is None:
                    predicted_patches=self.encoder.predict(patched_tessera)                                                                                 # We encode the patches.
                
                #print("Time to predict patches: " + str(time.time()-aux_time))
                aux_time=time.time()
                
                for ii in range(self.patches_per_height):                                            # For each patch...
                    for jj in range(self.patches_per_width):
                        absolute_index = ii * self.patches_per_width + jj                            # We get the absolute index for the patch in the tessera we are fetching.

                        encoded_patch_value = predicted_patches[absolute_index]                      # We get the encoded patch.

                        prob_Fore_cond_v = np.sum(np.abs(encoded_patch_value-self.MU[T_index,ii,jj]) > self.K * 
                            np.sqrt(self.SIGMA2[T_index,ii,jj])) > self.C                            # We compare with its aassociated MU and SIGMA2.
                        # print(prob_Fore_cond_v)
                        prob_Back_cond_v = 1 - prob_Fore_cond_v                                      # We get the background probability as 1 minus the foreground probability. 
                        foreground_probability_matrix[T_index,ii,jj] = prob_Fore_cond_v

                        self.SIGMA2[T_index,ii,jj] = (1 - self.ALPHA * prob_Back_cond_v) * self.SIGMA2[T_index,ii,jj] + self.ALPHA * prob_Back_cond_v * np.power(encoded_patch_value - self.MU[T_index,ii,jj], 2)
                        self.MU[T_index,ii,jj] = (1 - self.ALPHA * prob_Back_cond_v) * self.MU[T_index,ii,jj] + self.ALPHA * prob_Back_cond_v * encoded_patch_value

                #print("Time to get prob_fore and update model : "+ str(time.time()-aux_time))
                aux_time=time.time()

                foreground_grayscale_image[T_index, tessera_initial_point[0]:foreground_grayscale_image.shape[1]-1*(self.patch_side-tessera_initial_point[0]),
                    tessera_initial_point[1]:foreground_grayscale_image.shape[2]-1*(self.patch_side-tessera_initial_point[1])] = image_utils.create_segmented_grayscale_image_from_foreground_matrix(
                    foreground_probability_matrix[T_index, :, :], 
                    self.patch_side, self.patch_side, 
                    original_image_height = self.tessera_height, original_image_width = self.tessera_width)  # We createa grayscale to represent the tessera result and insert it into the tessera position.

            self.tesseras_images = foreground_grayscale_image
            final_foreground_grayscale_image = np.mean(foreground_grayscale_image, axis=0)/255. >= 0.5  #We get final foreground grayscale image by  voting for each pixel trhough the results in each tessera. This result should be cut to delete padding.
            #print(final_foreground_grayscale_image.shape)
            final_foreground_grayscale_image = np.reshape(final_foreground_grayscale_image, [final_foreground_grayscale_image.shape[0], final_foreground_grayscale_image.shape[1],1])
            self.processed_images += 1

            self.last_segmented_image = final_foreground_grayscale_image*255.
            #print(final_foreground_grayscale_image.shape)
            
            #print("Total time to process a frame: " + str(time.time()-start_time))
            return final_foreground_grayscale_image                                                                                           # We return the data to train...

    def get_last_segmented_image(self):
        #We need to cut the image to have the right size.
        return self.last_segmented_image[self.padding_value:-1*(self.extra_padding_height_to_cover_the_last_patch + self.padding_value),
            self.padding_value:-1*(self.extra_padding_width_to_cover_the_last_patch + self.padding_value)] 
        
    def get_last_segmented_tesseras_adjusted_to_image_shape(self):
        #We need to cut the tesseras.
        tesseras = np.zeros([self.T, self.original_image_height,
        self.original_image_width])
        for tes in range(self.T):
            tesseras[tes,:,:] = self.tesseras_images[tes, 
                (self.padding_value):-1*(self.extra_padding_height_to_cover_the_last_patch + self.padding_value),
                (self.padding_value):-1*(self.extra_padding_width_to_cover_the_last_patch + self.padding_value)]        
        return tesseras
        
    def get_last_segmented_tesseras(self):
        return self.tesseras_images
