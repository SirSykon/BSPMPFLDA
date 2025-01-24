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
    T = -1                                                  # Number of tesseras. This number should be 2^n form.
    ALPHA = -1                                              # Background model update step size.
    
    encoder = None                                          # encoder object.
    narrowest_layer_size = None                             # narrowest layer size. It must bee qual to encoder output layer.
    min_encode_value = None                                 # Min value for each output component in narrowest layer size
    max_encode_value = None                                 # Max value for each output component in narrowest layer size
    padding_value = None                                    # Value to serve as paddding to increase image size in order to have margin to get patches (example: we have a 16x16 patch but we segmented with 8x8 resolution, it is needed 8 as padding).
    pi_Fore = None                                          # A priori Foreground probability.
    pi_Back = None                                          # A priori Background probability.
    SIGMA2 = None                                           # variance values for each component for each patch.
    MU = None                                               # average values for each component for each patch.
    SAMPLEVARIANCE = None                                   # Sample variance for each component for each patch.
    M2 = None                                               # Auxiliar value for online calculation of variance and average.
    RO = None                                               # Probability treshold. If None, a patch will be declared foreground iff R_Fore > R_Back, else R_Fore > RO.
    const_k_t_cond_MU_COVARIANCE = None                     # Constant used in model calculations.
    
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
    
    MIN_VARIANCE_VALUE = 0.001
    

    def __init__(self, encoder=None, patch_side = -1, number_of_images_for_training = -1, ALPHA = -1, T = -1, RO = None, min_encode_value = None, max_encode_value = None, initial_pi_Back = 0.5, initial_pi_Fore = 0.5, update_pi = False):            # Class to represent the background model base class for Foreground detection
        self.encoder = encoder
        self.narrowest_layer_size = self.encoder.get_narroest_layer_size()
        self.patch_side = patch_side
        self.padding_value = self.patch_side
        self.training_done = False
        self.data_initialized = False
        self.number_of_images_for_training = number_of_images_for_training
        self.processed_images = 0
        self.ALPHA = ALPHA
        self.T=T
        self.RO = RO
        self.min_encode_value = min_encode_value
        self.max_encode_value = max_encode_value
        self.pi_Back = initial_pi_Back
        self.pi_Fore = initial_pi_Fore
        self.update_pi = update_pi
        
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
        print("RO =" + str(self.RO))
        print("Tesseras initial points:")
        print(self.tesseras_initial_points)       
        print("L (Encoder Output Layer) = " + str(self.narrowest_layer_size) + " " + str(type(self.narrowest_layer_size)))
        self.const_k_t_cond_MU_COVARIANCE = math.pow(2*math.pi, -1*self.narrowest_layer_size/2)
        print("const_k_t_cond_MU_COVARIANCE = " + str(self.const_k_t_cond_MU_COVARIANCE))
        print("Patch Side size = " + str(self.patch_side) + " " + str(type(self.patch_side)))
        print("Padding value = " + str(self.padding_value) + " " + str(type(self.padding_value)))
        print("Number of Images for Training = " + str(self.number_of_images_for_training) + " " + str(type(self.number_of_images_for_training)))
        print("Initial a priori foreground probability: " + str(self.pi_Fore))
        print("Initial a priori background probability: " + str(self.pi_Back))
        print("Update pi: " + str(self.update_pi))
        
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
                    
                    if np.prod(self.SIGMA2[T_index, ii, jj]) == 0:
                        print("Error: Initial Variances in patch ("+str(ii)+","+str(jj)+") in T = "+str(T_index)+":")
                        self.SIGMA2[T_index, ii, jj] = self.SIGMA2[T_index, ii, jj] + ((self.SIGMA2[T_index, ii, jj] < self.MIN_VARIANCE_VALUE) * self.MIN_VARIANCE_VALUE)
                        print(self.SIGMA2[T_index, ii, jj])
                        
                    if np.prod(self.SAMPLEVARIANCE[T_index, ii, jj]) == 0:
                        print("Error: Initial Sample Variances in patch ("+str(ii)+","+str(jj)+") in T = "+str(T_index)+":")
                        self.SAMPLEVARIANCE[T_index, ii, jj] = self.SAMPLEVARIANCE[T_index, ii, jj] + ((self.SAMPLEVARIANCE[T_index, ii, jj] < self.MIN_VARIANCE_VALUE) * self.MIN_VARIANCE_VALUE)
                        print(self.SAMPLEVARIANCE[T_index, ii, jj])
                            
                        #self.SIGMA2[:, :, :, :] = 0.2
                        
        self.training_done = True                                                                               # We declare training as done.

    def update_background_model(self, MU, VARIANCE, SAMPLEVARIANCE, prob_Back_cond_v, v):
        new_VARIANCE = (1 - self.ALPHA * prob_Back_cond_v) * VARIANCE + self.ALPHA * prob_Back_cond_v * np.power(v - MU, 2)
        new_SAMPLEVARIANCE = (1 - self.ALPHA * prob_Back_cond_v) * SAMPLEVARIANCE + self.ALPHA * prob_Back_cond_v * np.power(v - MU, 2)
        new_MU = (1 - self.ALPHA * prob_Back_cond_v) * MU + self.ALPHA * prob_Back_cond_v * v
        
        new_VARIANCE = new_VARIANCE + ((new_VARIANCE<self.MIN_VARIANCE_VALUE) * self.MIN_VARIANCE_VALUE)
        
        if self.update_pi:
            new_pi_Back = (1-self.ALPHA)*self.pi_Back + self.ALPHA*prob_Back_cond_v
            new_pi_Fore = (1-self.ALPHA)*self.pi_Fore + self.ALPHA*(1-prob_Back_cond_v)
        else:
            new_pi_Back = self.pi_Back
            new_pi_Fore = self.pi_Fore
        
        return new_MU, new_VARIANCE, new_SAMPLEVARIANCE, new_pi_Back, new_pi_Fore
        
    def get_encoded_patches_from_image(self, image):
        #We add padding and normalize the image.
        image = image.astype(float) / 255.                                                                                                               # We normalize the image.
                
        normalized_image = np.random.uniform(0, 1, 
            [self.original_image_height_with_padding, self.original_image_width_with_padding, 3])                                                       # We generate uniform noise image.
        normalized_image[self.patch_side:self.original_image_height_with_padding-(self.patch_side + self.extra_padding_height_to_cover_the_last_patch), 
            self.patch_side:self.original_image_width_with_padding-(self.patch_side + self.extra_padding_width_to_cover_the_last_patch)] = image        # We set normalized image.

        all_patched_tesseras = image_utils.split_in_patches_one_image_with_tesseras(
            normalized_image, self.patch_side, 
            self.tesseras_initial_points, self.patches_per_height, self.patches_per_width)  # We split the image into patches

        all_predicted_patches = None
        while all_predicted_patches is None:
            all_predicted_patches=self.encoder.predict(all_patched_tesseras)                # We encode all patches.

        return all_predicted_patches
        
    def iwinac_get_foreground_probability(self, MU, VARIANCE, v):
        return np.sum(np.abs((v-MU) > (self.K * VARIANCE))) > self.C   # We compare with its aassociated MU and SIGMA2.
        
    def get_foreground_probability(self, MU, VARIANCE, t, pi_Fore, pi_Back, printbool):
        #print("----------------------------------------")
        if printbool:
            print(t)
        if printbool:
            print(MU)
        if printbool:
            print(VARIANCE)
        aux = np.subtract(t, MU)
        if printbool:
            print(aux)
        aux = np.multiply(aux,aux)
        if printbool:    
            print(aux)
        aux = np.divide(aux, VARIANCE)
        if printbool:
            print(aux)
        exponent = (-1/2)*np.sum(aux)
        if printbool:
            print(exponent)
        det_covariance_matrix = np.prod(VARIANCE)
        if printbool:
            print(det_covariance_matrix)
        K_t_cond_MU_COVARIANCE = self.const_k_t_cond_MU_COVARIANCE * math.pow(det_covariance_matrix,-1/2) * math.exp(exponent)
        if printbool:
            print(K_t_cond_MU_COVARIANCE)
        p_t_cond_Back = K_t_cond_MU_COVARIANCE
        
        if self.max_encode_value-self.min_encode_value == 1:
            vol_H = 1
        else:
            vol_H = np.power(self.max_encode_value-self.min_encode_value, self.narrowest_layer_size)
        U_t = 1/vol_H
        p_t_cond_Fore = U_t

        denominator = pi_Back*p_t_cond_Back + pi_Fore*p_t_cond_Fore
        P_Fore_cond_t = pi_Fore*p_t_cond_Fore/denominator
        R_Fore = P_Fore_cond_t
        
        if self.RO is None:
            P_Back_cond_t = pi_Back*p_t_cond_Back/denominator
            R_Back = P_Back_cond_t
            if printbool:   
                print("R_Back = " + str(R_Back) + " R_Fore = " + str(R_Fore))
            return R_Fore > R_Back
        else:
            return R_Fore > self.RO

    def next_image(self, image, training=False):                                                                                    # Method to deal with next image.
        # image -> the image to turn into patch and encode.
        # training -> are we on training frames?
        #print(self.pi_Fore)
        #print(self.pi_Back)
        if training:                                                                                                                # This image will be used for training.
            if self.training_done or self.processed_images >= self.number_of_images_for_training:                                   # If training has been already done.
                print("Error: background model training already done.")                                                             # Error we do nothing.
                quit()
            else:                                                                                                                   # If training has not be done yet.
                if not self.data_initialized:                                                                                       # If model data has not been initialized
                    self.initialize_data(image)                                                                                     # We initialize them.
                all_predicted_patches = self.get_encoded_patches_from_image(image)

                for T_index in range(self.T):        # For each tessera...                

                    predicted_patched_tessera = all_predicted_patches[T_index*self.patches_per_tessera:(T_index+1)*self.patches_per_tessera]   # We get the encoded patches from the tessera.
                    
                    for ii in range(self.patches_per_height):                                                                                                   # For each patch...
                        for jj in range(self.patches_per_width):
                            absolute_index = ii * self.patches_per_width + jj                                                                                   # We get the absolute index for the patch in the tessera we are fetching.

                            encoded_patch_value = predicted_patched_tessera[absolute_index]                                                                     # We get the encoded patch.
                            
                            if self.processed_images == 0:
                                self.MU[T_index,ii,jj] = encoded_patch_value
                                
                            delta = encoded_patch_value - self.MU[T_index,ii,jj]
                            self.MU[T_index,ii,jj] += delta/(self.processed_images + 1)
                            delta2 = encoded_patch_value - self.MU[T_index,ii,jj]
                            self.M2[T_index,ii,jj] += np.multiply(delta, delta2)   
                            
                            #if ii==5 and jj==17 and T_index ==0:
                            #    print(self.MU[T_index,ii,jj])            
            
            self.processed_images += 1
            return None

        else:                                                                                                                # This image will not be used as training.
            if not self.training_done:                                                                                       # If training process has not been finalized.
                self.get_initial_background_model_from_training_image_set()                                                  # We apply final calculus over training data and finalize it.
            
            foreground_probability_matrix = np.zeros((self.T, self.patches_per_height, self.patches_per_width))
            foreground_grayscale_image = np.zeros((self.T, self.original_image_height_with_padding, self.original_image_width_with_padding))
            
            all_predicted_patches = self.get_encoded_patches_from_image(image)
                    
            for T_index in range(self.T):        # For each tessera...                
                
                tessera_initial_point = self.tesseras_initial_points[T_index]
                
                predicted_patched_tessera = all_predicted_patches[T_index*self.patches_per_tessera:(T_index+1)*self.patches_per_tessera]   # We get the encoded patches from the tessera.
                
                for ii in range(self.patches_per_height):                                            # For each patch in the tessera...
                    for jj in range(self.patches_per_width):
                        absolute_index = ii * self.patches_per_width + jj                            # We get the absolute index for the patch in the tessera we are fetching.

                        encoded_patch_value = predicted_patched_tessera[absolute_index]              # We get the encoded patch.
                        
                        printbool = ii == 10 and jj==5
                        prob_Fore_cond_v = self.get_foreground_probability(self.MU[T_index,ii,jj], self.SIGMA2[T_index,ii,jj], encoded_patch_value, self.pi_Fore, self.pi_Back, False)
                        
                        prob_Back_cond_v = 1 - prob_Fore_cond_v                                      # We get the background probability as 1 minus the foreground probability. 
                        foreground_probability_matrix[T_index,ii,jj] = prob_Fore_cond_v
                        
                        self.MU[T_index,ii,jj], self.SIGMA2[T_index,ii,jj], self.SAMPLEVARIANCE[T_index,ii,jj], self.pi_Back, self.pi_Fore = self.update_background_model(self.MU[T_index,ii,jj], self.SIGMA2[T_index,ii,jj], self.SAMPLEVARIANCE[T_index,ii,jj], prob_Back_cond_v, encoded_patch_value)

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
