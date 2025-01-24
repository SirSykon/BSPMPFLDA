import numpy as np

gaussian_noise_mean = 0
gaussian_noise_low_standard_desviation = 0.1
gaussian_noise_medium_low_standard_desviation = 0.2
gaussian_noise_medium_high_standard_desviation = 0.3
gaussian_noise_high_standard_desviation = 0.4
salt_and_pepper_ratio = 0.2
mask_noise_ratio = 0.2

def add_gaussian_noise(image, gaussian_noise_mean=0, gaussian_noise_standard_desviation=0.1):
    noise_mask = np.random.normal(loc = gaussian_noise_mean, scale = gaussian_noise_standard_desviation, size = image.shape)
    noise_image = np.add(image, noise_mask)
    noise_image = np.clip(noise_image, a_min = 0, a_max = 1, out = noise_image)
    return noise_image

def add_uniform_noise(image, low=-0.5, high=0.5):
    noise_mask = np.random.uniform(low=low, high=high, size = image.shape)
    noise_image = np.add(image, noise_mask)
    noise_image = np.clip(noise_image, a_min = 0, a_max = 1, out = noise_image)
    return noise_image

def add_salt_and_pepper_noise(image, min=0., max=1., ratio=salt_and_pepper_ratio):
    noise_mask_c1 = np.random.uniform(low=min, high=max, size = image.shape[:-1])
    noise_mask = np.zeros(image.shape)
    noise_mask[:,:,0] = noise_mask_c1
    noise_mask[:,:,1] = noise_mask_c1
    noise_mask[:,:,2] = noise_mask_c1
    black_noise_image = image + (noise_mask > (max - ratio/2.))*max
    noise_image = black_noise_image - (noise_mask < (ratio/2.))*max
    noise_image = np.clip(noise_image, a_min = 0, a_max = 1, out = noise_image)
    return noise_image

def add_mask_noise(image, mask_shape=(2,2), ratio=mask_noise_ratio):
    noise_mask_c = np.random.uniform(low=0, high=1, size = image.shape[:-1])
    noise_mask_c1 = (noise_mask_c >= (1-(ratio/(mask_shape[0]*mask_shape[1]))))*1
    noise_mask_c3 = np.zeros(image.shape)
    noise_mask_c3[:,:,0] = noise_mask_c1
    noise_mask_c3[:,:,1] = noise_mask_c1
    noise_mask_c3[:,:,2] = noise_mask_c1
    noise_mask_c3 = 1-noise_mask_c3
    noise_mask = np.zeros(image.shape) + 1
    
    for ii in range(mask_shape[0]):
        for jj in range(mask_shape[1]):
            noise_mask[0:image.shape[0]-ii, 0:image.shape[1]-jj, :] = noise_mask_c3[ii:, jj:, :]

    noise_image = np.multiply(image, noise_mask)
    
    noise_image = np.clip(noise_image, a_min = 0, a_max = 1, out = noise_image)
    
    return noise_image
            
def add_small_mask_noises(image):
    return add_mask_noise(image, mask_shape=(1,1))
    
def add_medium_mask_noises(image):
    return add_mask_noise(image, mask_shape=(2,2))
    
def add_big_mask_noises(image):
    return add_mask_noise(image, mask_shape=(3,3))

def add_low_gaussian_noise(image):
    return add_gaussian_noise(image, gaussian_noise_mean = gaussian_noise_mean, gaussian_noise_standard_desviation = gaussian_noise_low_standard_desviation)

def add_medium_low_gaussian_noise(image):
    return add_gaussian_noise(image, gaussian_noise_mean = gaussian_noise_mean, gaussian_noise_standard_desviation = gaussian_noise_medium_low_standard_desviation)
  
def add_medium_high_gaussian_noise(image):
    return add_gaussian_noise(image, gaussian_noise_mean = gaussian_noise_mean, gaussian_noise_standard_desviation = gaussian_noise_medium_high_standard_desviation)
  
def add_high_gaussian_noise(image):
    return add_gaussian_noise(image, gaussian_noise_mean = gaussian_noise_mean, gaussian_noise_standard_desviation = gaussian_noise_high_standard_desviation)

def add_no_noise(image):
    return image
