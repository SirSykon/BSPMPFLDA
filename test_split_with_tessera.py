import numpy as np
import image_utils


matrix = np.zeros([16,16,3])
aran = np.arange(0,16*16)
matrix[:,:,0] =  np.reshape(aran, [16,16])
matrix[:,:,1] =  np.reshape(aran, [16,16])
matrix[:,:,2] =  np.reshape(aran, [16,16])

print(matrix)

out = image_utils.split_in_patches_one_image_with_tessera(matrix,4,[2,2],3,3)

print(out)