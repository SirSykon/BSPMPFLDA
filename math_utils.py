import numpy as np

def cartesian_product2D(a,b):
    return np.array(np.meshgrid(a,b)).T.reshape(-1,2)