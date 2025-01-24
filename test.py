import tensorflow as tf
from keras import backend as KB
import os
import numpy as np
import time

###################################
os.environ["CUDA_VISIBLE_DEVICES"]="0"                      # To force tensorflow to only see one GPU.
# TensorFlow wizardry
config = tf.ConfigProto()
     
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True                      # pylint: disable=import-error
     
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.1    # pylint: disable=import-serror
     
# Create a session with the above options specified.
KB.tensorflow_backend.set_session(tf.Session(config=config))
###################################

np_matrix_a = np.arange(1000000).reshape((1000,1000)).astype('float64')
np_matrix_b = np.arange(1000000,2000000).reshape((1000,1000)).astype('float64')

test_time = time.time()

a = np.add(np_matrix_a, np_matrix_b)
a = np.add(a, np_matrix_b)
a = np.add(a, np_matrix_b)
a = np.add(a, np_matrix_b)
a = np.add(a, np_matrix_b)
a = np.add(a, np_matrix_b)
a = np.add(a, np_matrix_b)
a = np.add(a, np_matrix_b)
a = np.add(a, np_matrix_b)
a = np.add(a, np_matrix_b)
a = np.add(a, np_matrix_b)
a = np.add(a, np_matrix_b)
a = np.subtract(a, np_matrix_b)
print(np.matmul(a, np_matrix_b))

print(str(time.time()-test_time))

test_time = time.time()

a = tf.Variable(np_matrix_a, name="matrix_a")
b = tf.Variable(np_matrix_b, name="matrix_b")
init = tf.variables_initializer([a, b], name="init")
x = tf.add(a,b)
x = tf.add(x,b)
x = tf.add(x,b)
x = tf.add(x,b)
x = tf.add(x,b)
x = tf.add(x,b)
x = tf.add(x,b)
x = tf.add(x,b)
x = tf.add(x,b)
x = tf.add(x,b)
x = tf.add(x,b)
x = tf.add(x,b)
x = tf.subtract(x,b)
x = tf.matmul(x, b)

with tf.Session() as s:
    writer = tf.summary.FileWriter('graphs', s.graph)
    s.run(init)
    print(s.run(x))
    
print(str(time.time()-test_time))
