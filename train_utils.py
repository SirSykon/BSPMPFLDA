import matplotlib
import os
import cv2
import numpy as np
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from keras.utils import plot_model
from glob import glob
from random import shuffle

def save_training_and_validation_loss_from_history(history, output_path):
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(output_path, "train_loss_figure.png"))

def save_training_and_validation_accuracy_from_history(history, output_path):
    plt.clf()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(output_path, "train_accuracy_figure.png"))

def save_model(model, output_path, model_name, plot_graph_model=True):
    h5py_model_file = os.path.join(output_path, model_name+".h5py")
    h5py_model_file_weights = os.path.join(output_path, model_name+"_weights.h5")
    model.save(h5py_model_file)
    json_model_file = os.path.join(output_path, model_name+".json")
    model_json = model.to_json()
    with open(json_model_file, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(h5py_model_file_weights)
    if plot_graph_model:
        plot_model(autoencoder, to_file = os.path.join(output_path, "graph_"+model_name+".png"), show_shapes = True)

def get_simple_batches(data, batch_size, data_loaded_in_RAM=True):                                    # Generator to create batchs from images in RAM.
    shuffle(data)
    for i in range(int(len(data)/batch_size)):
        new_batch = []
        for j in range(batch_size):
            if data_loaded_in_RAM:
                new_batch.append(data[i*batch_size + j])
            else:
                img = cv2.imread(data[i*batch_size + j])
                new_batch.append(np.copy(img))
        yield new_batch

def simple_generator(images_in_RAM, data_transform_function, batch_size, data_loaded_in_RAM = True, average_filter_size = None, median_filter_size = None):                           # Class to generate altered images on the fly batch by batch.
    while True:
        for y_batch in get_simple_batches(images_in_RAM, batch_size, data_loaded_in_RAM=data_loaded_in_RAM):
            x_batch = [data_transform_function(y) for y in y_batch]
            if not (average_filter_size is None):
                y_batch = [cv2.blur(np.float32(y),(average_filter_size,average_filter_size)) for y in y_batch]
            if not (median_filter_size is None):
                y_batch = [cv2.medianBlur(np.float32(y),median_filter_size) for y in y_batch]
            yield np.array(x_batch), np.array(y_batch)
    
def residual_generator(images_in_RAM, data_transform_function, batch_size, data_loaded_in_RAM = True, average_filter_size = None, median_filter_size = None):                           # Class to generate altered images on the fly batch by batch.
    while True:
        for img_batch in get_simple_batches(images_in_RAM, batch_size, data_loaded_in_RAM=data_loaded_in_RAM):
            x_batch = [data_transform_function(x) for x in img_batch]
            y_batch = []
            for img, x in zip(img_batch, x_batch):
                y = x.astype('float32') - img + 1
                y_batch.append(y)
            yield np.array(x_batch), np.array(y_batch)

def residual_generator2(images_in_RAM, data_transform_function, batch_size, data_loaded_in_RAM = True, average_filter_size = None, median_filter_size = None):                           # Class to generate altered images on the fly batch by batch.
    while True:
        for img_batch in get_simple_batches(images_in_RAM, batch_size, data_loaded_in_RAM=data_loaded_in_RAM):
            x_batch = img_batch
            y_batch = []
            for x in x_batch:
                if not (average_filter_size is None):
                    y = x - cv2.blur(np.float32(x),(average_filter_size,average_filter_size)) + 1
                if not (median_filter_size is None):
                    y = x - cv2.medianBlur(np.float32(x),median_filter_size) + 1
                y_batch.append(y)                
            yield np.array(x_batch), np.array(y_batch)

def multiple_filters_generator(images_in_RAM, data_transform_function, batch_size, data_loaded_in_RAM = True, average_filter_size = None, median_filter_size = None):                           # Class to generate altered images on the fly batch by batch.
    while True:
        for img_batch in get_simple_batches(images_in_RAM, batch_size, data_loaded_in_RAM=data_loaded_in_RAM):
            x_batch = []
            for img in img_batch:
                noised_img = data_transform_function(img.copy())
                input_data = np.zeros(shape=[noised_img.shape[0], noised_img.shape[1], 3*noised_img.shape[2]])
                input_data[:,:,0:3] = np.copy(noised_img)
                input_data[:,:,3:6] = cv2.medianBlur(np.float32(noised_img),median_filter_size)
                input_data[:,:,6:] = cv2.blur(np.float32(noised_img),(average_filter_size,average_filter_size))
                x_batch.append(input_data)
            y_batch = img_batch
            yield np.array(x_batch), np.array(y_batch)

def train_model_with_generator(model, train_data, validation_data, training_epochs, batch_size, data_transform_function, preload_data_in_RAM, average_filter_size = None, median_filter_size= None):

    history = model.fit_generator(simple_generator(train_data, data_transform_function, batch_size, data_loaded_in_RAM = preload_data_in_RAM, average_filter_size = average_filter_size, median_filter_size = median_filter_size),
                    epochs=training_epochs,
                    steps_per_epoch=int(len(train_data)/batch_size),
                    shuffle=True,
                    validation_data = simple_generator(validation_data, data_transform_function, batch_size, data_loaded_in_RAM = preload_data_in_RAM, average_filter_size = average_filter_size, median_filter_size = median_filter_size),
                    validation_steps = int(len(validation_data)/batch_size))
    return history

def train_model_with_residual_generator(model, train_data, validation_data, training_epochs, batch_size, data_transform_function, preload_data_in_RAM, average_filter_size = None, median_filter_size= None):

    history = model.fit_generator(residual_generator2(train_data, data_transform_function, batch_size, data_loaded_in_RAM = preload_data_in_RAM, average_filter_size = average_filter_size, median_filter_size = median_filter_size),
                    epochs=training_epochs,
                    steps_per_epoch=int(len(train_data)/batch_size),
                    shuffle=True,
                    validation_data = residual_generator2(validation_data, data_transform_function, batch_size, data_loaded_in_RAM = preload_data_in_RAM, average_filter_size = average_filter_size, median_filter_size = median_filter_size),
                    validation_steps = int(len(validation_data)/batch_size))
    return history

def train_model_with_multiple_filters_generator(model, train_data, validation_data, training_epochs, batch_size, data_transform_function, preload_data_in_RAM, average_filter_size = None, median_filter_size= None):

    history = model.fit_generator(multiple_filters_generator(train_data, data_transform_function, batch_size, data_loaded_in_RAM = preload_data_in_RAM, average_filter_size = average_filter_size, median_filter_size = median_filter_size),
                    epochs=training_epochs,
                    steps_per_epoch=int(len(train_data)/batch_size),
                    shuffle=True,
                    validation_data = multiple_filters_generator(validation_data, data_transform_function, batch_size, data_loaded_in_RAM = preload_data_in_RAM, average_filter_size = average_filter_size, median_filter_size = median_filter_size),
                    validation_steps = int(len(validation_data)/batch_size))
    return history
