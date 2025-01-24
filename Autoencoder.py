import keras
import os
from keras import backend as K
import tensorflow as tf

class Autoencoder:                  # Class to represent the encoder model.
    autoencoder_model = None

    def __init__(self, autoencoder_model_path=None, models_path = None, narrowest_layer_size = -1):
        if autoencoder_model_path == None:
            if models_path != None and narrowest_layer_size != -1:
                models_path = os.path.join(models_path, "models_"+str(narrowest_layer_size), "TinyImage_autoencoder_model_RGB.h5py")
                print("Loading encoder from " + autoencoder_model_path)
                self.autoencoder_model = keras.models.load_model(autoencoder_model_path)
                self.narrowest_layer_size = narrowest_layer_size

            else:
                print("Error. No enough information to load the .h5py file.")
        else:
            print("Loading encoder from " + autoencoder_model_path)
            # Code to avoid FailedPreconditionError: lack of intialization
            self.autoencoder_model = keras.models.load_model(autoencoder_model_path)
            self.narrowest_layer_size = self.autoencoder_model.output.shape[1] 
            print(self.narrowest_layer_size)
          #  K.set_session(tf.Session(graph=self.encoder_model.output.graph)) 
         #   init = K.tf.global_variables_initializer() 
         #   K.get_session().run(init)

        print(self.autoencoder_model)
        print(self.autoencoder_model.layers)
        print(self.autoencoder_model.inputs)
        print(self.autoencoder_model.outputs)
        print(self.autoencoder_model.summary())
        print(self.autoencoder_model.get_config())
        
    def get_narroest_layer_size(self):
        return self.narrowest_layer_size

    def predict(self,input_data):
        return self.autoencoder_model.predict(input_data)
