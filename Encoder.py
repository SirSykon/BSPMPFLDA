import keras
from keras.models import model_from_json
import os
from keras import backend as K
import tensorflow as tf

def create_relu_advanced(max_value=1.):
    def relu_advanced(x):
        return K.relu(x, max_value=K.cast_to_floatx(max_value))
    return relu_advanced
    
def relu_advanced(x):
    return K.relu(x, max_value=K.cast_to_floatx(1))

class Encoder:                  # Class to represent the encoder model.
    encoder_model = None
    narrowest_layer_size = None
    
    custom_objects={'relu_advanced': create_relu_advanced()}
    print(custom_objects)
    
    busy = None                 #Variable to now if this enocer is busy predictiong another data.

    def __init__(self, encoder_model_path=None, models_path = None, narrowest_layer_size = -1, load_from_json = False):
        self.busy = False
        if encoder_model_path == None:
            if models_path != None and narrowest_layer_size != -1:
                models_path = os.path.join(models_path, "models_"+str(narrowest_layer_size), "TinyImage_encoder_model_RGB.h5py")
                print("Loading encoder from " + encoder_model_path)
                self.encoder_model = keras.models.load_model(encoder_model_path, custom_objects=self.custom_objects)
                self.narrowest_layer_size = narrowest_layer_size

            else:
                print("Error. No enough information to load the .h5py file.")
        else:
            
            if load_from_json:
                json_file_path = encoder_model_path+".json"
                weights_file = encoder_model_path+"_weights.h5"
                print("Loading encoder from " + json_file_path + " and " + weights_file)
                json_file = open(json_file_path, "r")
                loaded_model_json = json_file.read()
                json_file.close()
                self.encoder_model = model_from_json(loaded_model_json, custom_objects=self.custom_objects)
                self.encoder_model.load_weights(weights_file)
            else:
                print("Loading encoder from " + encoder_model_path)
                # Code to avoid FailedPreconditionError: lack of intialization
                self.encoder_model = keras.models.load_model(encoder_model_path)
            self.narrowest_layer_size = int(self.encoder_model.output.shape[1]) 
            print(self.narrowest_layer_size)
          #  K.set_session(tf.Session(graph=self.encoder_model.output.graph)) 
         #   init = K.tf.global_variables_initializer() 
         #   K.get_session().run(init)

        print(self.encoder_model)
        print(self.encoder_model.layers)
        print(self.encoder_model.inputs)
        print(self.encoder_model.outputs)
        print(self.encoder_model.summary())
        print(self.encoder_model.get_config())
        
    def get_narroest_layer_size(self):
        return self.narrowest_layer_size

    def predict(self,input_data):
        if not self.busy:
            self.busy = True
            prediction = self.encoder_model.predict(input_data)
            self.busy = False
            return prediction
        else:
            return None
