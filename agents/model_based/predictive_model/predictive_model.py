from keras.layers import Input, Concatenate, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape, multiply, ZeroPadding2D
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
import numpy as np
import sys,os

sys.path.insert(1,os.path.join(sys.path[0], '../'))
from utils import encode_action, preprocess_frame_bw_next_state

INPUT_DIM = (104,104,12) # 4 stacked frames
Z_DIM = 32
DENSE_SIZE = 1152
LEARNING_RATE = 0.0001
KL_TOLERANCE = 0.5
BATCH_SIZE = 16

def sampling(args):
    z_mean, z_sigma = args
    epsilon = K.random_normal(shape=K.shape(z_mean), mean=0., stddev=1.0)
    return z_mean +  z_sigma * epsilon

def convert_to_sigma(z_log_var):
    return K.exp(z_log_var / 2)

class AutoEncoder():
    def __init__(self, action_dim):
        self.input_dim = INPUT_DIM
        self.z_dim = Z_DIM
        self.action_dim = action_dim
        self.learning_rate = LEARNING_RATE
        self.kl_tolerance = KL_TOLERANCE   
        self.batch_size = BATCH_SIZE
        
        self.model = self._build() 

    def _build(self):
                
        vae_x = Input(shape=self.input_dim, name='observation_input')       
        
        ####################################################################
        #                        Step 1: CONVOLUTION                       #
        ####################################################################

        h = Conv2D(32, 6, strides=2, activation='relu')(vae_x)              
        h1 = Conv2D(64, 6, strides=2, activation='relu')(h)                 
        h2 = Conv2D(128, 6, strides=2, activation='relu')(h1)               
        h3 = Conv2D(128, 4, strides=2, activation='relu')(h2)               
        h4 = Flatten()(h3)                                                 

        ####################################################################
        #                     Step 2: ACTION ENCODING                      #
        ####################################################################

        action_input = Input(shape=(self.action_dim,), name='action_input')    # 4
        action_dense = Dense(DENSE_SIZE, name='action_dense')(action_input)
        action_transformation = multiply([h4, action_dense])

        ####################################################################
        #                     Step 3: DECONVOLUTION                        #
        ####################################################################

        decoder_reshape = Reshape((3,3,128), name='unflatten')(action_transformation)
        decoder = Conv2DTranspose(128, 5, strides=2, activation='relu')(decoder_reshape)
        decoder_2 = Conv2DTranspose(64, 6, strides=2, activation ='relu')(decoder)
        # padding = ZeroPadding2D(padding=(1,0))(decoder_2)        
        decoder_3 = Conv2DTranspose(32, 7, strides=2, activation ='relu')(decoder_2)
        # padding_2 = ZeroPadding2D(padding=(3,1))(decoder_3)
        # decoder_4 = Conv2DTranspose(32, 6, strides=2, activation ='relu')(decoder_3) 
        decoder_out = Conv2DTranspose(3, 8, strides=2, activation='sigmoid')(decoder_3)
        
        vae_full = Model([vae_x,action_input],decoder_out)

        ####################################################################
        #                        LOSS FUNCTIONS                            #
        ####################################################################

        def vae_r_loss(y_true, y_pred):
            r_loss = K.sum(K.square(y_true - y_pred), axis = [1,2,3])
            return r_loss

        def vae_loss(y_true, y_pred):
            return vae_r_loss(y_true, y_pred) 
        
        opti = Adam(lr=LEARNING_RATE)
        vae_full.compile(optimizer=opti, loss = vae_loss,  metrics = [vae_r_loss])

        # print(vae_full.summary())
        return (vae_full)

    ####################################################################
    #                          Step 4: TRAINING                        #
    ####################################################################    

    def train(self, obs, action=None, next_frame=None):
        if action is None:
            self.model.fit(x=obs,y=obs, 
                           shuffle=True, 
                           epochs=1, 
                           batch_size=self.batch_size)
        else:
            input_data = [obs, action]
            self.model.fit(x=input_data, y=next_frame,
                    shuffle=True,
                    epochs=1,
                    batch_size=self.batch_size)
    
    def set_weights(self, filepath):
        self.model.load_weights(filepath)
    
    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def predict(self, observation, action=None):
        if action is not None:
            return self.model.predict([observation,action])
        else:
            return self.model.predict(observation)
    
    def print_layer_shapes(self):
        self.model.summary()
    
    def generate_output_states(self,input_state):
        next_states = []
        # Generating next states states using autoencoder
        for i in range(self.action_dim):
            ohe_action = encode_action(self.action_dim, i)
            ohe_action = np.expand_dims(ohe_action, axis=0)     

            predicted_next = self.predict(input_state, ohe_action)
            predicted_next = (predicted_next[0,:,:,:]*255.).astype(np.uint8)
            next_states.append(preprocess_frame_bw_next_state(predicted_next))

        return np.stack(next_states,axis=2)
