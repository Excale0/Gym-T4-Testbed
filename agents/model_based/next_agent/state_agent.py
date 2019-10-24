import sys
import os
import numpy as np
from keras.layers import Conv2D, Dense, Input, Flatten
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

sys.path.insert(1, os.path.join(sys.path[0], '../'))
from predictive_model.load_predictive_model import load_predictive_model

INPUT_DIM = (80,104,1)
file_path = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(file_path, "models/")


class StateAgent():
    def __init__(self,action_dim,env_name):
        self.action_dim = action_dim
        self.model = self.build_model(action_dim)
        self.predictive_model = load_predictive_model(env_name,action_dim)

        if not os.path.exists(MODEL_PATH):
            os.umask(0o000)
            os.makedirs(MODEL_PATH)

        model_name = '%s_next_agent_weights.h5' % env_name
        self.model_file = os.path.join(MODEL_PATH, model_name)

    def build_model(self, action_dim):
        frame_input = Input(shape=(INPUT_DIM[1],INPUT_DIM[0],INPUT_DIM[2]*self.action_dim))
        
        conv_1 = Conv2D(filters=32, kernel_size=5, strides=2, activation='relu')(frame_input)
        conv_2 = Conv2D(filters=64, kernel_size=5, strides=2, activation='relu')(conv_1)
        conv_3 = Conv2D(filters=64, kernel_size=5, strides=2, activation='relu')(conv_2)
        conv_4 = Conv2D(filters=128, kernel_size=5, strides=2, activation='relu')(conv_3)
        flatten = Flatten(name='flatten')(conv_4)
        dense_1 = Dense(512)(flatten)
        dense_3 = Dense(256, activation='relu')(dense_1)
        output = Dense(action_dim, activation='softmax')(dense_3)

        optimizer = Adam(lr=0.0001)
        model = Model(frame_input, output)

        def model_loss(y_true,y_pred):
            return K.sum(K.square(y_true-y_pred))

        model.compile(optimizer=optimizer, loss=model_loss, metrics=[model_loss])
        return model
    
    def train(self, input_states, output_label,epochs):
        self.model.fit(x=input_states,
                       y=output_label,
                       shuffle=True,
                       epochs=epochs,
                       batch_size=8)
    
    def set_weights(self):
        self.model.load_weights(self.model_file)
    
    def save_weights(self):
        self.model.save_weights(self.model_file)

    def predict(self, input_states):
        return self.model.predict(input_states)
    
    def choose_action_from_next_states(self, next_states):
        return np.argmax(self.predict(next_states))
        





