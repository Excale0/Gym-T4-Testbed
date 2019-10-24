import numpy as np
from PIL import Image

def encode_action(size, action):
    action_vector = [ 0 for i in range(size) ]
    action_vector[action] = 1
    return action_vector

def preprocess_frame(frame):
    converted_obs = Image.fromarray(frame, 'RGB')
    converted_obs = converted_obs.resize((80, 104), Image.ANTIALIAS) # downsize
    converted_obs = np.array(converted_obs).astype('float')
    converted_obs = np.pad(converted_obs,((0,0),(0,24),(0,0)), 'constant') # pad to make square
    return converted_obs/255.

def preprocess_frame_dqn(frame):
    converted_obs = Image.fromarray(frame, 'RGB') 
    converted_obs = converted_obs.convert('L')  # convert to grayscale
    converted_obs = converted_obs.resize((80, 104), Image.ANTIALIAS)
    converted_obs = np.array(converted_obs).astype('float')
    converted_obs = np.expand_dims(converted_obs, axis=2)
    return converted_obs/255.
    
def preprocess_frame_bw_next_state(frame):
    converted_obs = Image.fromarray(frame, 'RGB')
    converted_obs = converted_obs.convert('L')  # convert to grayscale
    converted_obs = converted_obs.crop((0,0,80,104)) # crop
    converted_obs = np.array(converted_obs).astype('float')
    return converted_obs/255.

