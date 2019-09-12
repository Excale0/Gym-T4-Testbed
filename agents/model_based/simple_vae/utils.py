import numpy as np
from PIL import Image

def encode_action(size, action):
    action_vector = [ 0 for i in range(size) ]
    action_vector[action] = 1
    return action_vector

def preprocess_frame(frame):
    # convert image to greyscale, downsize
    converted_obs = Image.fromarray(frame, 'RGB')
    # converted_obs = converted_obs.convert('L')  # to gray
    converted_obs = converted_obs.resize((80, 104), Image.ANTIALIAS)
    # converted_obs = converted_obs.crop((0,20,84,104))
    converted_obs = np.array(converted_obs).astype('float')
    converted_obs = np.pad(converted_obs,((0,0),(0,24),(0,0)), 'constant')
    return converted_obs/255.
