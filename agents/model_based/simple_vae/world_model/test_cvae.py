from simple_vae import CVAE
import cv2
import sys
import numpy as np
import os
import imageio
from generate_gif import create_gif
from load_world_model import load_world_model


def diff(original,predicted):
    return abs(original - predicted)
    
env_name = sys.argv[1]   

IMAGE_FOLDER = './images/%s/' % env_name

if not os.path.exists(IMAGE_FOLDER):
    os.umask(0o000)
    os.makedirs(IMAGE_FOLDER)

# right now its only using trained rollouts 
data_file = './data/world_models/' + 'trained/rollout_' + env_name + '/rollout-100.npz'

obs_data = np.load(data_file)['obs']
action_data = np.load(data_file)['actions']
next_data = np.load(data_file)['next_frame']

cvae = load_world_model(env_name, len(action_data[0])) 

for i in range(len(obs_data)):
    obs = obs_data[i]
    action = action_data[i]
    ground_truth = next_data[i]*255.
    # print(ground_truth.shape)

    current_frame = obs[:, :, :3]*255.
    # current_frame = np.expand_dims(current_frame, axis=3)

    obs = np.expand_dims(obs, axis=0)
    action = np.expand_dims(action, axis=0)

    predicted_next = cvae.predict(obs,action)

    # Have to use the extra layer and multiply rgb values by 255 to get the original image since before we divided by 255 during storage
    predicted_image = predicted_next[0, :, :, :]*255.
    # print(predicted_image.shape)

    triple = np.concatenate(
        (current_frame, predicted_image, ground_truth, diff(predicted_image, ground_truth)), axis=1)
    
    if i == 50:
        cv2.imwrite("./{}_50.jpg".format(env_name), triple)

    cv2.imwrite(IMAGE_FOLDER + '%03d.png' % i, triple)

create_gif(env_name)
