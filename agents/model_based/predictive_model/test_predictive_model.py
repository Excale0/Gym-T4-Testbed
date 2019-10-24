import argparse
import cv2
import sys
import numpy as np
import os
import imageio
from generate_gif import create_gif

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROLLOUT_DIR = os.path.join(PARENT_DIR, "data")
IMAGE_DIR = os.path.join(PARENT_DIR, "images")

# Need to insert parent folder in positiion 0 of path here instead of 1 so that the load_predictive_model import 
# works in this script and in next_agent/generate_agent_date.py
sys.path.insert(0, os.path.join(sys.path[0], '../'))

from load_predictive_model import load_predictive_model

def diff(original,predicted):
    return abs(original - predicted)

def main(args):
    env_name = args.env_name
    informed = args.informed

    env_images_folder = IMAGE_DIR + "/" + env_name

    if not os.path.exists(env_images_folder):
        os.umask(0o000)
        os.makedirs(env_images_folder)

    if informed:
        rollout_file = ROLLOUT_DIR + '/informed_rollout_' + args.env_name + '/rollout-1.npz' # change the file number as needed
    else:
        rollout_file = ROLLOUT_DIR + '/random_rollout_' + args.env_name + '/rollout-1.npz'

    obs_data = np.load(rollout_file)['obs']
    action_data = np.load(rollout_file)['actions']
    next_data = np.load(rollout_file)['next_frame']

    predictive_model = load_predictive_model(env_name, len(action_data[0])) 

    for i in range(len(obs_data)):
        obs = obs_data[i]
        action = action_data[i]
        ground_truth = next_data[i]*255.

        current_frame = obs[:, :, :3]*255.

        obs = np.expand_dims(obs, axis=0)
        action = np.expand_dims(action, axis=0)

        predicted_next = predictive_model.predict(obs,action)

        # Have to use the extra layer and multiply rgb values by 255 to get the original image since before we divided by 255 during storage
        predicted_image = predicted_next[0, :, :, :]*255.

        triple = np.concatenate(
            (current_frame, predicted_image, ground_truth, diff(predicted_image, ground_truth)), axis=1)
        
        if i == 50:
            cv2.imwrite(PARENT_DIR + "/{}_50.jpg".format(env_name), triple)

        cv2.imwrite(env_images_folder + '/%03d.png' % i, triple)

    create_gif(env_images_folder, env_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Test predictive model'))
    parser.add_argument('--env_name', type=str, help='name of environment', default="PongDeterministic-v4")
    parser.add_argument('--informed', action='store_true', help='if true, will attempt to test on informed rollouts instead of random rollouts')
    args = parser.parse_args()
    
    main(args)