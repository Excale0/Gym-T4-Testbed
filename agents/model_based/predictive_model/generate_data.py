import os
import sys
import numpy as np
import gym
import cv2
import random
from collections import deque
from PIL import Image

sys.path.insert(1, os.path.join(sys.path[0], '../'))

from utils import encode_action, preprocess_frame_dqn, preprocess_frame
from dqn_agent.load_dqn import load_dqn

#import matplotlib.pyplot as plt

import argparse

parent_dir = os.path.dirname(os.path.abspath(__file__))
ROLLOUT_DIR = os.path.join(parent_dir, "data")

def main(args):


    env_name = args.env_name
    total_episodes = args.total_episodes
    time_steps = args.time_steps
    informed = args.informed
    # action_refresh_rate = args.action_refresh_rate

    if informed:
        full_path = ROLLOUT_DIR + '/informed_rollout_' + args.env_name
    else:
        full_path = ROLLOUT_DIR + '/random_rollout_' + args.env_name

    if not os.path.exists(full_path):
        os.umask(0o000)
        os.makedirs(full_path)

    envs_to_generate = [env_name]

    for current_env_name in envs_to_generate:
        print("Generating data for env {}".format(current_env_name))

        env = gym.make(current_env_name) # Create the environment
        env.seed(0)

        s = 0

        if informed:
            agent = load_dqn(env)
        
        while s < total_episodes:
            
            rollout_file = os.path.join(full_path,  'rollout-%d.npz' % s) 

            observation = env.reset()
            frame_queue = deque(maxlen=4)
            dqn_queue = deque(maxlen=4)

            t = 0

            obs_sequence = []
            action_sequence = []
            next_sequence = []

            while t < time_steps:                  

                # convert image to greyscale, downsize
                converted_obs = preprocess_frame(observation)
                
                if t == 0:
                    for i in range(4):
                        frame_queue.append(converted_obs)
                else:
                    frame_queue.pop()
                    frame_queue.appendleft(converted_obs)
                
                stacked_state = np.concatenate(frame_queue, axis=2)
                obs_sequence.append(stacked_state)

                if informed:
                    dqn_obs = preprocess_frame_dqn(observation)
                    if t == 0:
                        for i in range(4):
                            dqn_queue.append(dqn_obs)
                    else:
                        dqn_queue.pop()
                        dqn_queue.appendleft(dqn_obs)
                    stacked = np.concatenate(dqn_queue, axis=2)
                    action = agent.choose_action(stacked)
                else:
                    action = env.action_space.sample()

                action_sequence.append(encode_action(env.action_space.n,action))

                observation, _, _, _ = env.step(action) # Take a random action  
                t = t + 1

                next_sequence.append(preprocess_frame(observation))

            print("Episode {} finished after {} timesteps".format(s, t))


            np.savez_compressed(rollout_file, obs=obs_sequence, actions=action_sequence, next_frame=next_sequence)

            s = s + 1

        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Create new training data for predictive model'))
    parser.add_argument('--env_name', type=str, help='name of environment', default="PongDeterministic-v4")
    parser.add_argument('--informed', action='store_true', 
                        help='if true, will use a trained agent to explore instead of random rollouts')
    parser.add_argument('--total_episodes', type=int, default=200,
                        help='total number of episodes to generate')
    parser.add_argument('--time_steps', type=int, default=200,
                        help='number of timesteps in every episode')

    args = parser.parse_args()
    main(args)