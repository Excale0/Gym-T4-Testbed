# This file is used to generate training data for the next state agent
# It requires:
#   - A predictive autoencoder
#   - A capable agent in the environment of training
import os
import sys
import numpy as np
import gym
import cv2
import random
from collections import deque
from PIL import Image
import argparse

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from world_model.load_world_model import load_world_model
from utils import preprocess_frame, encode_action

ROLLOUT_DIR = './data/'

def generate_agent_episodes(args):
    full_path = ROLLOUT_DIR + 'rollout_' + args.env_name
    
    if not os.path.exists(full_path):
        os.umask(0o000)
        os.makedirs(full_path)

    env_name = args.env_name
    total_episodes = args.total_episodes
    time_steps = args.time_steps

    envs_to_generate = [env_name]

    for current_env_name in envs_to_generate:
        print("Generating data for env {}".format(current_env_name))

        env = gym.make(current_env_name) # Create the environment

        # First load the agent and the predictive auto encoder with their weights
        # agent().load_weights(x.h5)
        predictor = load_world_model(current_env_name,env.action_space.n)

        s = 0
        
        while s < total_episodes:
            
            rollout_file = os.path.join(full_path,  'rollout-%d.npz' % s) 

            observation = env.reset()
            frame_queue = deque(maxlen=4)
            
            t = 0

            next_state_sequence = []
            correct_state_sequence = []

            while t < time_steps:  
                # Get agent to sample action
                action = env.action_space.sample()
                
                # convert image to greyscale, downsize
                
                converted_obs = preprocess_frame(observation)
                
                if t == 0:
                    for i in range(4):
                        frame_queue.append(converted_obs)
                else:
                    frame_queue.pop()
                    frame_queue.appendleft(converted_obs)
                
                observation_states = np.concatenate(frame_queue, axis=2)
                next_states = predictor.generate_output_states(np.expand_dims(observation_states, axis=0))
                next_state_sequence.append(next_states)
                correct_state_sequence.append(encode_action(env.action_space.n,action))

                observation, _, _, _ = env.step(action) # Take a random action  
                t = t + 1

            print("Episode {} finished after {} timesteps".format(s, t))


            np.savez_compressed(rollout_file, next=next_state_sequence, correct=correct_state_sequence)

            s = s + 1

        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Create new training data'))
    parser.add_argument('--env_name', type=str, help='name of environment', default="Pong-v0")
    parser.add_argument('--total_episodes', type=int, default=100,
                        help='total number of episodes to generate per worker')
    parser.add_argument('--time_steps', type=int, default=200,
                        help='how many timesteps at start of episode?')
    parser.add_argument('--render', default=0, type=int,
                        help='render the env as data is generated')
    parser.add_argument('--run_all_envs', action='store_true',
                        help='if true, will ignore env_name and loop over all envs in train_envs variables in config.py')

    args = parser.parse_args()
    generate_agent_episodes(args)
