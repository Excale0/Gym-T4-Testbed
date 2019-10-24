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

sys.path.insert(1, os.path.join(sys.path[0], '../'))
from predictive_model.load_predictive_model import load_predictive_model
from utils import preprocess_frame, encode_action, preprocess_frame_dqn
from dqn_agent.simple_dqn import Agent

folder_path = os.path.dirname(os.path.abspath(__file__))
ROLLOUT_DIR = os.path.join(folder_path, "data")

def generate_agent_episodes(args):

    full_path = ROLLOUT_DIR + '/rollout_' + args.env_name
    
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
        env.seed(0)

        # First load the DQN agent and the predictive auto encoder with their weights
        agent = Agent(gamma=0.99, epsilon=0.0, alpha=0.0001,
                input_dims=(104,80,4), n_actions=env.action_space.n, mem_size=25000,
                eps_min=0.0, batch_size=32, replace=1000, eps_dec=1e-5, env_name=current_env_name)
        agent.load_models()

        predictor = load_predictive_model(current_env_name,env.action_space.n)

        s = 0
        
        while s < total_episodes:
            
            rollout_file = os.path.join(full_path,  'rollout-%d.npz' % s) 

            observation = env.reset()
            frame_queue = deque(maxlen=4)
            dqn_queue = deque(maxlen=4)
            
            t = 0

            next_state_sequence = []
            correct_state_sequence = []
            total_reward = 0
            while t < time_steps:  
                # preprocess frames for predictive model and dqn                
                converted_obs = preprocess_frame(observation)
                converted_obs_dqn = preprocess_frame_dqn(observation)
                
                if t == 0:
                    for i in range(4):
                        frame_queue.append(converted_obs)
                        dqn_queue.append(converted_obs_dqn)
                else:
                    frame_queue.pop()
                    dqn_queue.pop()
                    frame_queue.appendleft(converted_obs)
                    dqn_queue.appendleft(converted_obs_dqn)
                
                observation_states = np.concatenate(frame_queue, axis=2)
                dqn_states = np.concatenate(dqn_queue,axis=2)
                next_states = predictor.generate_output_states(np.expand_dims(observation_states, axis=0))
                next_state_sequence.append(next_states)
                action = agent.choose_action(dqn_states)
                correct_state_sequence.append(encode_action(env.action_space.n,action))

                observation, reward, done, info = env.step(action) # Take a random action  
                total_reward += reward
                t = t + 1

            print("Episode {} finished after {} timesteps with reward {}".format(s, t, total_reward))

            np.savez_compressed(rollout_file, next=next_state_sequence, correct=correct_state_sequence)

            s = s + 1

        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Create new training data'))
    parser.add_argument('--env_name', type=str, help='name of environment', default="PongDeterministic-v4")
    parser.add_argument('--total_episodes', type=int, default=80,
                        help='total number of episodes to generate')
    parser.add_argument('--time_steps', type=int, default=400,
                        help='number of timesteps per episode')

    args = parser.parse_args()
    generate_agent_episodes(args)
