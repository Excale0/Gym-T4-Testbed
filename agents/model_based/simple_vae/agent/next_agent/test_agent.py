from state_agent import StateAgent
import sys
import os
import numpy as np
import cv2
import gym
from collections import deque

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from world_model.load_world_model import load_world_model
from utils import preprocess_frame, encode_action, preprocess_frame_dqn

weights_path = "./agent_weights.h5"

def test_against_rollout():
    
    data_file = "./data/rollout_PongDeterministic-v4/rollout-1.npz"
    data = np.load(data_file)
    next_states = data['next']
    correct = data['correct']

    print(next_states.shape)
    print(correct.shape)

    agent = StateAgent(6, 'PongDeterministic-v4')
    agent.set_weights(weights_path)

    correct_count = 0.0
    total_episodes = 400

    for i in range(total_episodes):
        current_next = next_states[i]
        current_correct = correct[i]
        current_correct = np.expand_dims(current_correct, axis=0)
        current_next = np.expand_dims(current_next, axis=0)

        prediction = agent.predict(current_next)
        # print(prediction)
        #print(current_correct)
        if np.argmax(prediction[0]) == np.argmax(current_correct):
            correct_count += 1

        print(correct_count/total_episodes)


def test_against_environment(num_runs):
    env_name = 'PongDeterministic-v4'
    env = gym.make(env_name)
    env.seed(0)

    agent = StateAgent(6, 'PongDeterministic-v4')
    agent.set_weights(weights_path)

    try:
        predictor = load_world_model(env_name, env.action_space.n)
    except:
        print ("Error loading model, check environment name and action space dimensions")
    
    rewards = []

    for i in range(num_runs):
        frame_queue = deque(maxlen=4)

        observation = env.reset()
        done = False

        init_queue(frame_queue,observation)
        total_reward = 0.0

        while not done:
            observation_states = np.concatenate(frame_queue, axis=2)
            next_states = predictor.generate_output_states(np.expand_dims(observation_states, axis=0))
            agent_action = agent.choose_action_from_next_states(np.expand_dims(next_states,axis=0))
            observation, reward, done, info = env.step(agent_action)
            
            total_reward += reward

            frame_queue.pop()
            frame_queue.appendleft(preprocess_frame(observation))
        print("Completed episode {} with reward {}".format(i+1, total_reward))
        rewards.append(total_reward)
    
    print("Test complete - Average score: {}    Max score: {}".format(np.average(rewards),np.max(rewards)))
    

def init_queue(queue, observation):
    for i in range(4):
        queue.append(preprocess_frame(observation))
    


if __name__ == "__main__":
    test_against_environment(5)
