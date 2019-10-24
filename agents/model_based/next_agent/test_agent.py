from state_agent import StateAgent
import sys
import os
import numpy as np
import cv2
import gym
import argparse
import matplotlib.pyplot as plt
import time
from collections import deque

sys.path.insert(1, os.path.join(sys.path[0], '../'))
from predictive_model.load_predictive_model import load_predictive_model
from utils import preprocess_frame, encode_action, preprocess_frame_dqn
from dqn_agent.simple_dqn import Agent

parent_dir = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR = os.path.join(parent_dir, "graphs")

plt.rcParams.update({'font.size': 35})
plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['figure.figsize'] = [16.0,12.0]

def test_against_environment(env_name,num_runs,agent_name):
    env = gym.make(env_name)
    # env.seed(0)
    try:
        predictor = load_predictive_model(env_name, env.action_space.n)
        if agent_name == 'Next_agent':
            agent = StateAgent(env.action_space.n, env_name)
            agent.set_weights()
        elif agent_name == 'DQN':
            agent = Agent(gamma=0.99, epsilon=0.00, alpha=0.0001,
                  input_dims=(104,80,4), n_actions=env.action_space.n, mem_size=25000,
                  eps_min=0.00, batch_size=32, replace=1000, eps_dec=1e-5, env_name=env_name)
            agent.load_models()
    except:
        print ("Error loading model, check environment name and action space dimensions")
    
    rewards = []

    start = time.time()

    total_steps = 0.0
    for i in range(num_runs):
        frame_queue = deque(maxlen=4)

        observation = env.reset()
        done = False

        if agent_name == 'DQN':
            init_queue(frame_queue,observation,True)
        else:
            init_queue(frame_queue,observation)

        total_reward = 0.0
        frame_count = 0
        while not done:
            observation_states = np.concatenate(frame_queue, axis=2)

            # Human start of breakout since the next state agent just keeps moving to the left
            if agent_name == 'Next_agent':
                if env_name == 'BreakoutDeterministic-v4' and not frame_count:
                    agent_action = 1
                else:
                    next_states = predictor.generate_output_states(np.expand_dims(observation_states, axis=0))
                    agent_action = agent.choose_action_from_next_states(np.expand_dims(next_states,axis=0))
            elif agent_name == 'DQN':
                agent_action = agent.choose_action(observation_states)
            else:
                agent_action = env.action_space.sample()
            
            observation, reward, done, _ = env.step(agent_action)
            total_reward += reward
            frame_count += 1
            total_steps += 1

            frame_queue.pop()
            if agent_name == 'DQN':
                frame_queue.appendleft(preprocess_frame_dqn(observation))
            else:
                frame_queue.appendleft(preprocess_frame(observation))

        print("Completed episode {} with reward {}".format(i+1, total_reward))
        rewards.append(total_reward)
    end = time.time()

    time_taken = (end-start) / total_steps
    
    print("Test complete - Average score: {}    Max score: {}".format(np.average(rewards),np.max(rewards)))
    return (rewards,time_taken)

def init_queue(queue, observation, dqn=False):
    for i in range(4):
        if dqn:
            queue.append(preprocess_frame_dqn(observation))
        else:
            queue.append(preprocess_frame(observation))

def save_graph(env_name, num_runs, timing_steps):
    reward_dqn, time_dqn = test_against_environment(env_name,num_runs,'DQN')
    reward_random, _ = test_against_environment(env_name,num_runs,'Random')
    reward, time_next_agent = test_against_environment(env_name,num_runs,'Next_agent')
    names = ['Random', 'DQN', 'Next State Agent']
    names_time = ['DQN', 'Next State Agent']
    scores = [ np.average(i) for i in [reward_random, reward_dqn, reward]]
    times = [time_dqn*timing_steps, time_next_agent*timing_steps]
    plot_scores(env_name, num_runs, names, scores)
    plot_time(env_name, num_runs, names_time, times, timing_steps)



def plot_scores(env_name, num_runs, names, scores):
    env_name = env_name.replace('Deterministic-v4','')
    title = 'Average score in {} ({} games)'.format(env_name,num_runs) 
    fig, ax = plt.subplots()
    ax.set_ylabel('Score')
    ax.bar(names, scores, color=['#7986CB','#64B5F6','#BA68C8'])
    fig.suptitle(title)
    fig.savefig(GRAPH_DIR + '/{}_scores.png'.format(env_name))

def plot_time(env_name, num_runs, names, times, timing_steps):
    env_name = env_name.replace('Deterministic-v4','')
    title = 'Average time for {} steps in {}'.format(timing_steps,env_name) 
    fig, ax = plt.subplots()
    ax.set_ylabel('Time(seconds)')
    ax.bar(names,times, color=['#64B5F6','#BA68C8'])
    fig.suptitle(title)
    fig.savefig(GRAPH_DIR + '/{}_times.png'.format(env_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Test next state agent'))
    parser.add_argument('--env_name', type=str, help='name of environment', default="PongDeterministic-v4")
    args = parser.parse_args()
    
    num_runs = 2
    save_graph(args.env_name,num_runs,100)
