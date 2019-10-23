from state_agent import StateAgent
import sys
import os
import numpy as np
import cv2
import gym
import matplotlib.pyplot as plt
import time
from collections import deque
from gym.envs.classic_control import rendering

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from predictive_model.load_predictive_model import load_predictive_model
from utils import preprocess_frame, encode_action, preprocess_frame_dqn
from simple_dqn import Agent


plt.rcParams.update({'font.size': 35})
plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['figure.figsize'] = [16.0,12.0]

def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0: 
        if not err: 
            print("Number of repeats must be larger than 0, k: {}, l: {}, returning default array!").format(k, l)
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)

def test_against_environment(env_name,num_runs,agent_name):
    env = gym.make(env_name)
    # env.seed(0)
    weights_path = "./agent_weights_pong_max.h5"
    try:
        predictor = load_predictive_model(env_name, env.action_space.n)
        if agent_name == 'Next_agent':
            agent = StateAgent(env.action_space.n, env_name)
            agent.set_weights(weights_path)
        elif agent_name == 'DQN':
            agent = Agent(gamma=0.99, epsilon=0.00, alpha=0.0001,
                  input_dims=(104,80,4), n_actions=env.action_space.n, mem_size=25000,
                  eps_min=0.00, batch_size=32, replace=1000, eps_dec=1e-5,
                  q_eval_fname='PongDeterministic-v4_q_network.h5', q_target_fname='PongDeterministic-v4_q_next.h5',
                  env_name=env_name)
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
            
            env.render()
            observation, reward, done, _ = env.step(agent_action)
            # print(agent_action)
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
    fig.savefig('graphs/{}_scores.png'.format(env_name))

def plot_time(env_name, num_runs, names, times, timing_steps):
    env_name = env_name.replace('Deterministic-v4','')
    title = 'Average time for {} steps in {}'.format(timing_steps,env_name) 
    fig, ax = plt.subplots()
    ax.set_ylabel('Time(seconds)')
    ax.bar(names,times, color=['#64B5F6','#BA68C8'])
    fig.suptitle(title)
    fig.savefig('graphs/{}_times.png'.format(env_name))

def test_plot():
    env_name = 'PongDeterministic-v4'
    num_runs = 5
    names = ['Random', 'DQN', 'Next State Agent']
    scores = [21,15,-21]
    plot_scores(env_name, num_runs, names, scores)




if __name__ == "__main__":

    num_runs = 1000
    env_name = "PongDeterministic-v4"

    reward, time_next_agent = test_against_environment(env_name,num_runs,'Next_agent')
    # test_plot()
