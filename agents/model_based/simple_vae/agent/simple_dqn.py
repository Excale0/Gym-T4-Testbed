from keras.layers import Dense, Activation, Conv2D, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from world_model.simple_vae import CVAE
from world_model.load_world_model import load_world_model
from utils import encode_action
from collections import deque

ENV_NAME = 'PongDeterministic-v4'


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                      dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                          dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

def build_dqn(lr, n_actions, input_dims, fc1_dims):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=6, strides=2, activation='relu',
                     input_shape=(*input_dims,)))
    model.add(Conv2D(filters=64, kernel_size=6, strides=2, activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=6, strides=2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(fc1_dims, activation='relu'))
    model.add(Dense(n_actions))

    model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error')

    return model

class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, replace,
                 input_dims, eps_dec=0.996,  eps_min=0.01,
                 mem_size=1000000, q_eval_fname='Pong_q_network.h5',
                 q_target_fname='Pong_q_next.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.replace = replace
        self.q_target_model_file = q_target_fname
        self.q_eval_model_file = q_eval_fname
        self.learn_step = 0
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 512)
        self.q_next = build_dqn(alpha, n_actions, input_dims, 512)
        self.world_model = load_world_model(ENV_NAME, n_actions)

    def replace_target_network(self):
        if self.replace is not None and self.learn_step % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation], copy=False, dtype=np.float32)            
            policy = self.q_eval.predict(state)
            action = np.argmax(policy)
            # print(policy,action)

        return action

    def choose_action_aggregated(self, observation):
        if np.random.random() < self.epsilon:
            action_to_take = np.random.choice(self.action_space)
        else:
            observations = deque(maxlen=4)
            state = np.array([observation], copy=False, dtype=np.float32)
            aggregated_qs = np.zeros(len(self.action_space), dtype=np.float32)

            # Get Q values for current state

            policy = self.q_eval.predict(state)

            actions_explored = 0
            # Predict the next state for every action
            for action in self.action_space:

                # need to one hot encode and expand dims for world model prediction
                ohe_action = encode_action(len(self.action_space), action)
                ohe_action = np.expand_dims(ohe_action, axis=0)     

                predicted_next = self.world_model.predict(state, ohe_action)
                predicted_next = predicted_next[0,:,:,:]

                observations = np.split(observation, 4, axis=2)

                # if exploring first action then remove last frame and add predicted frame to the start
                # if exploring subsequent actions remove the previous prediction (first frame) and replace with new prediction
                if actions_explored == 0:
                    observations.pop()
                    observations.insert(0,predicted_next)
                else:
                    observations.pop(0)
                    observations.insert(0,predicted_next)

                stacked_frames = np.concatenate(observations,axis=2)
                new_state = np.array([stacked_frames], copy=False, dtype=np.float32)

                # Get Q values for predicted state
                next_policy = self.q_eval.predict(new_state)

                # Aggregate Q values
                aggregated_policy = policy + self.gamma*next_policy

                # Find the highest q value from policy
                max_q = np.max(aggregated_policy)

                # Store as max aggregated q for taking action
                aggregated_qs[action] = max_q

                actions_explored += 1
            
            # Take the action which lead to the highest aggregated q
            action_to_take = np.argmax(aggregated_qs)

        return action_to_take

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = \
                                    self.memory.sample_buffer(self.batch_size)

            self.replace_target_network()

            q_eval = self.q_eval.predict(state)
            q_next = self.q_next.predict(new_state)
            q_next[done] = 0.0

            q_target = q_eval[:]

            indices = np.arange(self.batch_size)
            q_target[indices, action] = reward + \
                                        self.gamma*np.max(q_next,axis=1)

            self.q_eval.train_on_batch(state, q_target)

            self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min
            self.learn_step += 1

    def save_models(self):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.q_eval.save(os.path.join(dir_path, self.q_eval_model_file))
        self.q_next.save(os.path.join(dir_path, self.q_target_model_file))
        print('... saving models ...')

    def load_models(self):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.q_eval = load_model(os.path.join(dir_path, self.q_eval_model_file))
        self.q_nexdt = load_model(os.path.join(dir_path, self.q_target_model_file))
        print('... loading models ...')