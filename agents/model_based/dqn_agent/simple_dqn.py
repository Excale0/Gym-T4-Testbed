from keras.layers import Dense, Activation, Conv2D, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
import sys
import os
from pathlib import Path
from collections import deque


parent_path = os.path.join(sys.path[0], '../')
sys.path.insert(1, parent_path)

from utils import encode_action

file_path = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(file_path, "models/")

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

def build_network(lr, n_actions, input_dims, fc1_dims):

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
                 mem_size=1000000,env_name='BreakoutDeterministic-v4'):
                
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.replace = replace

        q_eval_name = '%s_q_network.h5' % env_name
        q_target_name = '%s_q_next.h5' % env_name

        if not os.path.exists(MODEL_PATH):
            os.umask(0o000)
            os.makedirs(MODEL_PATH)

        self.q_eval_model_file = os.path.join(MODEL_PATH, q_eval_name)
        self.q_target_model_file = os.path.join(MODEL_PATH, q_target_name)

        self.learn_step = 0
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_network(alpha, n_actions, input_dims, 512)
        self.q_target = build_network(alpha, n_actions, input_dims, 512)

    def replace_target_network(self):
        if self.replace is not None and self.learn_step % self.replace == 0:
            self.q_target.set_weights(self.q_eval.get_weights())

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation], copy=False, dtype=np.float32)            
            policy = self.q_eval.predict(state)
            action = np.argmax(policy)

        return action

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

            self.replace_target_network()

            q_eval = self.q_eval.predict(state)
            q_next = self.q_target.predict(new_state)
            q_next[done] = 0.0

            q_target = q_eval[:]

            indices = np.arange(self.batch_size)
            q_target[indices, action] = reward + self.gamma*np.max(q_next,axis=1)

            self.q_eval.train_on_batch(state, q_target)

            self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
            self.learn_step += 1

    def save_models(self):
        self.q_eval.save(self.q_eval_model_file)
        self.q_target.save(self.q_target_model_file)
        print('... saving models ...')

    def load_models(self):
        self.q_eval = load_model(self.q_eval_model_file)
        self.q_target = load_model(self.q_target_model_file)
        print('... loading models ...')