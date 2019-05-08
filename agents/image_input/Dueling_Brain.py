import datetime
import os
import random
import sys

import numpy as np

import agents.image_input.AbstractBrain as AbstractBrain
from agents.networks.dueling_dqn_networks import build_dueling_cartpole_network, build_dueling_dqn_network


class Learning(AbstractBrain.AbstractLearning):

    def __init__(self, observations, actions, config):
        super().__init__(observations, actions, config)
        # use network suitable for classic control games
        if config['environment'] == 'CartPole-v1':
            self.network = \
                build_dueling_cartpole_network(self.state_space, self.action_space, self.config['learning_rate'])
        # use network suitable for Atari games
        else:
            self.network = build_dueling_dqn_network(self.state_space, self.action_space, self.config['learning_rate'])

    def update_epsilon(self):
        if self.epsilon > self.config['epsilon_min']:
            self.epsilon = max(self.config['epsilon_min'], self.epsilon - self.epsilon_decay)

    def choose_action(self, state):
        if random.random() <= self.epsilon:
            # this explores by choosing a randomised action
            return random.randrange(self.action_space)
        else:
            # this exploits by choosing your max of your calculated q values
            return np.argmax(self.network.predict(np.expand_dims(state, axis=0)))

    def train_network(self, states, actions, rewards, next_states, dones, step):
        if step % self.config['network_train_frequency'] == 0:
            target = self.network.predict(states)
            target_next = self.network.predict(next_states)
            for i in range(self.config['batch_size']):
                if dones[i]:
                    target[i][actions[i]] = rewards[i]
                else:
                    # bellman equation
                    target[i][actions[i]] = rewards[i] + self.config['gamma'] * np.amax(target_next[i])

            self.network.fit(states, target, batch_size=len(dones), epochs=1, verbose=0)

        self.update_epsilon()

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.network.set_weights(self.network.get_weights())

    def save_network(self, save_path, model_name, timestamp=None):
        # create folder for model, if necessary
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # set timestamp if none was specified
        if timestamp is None:
            timestamp = str(datetime.datetime.now())
        # save model weights
        self.network.save_weights(save_path + model_name + '_' + timestamp + '.h5', overwrite=True)

    def load_network(self, save_path, model_name) -> None:
        if os.path.exists(save_path):
            self.network.load_weights(save_path + model_name)
            print('Loaded model ' + model_name + ' from disk')
        else:
            sys.exit("Model can't be loaded. Model file " + model_name + " doesn't exist at " + save_path + ".")
