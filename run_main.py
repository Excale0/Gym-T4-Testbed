"""
this is the universal run script for all environments

"""
import argparse
import datetime
import json
import sys
from argparse import RawTextHelpFormatter
from os.path import expanduser
from random import seed
import gym
from tensorflow import set_random_seed

from training.training_functions import train
from utils.summary import Summary

# add random seed and tensorflow seed to make results reproducible
seed(0)
set_random_seed(0)


# PATH = expanduser("~")
# Hardcoded file path for docker image
PATH = '/home'
MODEL_FILENAME = ''


# TODO: add method comments
# TODO: add replay mode (load previous model and let it act in environments without learning)
# TODO: add DDDQN with PER as comparison option
# TODO: update model-saving-option (store first episode, best episode from n steps)
# TODO: reduce number of optional config parameters used (e.g. target_update_frequency)

if __name__ == "__main__":

    # For more on how argparse works see documentation
    # create argument options
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("-file", "--filename", help='name of file containing parameters in json-format',)

    # retrieve user inputted args from cmd line
    args = parser.parse_args()

    # Read JSON data into the datastore variable
    config_file_path = PATH + args.filename
    if config_file_path:
        with open(config_file_path, 'r') as f:
            config = json.load(f)

# ============================================================================================================== #

    # Prepossessing folder
    # this takes care of the environment specifics and image processing
    if config['environment'] == 'Pong-v0':
        import utils.preprocessing.Pong_Preprocess as Preprocess
        MODEL_FILENAME = MODEL_FILENAME + "Pong"
        print('Pong works')
    elif config['environment'] == 'PongDeterministic-v4':
        import utils.preprocessing.Pong_Preprocess as Preprocess
        MODEL_FILENAME = MODEL_FILENAME + "PongDeterministic"
        print('Pong Deterministic works')
    elif config['environment'] == 'SpaceInvaders-v0':
        import utils.preprocessing.SpaceInvaders_Preprocess as Preprocess
        MODEL_FILENAME = MODEL_FILENAME + "SpaceInvaders"
        print('SpaceInvaders works')
    elif config['environment'] == 'MsPacman-v0':
        import utils.preprocessing.MsPacman_Preprocess as Preprocess
        MODEL_FILENAME = MODEL_FILENAME + "MsPacman"
        print('MsPacman works')
    elif config['environment'] == 'Breakout-v0':
        import utils.preprocessing.Breakout_Preprocess as Preprocess
        MODEL_FILENAME = MODEL_FILENAME + "Breakout"
        print('Breakout works')
    elif config['environment'] == 'Enduro-v0':
        import utils.preprocessing.Enduro_Preprocess as Preprocess
        MODEL_FILENAME = MODEL_FILENAME + "Enduro"
        print('Breakout works')
    elif config['environment'] == 'CartPole-v1':
        import utils.preprocessing.Cartpole_Preprocess as Preprocess
        MODEL_FILENAME = MODEL_FILENAME + "CartPole"
        print('Cartpole works')
    else:
        sys.exit("Environment not found")

    # create gym env
    env = gym.make(config['environment'])
    # add seed to make results reproducible
    env.seed(0)

    # ============================================================================================================== #

    # initialise processing class specific to environment
    processor = Preprocess.Processor()
    # state space is determined by the deque storing the frames from the env
    state_space = processor.get_state_space()

    if config['environment'] == 'CartPole-v1':
        state_space = (env.observation_space.shape[0],)

    # action space given by the environment
    action_space = env.action_space.n

    # ============================================================================================================== #

    # algorithm folder
    if config['algorithm'] == 'DQN':
        # import chosen Agent
        from agents.image_input.DQN_Brain import Learning
        # import memory for agent
        from agents.memory.Memory import RandomBatchMemory as Memory
        # create memory
        memory = Memory(config['memory_size'], state_space)
        # set path for output data
        PATH = PATH + '/output/DQN/'
        print('DQN works')
    elif config['algorithm'] == 'DoubleDQN':
        from agents.image_input.Double_DQN_Brain import Learning
        from agents.memory.Memory import RandomBatchMemory as Memory
        memory = Memory(config['memory_size'], state_space)
        PATH = PATH + '/output/DoubleDQN/'
        print('Double works')
    elif config['algorithm'] == 'DuelingDQN':
        from agents.image_input.Dueling_Brain import Learning
        from agents.memory.Memory import RandomBatchMemory as Memory
        memory = Memory(config['memory_size'], state_space)
        PATH = PATH + '/Gym-T4-Testbed/output/DuelingDQN/'
        print('Dueling works')
    elif config['algorithm'] == 'ActorCritic':
        from agents.image_input.Actor_Critic_Brain import Learning
        from agents.memory.Memory import RandomBatchMemory as Memory
        memory = Memory(config['memory_size'], state_space)
        PATH = PATH + '/Gym-T4-Testbed/output/ActorCritic/'
        print('Actor Critic works')
    elif config['algorithm'] == 'A2C':
        from agents.image_input.A2C_Brain import Learning
        from agents.memory.Memory import EpisodicMemory as Memory
        memory = Memory(config['memory_size'], state_space, action_space)
        PATH = PATH + '/Gym-T4-Testbed/output/A2C/'
        print('A2C works')
    elif config['algorithm'] == 'PolicyGradient':
        from agents.image_input.Policy_Gradient_Brain import Learning
        from agents.memory.Memory import EpisodicMemory as Memory
        memory = Memory(config['memory_size'], state_space, action_space)
        PATH = PATH + '/Gym-T4-Testbed/output/PolicyGradient/'
        print('Policy Gradient works')
    elif config['algorithm'] == 'PPO':
        from agents.image_input.PPO_Brain import Learning
        from agents.memory.Memory import EpisodicMemory as Memory
        memory = Memory(config['memory_size'], state_space, action_space)
        PATH = PATH + '/Gym-T4-Testbed/output/PPO/'
        print('ProximalPolicyOptimization works')
    else:
        sys.exit("Algorithm not found")

    learner = Learning(state_space, action_space, config)

    # ============================================================================================================== #

    if config['environment'] == 'DQN' or config['environment'] == 'DoubleDQN' or config['environment'] == 'DuelingDQN':
        plots = ['sumiz_step', 'sumiz_reward', 'sumiz_epsilon', 'sumiz_time']
    else:
        plots = ['sumiz_step', 'sumiz_reward', 'sumiz_time']

    summary = Summary(plots,
                      name=MODEL_FILENAME + str(datetime.datetime.now()),
                      save_path=PATH + '/graphs/',
                      min_reward=processor.reward_min,
                      max_reward=processor.reward_max)

    # ============================================================================================================== #

    # train learner and plot results
    train(env, learner, memory, processor, config, PATH, summary=summary)
