""""
this is the universal run script for all environments

"""
# import dependencies
import argparse
from argparse import RawTextHelpFormatter
import sys
import numpy as np
import gym
# for graphing
from utils.summary import Summary
import time
import datetime
# recording environment render as video mp4
# allows gifs to be saved of the training episode for use in the Control Center.

# ============================================

# For more on how argparse works see documentation
# create argument options
parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument("-alg", "--algorithm", help="select a algorithm: \n QLearning \n DQN \n DoubleDQN \n DuellingDQN \n DDDQN")
parser.add_argument("-env","--environment", help="select a environment: \n Pong-v0 \n SpaceInvaders-v0 \n MsPacman-v0")
parser.add_argument("-eps","--episodes", help="select number of episodes to graph")
# parser.add_argument("-steps", help="select number of steps")

# retrieve user inputted args from cmd line
args = parser.parse_args()

# Preprocessing folder
# this takes care of the environment specif7.8626167934090ics and image proccessing
if args.environment == 'Pong-v0':
    import utils.preprocessing.Pong_Preprocess as preprocess
    print('Pong works')
elif args.environment == 'SpaceInvaders-v0':
    import utils.preprocessing.SpaceInvaders_Preprocess as preprocess
    print('SpaceInvaders works')
elif args.environment == 'MsPacman-v0':
    import utils.preprocessing.MsPacman_Preprocess as preprocess
    print('MsPacman works')
elif args.environment == 'Breakout-v0':
    import utils.preprocessing.Breakout_Preprocess as preprocess
    print('Breakout works')
elif args.environment == 'CartPole-v1':
    import utils.preprocessing.Cartpole_Preprocess as preprocess
    print('Cartpole works')
else :
    sys.exit("Environment not found")


# algorithm folder
# the newtork is imported into brain file in the header so no need to import the network here aswell
if args.algorithm == 'QLearning':
    print('Q tables work')
elif args.algorithm == 'DQN':
    print('DQN works')
elif args.algorithm == 'DoubleDQN':
    print('Double works')
elif args.algorithm == 'DuellingDQN':
    print('Dueling works')
elif args.algorithm == 'DDDQN':
    print('PER works')
else :
    sys.exit("Algorithm not found")

# ============================================

# create gym env
env = gym.make(args.environment)
# initialise processing class specific to enviornment
processor = preprocess.Processor()
# state space is determined by the deque storing the frames from the env
state_space = processor.get_state_space()
if args.environment == 'CartPole-v1':
    state_space = env.observation_space.shape[0]
    # print("Goes into if loop")
# action space given by the environment
action_space = env.action_space.n

# print(state_space)

# print(action_space)

#**********************************************************************#
#if you want to look if there's any useless keys print the stuff below

# what_actions_do = env.unwrapped.get_action_meanings()
# print(what_actions_do)
#***********************************************************************#\

# here we change the action space if it contains 'useless' keys or actions that do the same thing
# if no useless keys it just returns the envs defined action space
# This function is created in the preprocess file
action_space=processor.new_action_space(action_space)
# initialise the algorithm class which also contains the network
learner = Brain.Learning(state_space, action_space)

# ============================================

# Graphing results
now = datetime.datetime.now()
MODEL_FILENAME = args.environment + '_' + args.algorithm + '_'
# our graphing function
#summary sets the ranges and targets and saves the graph
graph = Summary(summary_types = ['sumiz_step', 'sumiz_time', 'sumiz_reward', 'sumiz_epsilon'],
                # the optimal step count of the optimal policy
                step_goal = 0,
                # the maximum reward for the optimal policy
                reward_goal = 0,
                # maximum exploitation value
                epsilon_goal = 0.99,
                # desired name for file
                name=MODEL_FILENAME + str(now),
                # file path to save graph. i.e "/Desktop/Py/Scenario_Comparasion/Maze/Model/"
                save_path="/Gym-T4-Testbed/output/graphs/",
                # episode upper bound for graph
                episode_max= int(args.episodes),
                # step upper bound for graph
                step_max_m= processor.step_max,
                # time upper bound for graph
                time_max_m= processor.time_max,
                # reward upper bound for graph
                reward_min_m= processor.reward_min,
                # reward lower bound for graph
                reward_max_m= processor.reward_max
                )

# =================================================

DISCOUNTED_REWARDS_FACTOR=0.99
# ==================================================

# storing neural network weights and parameters
# SAVE_MODEL = True
# LOAD_MODEL = True
# if LOAD_MODEL == True:
#     neuralNet.model.save_weights(neuralNet.model.save_weights('./output/models/' + MODEL_FILENAME+ 'model.h5'))

# ============================================
print("\n ==== initialisation complete, start training ==== \n")

reward_episode =[]
for episode in range(int(args.episodes)):
    # storing frames as gifs, array emptied each new episode
    episode_frames = []

    observation = env.reset()
    episode_frames.append(observation)
    # Processing initial image cropping, grayscale, and stacking 4 of them
    observation = processor.Preprocessing(observation, True)

    start_time = time.time()
    sum_rewar_array = 0 #total rewards for graphing


    game_number = 0 # increases every time a someone scores a point
    game_step = 0 #for discounted rewards, steps for each round
    step=0 #count total steps for each episode for the graph

    # these arrays are used to calculated and store discounted rewards
    # arrays for other variable are needed for appending to transitions in our learner to work
    # arrays emptied after every round in an episode
    reward_array=[]
    if episode % 20 == 0:
        reward_episode =[]
    states=[]
    actions=[]
    next_states=[]
    dones=[]

    while True:
        if (episode > 150) and (args.environment == 'CartPole-v1'):
            env.render()
        #action chooses from  simplified action space without useless keys
        action = learner.choose_action(observation, episode)
        # actions map the simp,ified action space to the environment action space
        # if action space has no useles keys then action = action_mapped
        action_mapped = processor.mapping_actions_to_keys(action)

        # takes a step
        next_observation, reward, done, _ = env.step(action_mapped)
        
        episode_frames.append(next_observation)

        if args.environment == 'CartPole-v1':
            # punish if terminal state reached
            if done:
                reward = -reward
        # print('reward = ', reward)
        # appending <s, a, r, s', d> into arrays for storage
                
        states.append(observation)
        actions.append(action) # only append the '1 out of 3' action

        reward_array.append(reward)
        sum_rewar_array += reward
        reward_episode.append(sum_rewar_array)

        next_observation = processor.Preprocessing(next_observation, False)
        next_states.append(next_observation)
        dones.append(done)

        game_step += 1
        step+=1

        if done:
            avg_reward = np.mean(reward_episode)
            # append each <s, a, r, s', d> to learner.transitons for each game round
            for i in range(game_step):
                # print(i) 
                # print(states[i])
                learner.transitions.append((states[i], actions[i], reward_array[i],next_states[i],dones[i]))
                # print('reward = ', reward_array[i])
                # print(len(learner.transitions))

            print('Completed Episode = ' + str(episode), ' epsilon =', "%.4f" % learner.epsilon, ', steps = ', step)
            # print( ' avg reward = ', "%.4f" % avg_reward)
            # print('\n')

            # empty arrays after each round is complete
            states, actions, reward_episode, next_states, dones  = [], [], [], [], []
            # record video of environment render
            # env = gym.wrappers.Monitor(env,directory='Videos/' + MODEL_FILENAME + '/',video_callable=lambda episode_id: True, force=True,write_upon_reset=False)

            break


        observation = next_observation
        
        if args.environment == 'CartPole-v1':
            # train algorithm using experience replay
            learner.memory_replay(episode)
        
    # make gif
    # if episode != 0 and episode % 5 == 0:
    #     images = np.array(episode_frames)
    #     fname = './output/gifs/episode'+str(episode)+'.gif'
    #     with imageio.get_writer(fname, mode='I') as writer:
    #         for frame in images:
    #             writer.append_data(frame)

    # store model weights and parameters when episode rewards are above a certain amount
    # and after every number of episodes

    # if (SAVE_MODEL == True and episode % 5 == 0):
    #     neuralNet.model.save_weights('./output/models/' + MODEL_FILENAME+ 'model.h5', overwrite = True)

    # summarize plots the graph
    graph.summarize(episode, step, time.time() - start_time, sum_rewar_array, learner.epsilon, learner.e_greedy_formula)
# killing environment to prevent memory leaks
env.close()