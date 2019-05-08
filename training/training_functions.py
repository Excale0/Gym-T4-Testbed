import copy
import datetime
import time
from os.path import expanduser

import tensorflow

from agents.image_input.AbstractBrain import AbstractLearning
from agents.memory import Memory
from training.testing_functions import test
from utils.preprocessing.Pong_Preprocess import Processor
from utils.storing import make_gif, save_episode_to_summary

home = expanduser("~")

# save summary-plot after n episodes
summary_save_step = 10000
test_frequency = 10000


def train(env: any, learner: AbstractLearning, memory: Memory, processor: Processor,
          config, save_path, summary=None) -> None:
    """
    Trains learner in env and plots results
    :param env: gym environment
    :param learner: untrained learner
    :param memory: memory for storing experiences as tuples containing state, action, reward, next_state, done
    :param processor: pre-processor for given environment
    :param config: configurations for training
    :param save_path: path to folder: [home]/Gym-T4-Testbed/output/[algorithm]/
    :param summary: optional summary to save plots during training
    """

    # loading neural network weights and parameters
    if config['load_model']:
        learner.load_network(save_path + 'models/', config['model_file'])

    summary_writer = tensorflow.summary.FileWriter(save_path + 'tensorboard_summary/')

    print("\n ==== initialisation complete, start training ==== \n")

    # ============================================================================================================== #

    # keeping track of best episode within the last store_frequency steps
    max_reward = -1
    max_episode_number = -1
    max_episode_frames = []
    # data for summary-plots
    summary_rewards, summary_epsilons, summary_steps = [], [], []

    # ============================================================================================================== #

    state = env.reset()
    # for storing frames as gifs, array emptied each new episode
    episode_frames = [state]
    state = processor.process_state_for_memory(state, True)
    episode_start_time = time.time()
    episode_reward = 0
    episode_step = 0
    episode = 0

    # for episode in range(int(episodes)):
    for step in range(config['steps']):

        # action chooses from  simplified action space without useless keys
        action = learner.choose_action(processor.process_state_for_network(state))
        # takes a step
        next_state, reward, done, _ = env.step(action)

        # add new frame to use for gif
        episode_frames.append(next_state)

        if 'reward_clipping' in config and config['reward_clipping']:
            reward = processor.process_reward(reward)
        next_state = processor.process_state_for_memory(next_state, False)

        if config['environment'] == 'CartPole-v1':
            # punish if terminal state reached
            if done:
                reward = -reward

        # append <s, a, r, s', d> to learner.transitions
        memory.store_transition(state, action, reward, next_state, done)

        # train algorithm using experience replay
        if step >= config['initial_exploration_steps'] \
                and config['algorithm'] != 'A2C' and config['algorithm'] != 'PolicyGradient':
            states, actions, rewards, next_states, dones = memory.sample(config['batch_size'], processor)
            learner.train_network(states, actions, rewards, next_states, dones, step)

        episode_reward += reward
        episode_step += 1
        state = next_state

    # ============================================================================================================== #

        if done:
            # train algorithm with data from one complete episode
            if config['algorithm'] == 'A2C' or config['algorithm'] == 'PolicyGradient':
                states, actions, rewards, next_states, dones = memory.sample_all(processor)
                learner.train_network(states, actions, rewards, next_states, dones, step)

            print('Completed Episode = ' + str(episode),
                  ' epsilon =', "%.4f" % learner.epsilon,
                  ', steps = ', episode_step,
                  ", total reward = ", episode_reward,
                  ", episode time = ", "{0:.2f}".format(time.time() - episode_start_time))

            # save episode data to tensorboard summary
            if config['save_tensorboard_summary']:
                save_episode_to_summary(summary_writer, episode, step, time.time() - episode_start_time,
                                        episode_reward, learner.epsilon)

            # ============================================================================================================== #

            # update data for best episode
            if episode_reward > max_reward:
                max_reward = episode_reward
                max_episode_number = episode
                max_episode_frames = episode_frames

            # update data for summary-plot
            summary_steps.append(episode_step)
            summary_epsilons.append(learner.epsilon)
            summary_rewards.append(episode_reward)

            # reset episode data
            episode += 1
            state = env.reset()
            episode_frames = [state]
            state = processor.process_state_for_memory(state, True)
            episode_start_time = time.time()
            episode_reward = 0
            episode_step = 0

    # ============================================================================================================== #

        # test current model
        if (step % test_frequency == 0 and step != 0) or step == config['steps'] - 1:
            test(learner,
                 copy.deepcopy(env),
                 config,
                 copy.deepcopy(processor),
                 'test_' + config['algorithm'] + '_' + config['environment'] + '_' + str(datetime.datetime.now()),
                 save_path + 'graphs/tests/')

    # ============================================================================================================== #

        # update summary-plot
        if summary is not None and ((step % summary_save_step == 0 and step != 0) or step == config['steps'] - 1):
            summary.summarize(step_counts=summary_steps,
                              reward_counts=summary_rewards,
                              epsilon_values=summary_epsilons,
                              e_greedy_formula=learner.e_greedy_formula)
            summary_steps, summary_epsilons, summary_rewards = [], [], []

    # ============================================================================================================== #

        # make gif from episode frames
        # no image data available for cartpole
        if config['save_gif'] \
                and config['environment'] != 'CartPole-v1' \
                and ((step != 0 and step % config['gif_save_frequency'] == 0) or step == config['steps'] - 1):
            make_gif(max_episode_number, max_reward, save_path + 'gifs/', max_episode_frames)
            max_reward = -1
            max_episode_number = -1
            max_episode_frames = []

    # ============================================================================================================== #

        if config['save_model'] and \
                ((step != 0 and step % config['model_save_frequency'] == 0) or step == config['steps'] - 1):
            # store model weights and parameters when episode rewards are above a certain amount
            # and after every number of episodes
            learner.save_network(save_path + 'models/', config['environment'] + '_' + str(episode))

    # ============================================================================================================== #

    # killing environment to prevent memory leaks
    env.close()
