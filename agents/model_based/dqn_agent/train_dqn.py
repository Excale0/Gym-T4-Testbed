import numpy as np
from simple_dqn import Agent
from collections import deque
import gym
import argparse
from utils import preprocess_frame_dqn

def main(args):
    env_name = args.env_name
    new_model = args.new_model
    num_games = args.num_games

    env = gym.make(env_name)
    env.seed(0)

    # terminating_steps = 250000

    # set this to be a very low number so it works for a variety of games
    # should be set to minimum score of game
    best_score = -9999.0    
    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.0001,
                  input_dims=(104,80,4), n_actions=env.action_space.n, mem_size=25000,
                  eps_min=0.02, batch_size=32, replace=1000, eps_dec=1e-5, env_name=env_name)

    if not new_model:
        try:
            agent.load_models()
        except:
            print('No DQN models found for %s in models folder' % env_name)
            raise

    scores, eps_history = [], []
    n_steps = 0

    for i in range(num_games):
        done = False
        observation = env.reset()
        frame_queue = deque(maxlen=4)

        observation = preprocess_frame_dqn(observation)
        for j in range(4):
            frame_queue.append(observation)
        observation = np.concatenate(frame_queue, axis=2)

        score = 0

        while not done:
            action = agent.choose_action(observation)
            next_frame, reward, done, info = env.step(action)

            n_steps += 1
            score += reward
            
            frame_queue.pop()
            frame_queue.appendleft(preprocess_frame_dqn(next_frame))

            observation_ = np.concatenate(frame_queue, axis=2)

            agent.store_transition(observation, action,
                                     reward, observation_, int(done))
            agent.learn()

            observation = observation_

        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i,'score: ', score, ' average score %.3f' % avg_score, 'epsilon %.2f' % agent.epsilon, 'steps', n_steps)
        
        if avg_score > best_score:
            agent.save_models()
            print('avg score %.2f better than best score %.2f, saving model' % (avg_score, best_score))
            best_score = avg_score

        eps_history.append(agent.epsilon)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Train DQN'))
    parser.add_argument('--env_name', type=str, help='name of environment', default="PongDeterministic-v4")
    parser.add_argument('--new_model', action='store_true', help='if selected, trains a new DQN model')
    parser.add_argument('--num_games', type=int, help='number of games to train on', default=1000)
    args = parser.parse_args()
    main(args)