import numpy as np
from simple_dqn import Agent
from collections import deque
import gym
from utils import preprocess_frame

if __name__ == '__main__':
    env = gym.make('Pong-v0')

    num_games = 500
    load_checkpoint = False
    best_score = -21
    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.0001,
                  input_dims=(104,104,12), n_actions=6, mem_size=25000,
                  eps_min=0.02, batch_size=32, replace=1000, eps_dec=1e-5)

    if load_checkpoint:
        agent.load_models()

    scores, eps_history = [], []
    n_steps = 0

    for i in range(num_games):
        done = False
        observation = env.reset()
        frame_queue = deque(maxlen=4)

        observation = preprocess_frame(observation)
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
            frame_queue.appendleft(preprocess_frame(next_frame))

            observation_ = np.concatenate(frame_queue, axis=2)

            if not load_checkpoint:
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