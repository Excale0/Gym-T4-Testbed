from state_agent import StateAgent
import argparse
import numpy as np
import os
import gym

folder_path = os.path.dirname(os.path.abspath(__file__))
ROLLOUT_DIR = os.path.join(folder_path, "data")

def action_space_dimension(env_name):
    return gym.make(env_name).action_space.n

def import_data(episodes, action_dim, dir_name):
    time_steps = 400
    x = 80
    filelist = os.listdir(dir_name)
    length_filelist = len(filelist)

    if length_filelist > episodes:
        filelist = filelist[:episodes]

    if length_filelist < episodes:
        episodes = length_filelist

    next_states = np.zeros((time_steps*episodes, 104, 80, action_dim), dtype=np.float32)
    correct_state = np.zeros((time_steps*episodes, action_dim), dtype=np.float32)

    idx = 0
    file_count = 0

    for file in filelist:
        try:
            action_data = np.load(dir_name + file)['correct']
            next_data = np.load(dir_name + file)['next']

            next_states[idx:(idx + time_steps), :, :, :] = next_data
            correct_state[idx:(idx + time_steps), :] = action_data

            idx = idx + time_steps
            file_count += 1

            if file_count%50==0:
                print('Imported {} / {} ::: Current data size = {} observations'.format(file_count, episodes, idx))
            elif file_count == episodes:
                break

        except:
            print('Skipped {}...'.format(file))

    print('Imported {} / {} ::: Current data size = {} observations'.format(file_count, episodes, idx))

    return [next_states,correct_state]


def main(args):
    dir_name = ROLLOUT_DIR + "/rollout_informed_" + args.env_name + "/"
    # new_model = args.new_model
    episodes = int(args.N)
    action_dim = action_space_dimension(args.env_name)

    [next_states, correct_state] = import_data(episodes,action_dim,dir_name)
    try:
        agent = StateAgent(action_dim,args.env_name)
    except:
        print('NO DATA FOUND')
        raise
    
    agent.train(next_states,correct_state,32)
    agent.save_weights()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Train agent'))
  parser.add_argument('--N',default = 40, help='number of episodes to use to train')
#   parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
#   parser.add_argument('--epochs', default = 10, help='number of epochs to train for')
  parser.add_argument('--env_name', type=str, help='name of environment', default="PongDeterministic-v4")
  args = parser.parse_args()
  
  main(args)


