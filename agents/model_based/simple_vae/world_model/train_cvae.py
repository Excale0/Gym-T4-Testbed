from simple_vae import CVAE
from random import shuffle
import argparse
import numpy as np
import os
import gym

DIR_NAME = './data/world_models/'
M=100

SCREEN_SIZE_X = 104
SCREEN_SIZE_Y = 104

def action_space_dimension(env_name):
  return gym.make(env_name).action_space.n


def import_data(N, action_dim, dir_name):
  filelist = os.listdir(dir_name)
  shuffle(filelist)
  length_filelist = len(filelist)

  if length_filelist > N:
    filelist = filelist[:N]

  if length_filelist < N:
    N = length_filelist

  observation = np.zeros((M*N, SCREEN_SIZE_X, SCREEN_SIZE_Y, 12), dtype=np.float32)
  action = np.zeros((M*N, action_dim), dtype=np.float32)
  next_frame = np.zeros((M*N, SCREEN_SIZE_X, SCREEN_SIZE_Y, 3), dtype=np.float32)

  idx = 0
  file_count = 0

  for file in filelist:
    try:
        obs_data = np.load(dir_name + file)['obs']
        action_data = np.load(dir_name + file)['actions']
        next_data = np.load(dir_name + file)['next_frame']


        observation[idx:(idx + M), :, :, :] = obs_data
        action[idx:(idx + M), :] = action_data
        next_frame[idx:(idx + M), :, :, :] = next_data

        idx = idx + M
        file_count += 1

        if file_count%50==0:
            print('Imported {} / {} ::: Current data size = {} observations'.format(file_count, N, idx))
        elif file_count == N:
            break

    except:
        print('Skipped {}...'.format(file))

  print('Imported {} / {} ::: Current data size = {} observations'.format(file_count, N, idx))

  return [observation,action,next_frame], N

def main(args):

    new_model = args.new_model
    N = int(args.N)
    epochs = int(args.epochs)
    action_dim = action_space_dimension(args.env_name)
    cvae = CVAE(action_dim)
    # return
    dir_name = DIR_NAME + 'rollout_' + args.env_name + '/'
    weights_name = './cvae_weights_mult_' + args.env_name + '.h5'

    if not new_model:
        try:
            cvae.set_weights(weights_name)
        except:
            print("Either set --new_model or ensure %s/weights.h5 exists" % model_dir)
            raise

    try:
        data, N = import_data(N,action_dim,dir_name)
    except:
        print('NO DATA FOUND')
        raise
        
    print('DATA SHAPE = {}'.format(data[0].shape))

    for epoch in range(epochs):
        print('EPOCH ' + str(epoch))
        # cvae.train(data[0],data[1],data[2])
        cvae.train(data[0],data[1],data[2])
        cvae.save_weights(weights_name)
        

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Train VAE'))
  parser.add_argument('--N',default = 130, help='number of episodes to use to train')
  parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
  parser.add_argument('--epochs', default = 10, help='number of epochs to train for')
  parser.add_argument('--env_name', type=str, help='name of environment', default="Breakout-v0")
  args = parser.parse_args()
  
  main(args)