from vae import VAE
import argparse
import numpy as np
import os

DIR_NAME = './data/rollout/'
M=300

SCREEN_SIZE_X = 105
SCREEN_SIZE_Y = 80


def import_data(N):
  filelist = os.listdir(DIR_NAME)
  filelist.sort()
  length_filelist = len(filelist)

  if length_filelist > N:
    filelist = filelist[:N]

  if length_filelist < N:
    N = length_filelist

  data = np.zeros((M*N, SCREEN_SIZE_X, SCREEN_SIZE_Y, 3), dtype=np.float32)

  idx = 0
  file_count = 0

  for file in filelist:
      try:
        new_data = np.load(DIR_NAME + file)['obs']
        data[idx:(idx + M), :, :, :] = new_data

        idx = idx + M
        file_count += 1

        if file_count%50==0:
          print('Imported {} / {} ::: Current data size = {} observations'.format(file_count, N, idx))
      except:
        print('Skipped {}...'.format(file))

  print('Imported {} / {} ::: Current data size = {} observations'.format(file_count, N, idx))

  return data, N

def main(args):

    new_model = args.new_model
    N = int(args.N)
    epochs = int(args.epochs)

    vae = VAE()

    if not new_model:
        try:
            vae.set_weights('./vae_weights.h5')
        except:
            print("Either set --new_model or ensure ./vae/weights.h5 exists")
            raise

    try:
        data, N = import_data(N)
    except:
        print('NO DATA FOUND')
        raise
        
    print('DATA SHAPE = {}'.format(data.shape))

    for epoch in range(epochs):
        print('EPOCH ' + str(epoch))
        vae.train(data)
        vae.save_weights('./vae_weights.h5')
        

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Train VAE'))
  parser.add_argument('--N',default = 10000, help='number of episodes to use to train')
  parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
  parser.add_argument('--epochs', default = 10, help='number of epochs to train for')
  args = parser.parse_args()

  main(args)