import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from world_model.simple_vae import CVAE

def load_world_model(env_name, n_actions):
    # weights = 'world_model/cvae_weights_mult_%s.h5' % env_name
    weights_path = '/home/agents/model_based/simple_vae/agent/next_agent/../../world_model/cvae_weights_mult_%s.h5' %env_name
    model = CVAE(n_actions)
    model.set_weights(weights_path)

    return model
