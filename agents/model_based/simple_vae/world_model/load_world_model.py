import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from world_model.simple_vae import CVAE

def load_world_model(env_name, n_actions):
    # weights = 'world_model/cvae_weights_mult_%s.h5' % env_name
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(parent_dir, 'models/trained_rollout/cvae_weights_mult_%s.h5' %env_name)
    model = CVAE(n_actions)
    model.set_weights(weights_path)

    return model
