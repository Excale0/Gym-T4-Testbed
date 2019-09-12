from simple_vae import CVAE
import os

def load_world_model(env_name, n_actions):
    weights = './cvae_weights_mult_%s.h5' % env_name
    weights_path = os.path.abspath(weights)

    model = CVAE(n_actions)
    model.set_weights(weights_path)

    return model
