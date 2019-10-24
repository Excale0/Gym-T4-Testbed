import os
import sys

from predictive_model.predictive_model import AutoEncoder

def load_predictive_model(env_name, n_actions):
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(parent_dir, 'models/predictive_model_weights_%s.h5' % env_name)
    model = AutoEncoder(n_actions)
    model.set_weights(weights_path)

    return model
