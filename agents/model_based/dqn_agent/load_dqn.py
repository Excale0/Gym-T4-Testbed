import os
import sys
from dqn_agent.simple_dqn import Agent

def load_dqn(env):
    env_name = env.unwrapped.spec.id
    
    agent = Agent(gamma=0.99, epsilon=0.0, alpha=0.0001,
                  input_dims=(104,80,4), n_actions=env.action_space.n, mem_size=25000,
                  eps_min=0.02, batch_size=32, replace=1000, eps_dec=1e-5, env_name=env_name)
    
    agent.load_models()
    return agent
