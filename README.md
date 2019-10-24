# Gym-T4-Testbed - Part 4 Project 22: Hallucinogenic Learning
In this research project, we attempt to develop an agent that can act based on its predicted future states.
- Menake Ratnayake and Raymond Wang.

## Prerequisites
All scripts were developed and ran using Python 3
### Docker with Nvidia
Our project was developed and executed in a docker container. This ensured a consistent runtime environment
across all our devices that we were developing on. We also utilise the Nvidia docker environment to enable faster
training of our neural networks

#### Getting Started
1. Install the latest version of docker engine for your OS by heading to https://docs.docker.com/install/ for reference.
2. Install the latest Nvidia drivers for your graphics card to enable CUDA. (CUDA does not need to be installed and will be handled by the docker image)
3. Follow the instructions on https://github.com/NVIDIA/nvidia-docker and install nvidia-docker.

After the completion of these steps the docker shell script is now fully functional.

#### Executing the shell

To open the docker shell or command prompt within the docker environment, run the following command in the root directory

```
    ./dockerShell.sh
```

This should execute a containerised terminal 

The execution of all scripts will be done within this shell.

### Python Prerequisites
If a dockerShell is not desired, the python dependencies in `requirements.txt` should be installed to execute
the scripts in a local environment. Specifically, these are:

----------------
- numpy
- keras
- scikit-image
- opencv-python
- matplotlib
- imageio
- gym[atari]
- scipy

----------------

### Example environments
----------------
- PongDeterministic-v4
- BreakoutDeterministic-v4
- More can be found at https://gym.openai.com/envs/#atari
----------------

### Help running scripts
For most scripts, the usage instructions are shown by
```
python [scriptname.py] --help
```

### Predictive Model
To run the predictive model, first run `cd agents/model_based/predictive_model` from the root directory.

#### Generating Rollout Data
Execute `python generate_data.py`. The full usage details are detailed below.

```
usage: generate_data.py [-h] [--env_name ENV_NAME] [--method METHOD]
                        [--total_episodes TOTAL_EPISODES]
                        [--time_steps TIME_STEPS]

Create new training data

optional arguments:
  -h, --help            show this help message and exit
  --env_name ENV_NAME   name of Atari environment
  --method METHOD
  --total_episodes TOTAL_EPISODES
                        total number of episodes to generate per worker
  --time_steps TIME_STEPS
                        how many timesteps at start of episode?
```

This script currently uses a random agent without the informed flag. If the DQN model is saved then it can be loaded in by using the --informed flag.

#### Training the model
After data has been generated for a certain environment, the model can be trained for that environment by running `python train_predictive_model.py`. Ensure
that the environment name is the same as the environment that data was generated for. If it is a new model, use the `--new_model` flag.

```
usage: train_predictive_model.py [-h] [--N N] [--new_model] [--epochs EPOCHS]
                     [--env_name ENV_NAME] [--method METHOD]

Train predictive model

optional arguments:
  -h, --help           show this help message and exit
  --N N                number of episodes to use to train
  --new_model          start a new model from scratch?
  --epochs EPOCHS      number of epochs to train for
  --env_name ENV_NAME  name of environment
  --informed           if true, will attempt to train on informed rollouts instead of random rollouts
```

#### Testing the model
Once training has been completed, the model can be tested against a rollout that was generated previously. This can be done by executing `pythn test_predictive_model.py` script as follows.

```
usage: test_predictive_model.py [-h] [--env_name ENV_NAME] [--informed]

Test predictive model

optional arguments:
  -h, --help           show this help message and exit
  --env_name ENV_NAME  name of environment
  --informed           if true, will attempt to test on informed rollouts instead of random rollouts
```

Currently, the rollout is is tested against is hardcoded as rollout number 100. This script will also generate a gif in the gifs folder 
comparing the original, next frame, predicted frame and loss between the next and predicted frame.

### DQN Agent
To run scripts for the DQN agent, first run `cd agents/model_based/dqn_agent` from the root directory.

#### Training the agent
Unlike the predictive model, the DQN agent does not need rollout data. To train the DQN agent run `python train_dqn.py`.
The parameters for the script is shown below. This will generate a set of weights from DQN. One evaluation network and one target network is trained. Currently the script trains for 1000 episodes of the game, this can be modified within the script.

```
usage: train_dqn.py [-h] [--env_name ENV_NAME] [--new_model] [--num_games NUM_GAMES]

Train DQN

optional arguments:
  -h, --help           show this help message and exit
  --env_name ENV_NAME  name of environment
  --new_model          if selected, trains a new DQN model
  --num_games          number of games to train on
```

#### Testing the agent
Once a network has been trained, it can be tested running `python test_dqn.py`. Only the environment name needs to be parsed to this script.
```
usage: test_dqn.py [-h] [--env_name ENV_NAME]

Test DQN

optional arguments:
  -h, --help           show this help message and exit
  --env_name ENV_NAME  name of environment
```

This tests the agent in the environment for 5 games. The number of games can be modified within the script.

### Next State Agent
To run scripts for the DQN agent, first run `cd agents/model_based/next_agent` from the root directory.

#### Generating Rollout Data
Prior to generating the rollout data for this agent, ensure that a predictive model and a DQN agent are already trained for this environment.
Run `python generate_agent_data.py` to generate rollout data. The full usage details are detailed below.

```
usage: generate_agent_data.py [-h] [--env_name ENV_NAME]
                              [--total_episodes TOTAL_EPISODES]
                              [--time_steps TIME_STEPS]

Create new training data

optional arguments:
  -h, --help                        show this help message and exit
  --env_name ENV_NAME               name of environment
  --total_episodes TOTAL_EPISODES   total number of episodes to generate
  --time_steps TIME_STEPS           number of timesteps per episode
```

This will fit the predictions using the predictive model to the agents chosen by the DQN agent in the environment.

#### Training the model
Once rollout data is generated, the model can be trained for that environment by running `python train_agent.py`. 

```
usage: train_agent.py [-h] [--N N] [--env_name ENV_NAME]
                      [--time_steps TIME_STEPS] [--epochs EPOCHS]

Train next state agent

optional arguments:
  -h, --help                show this help message and exit
  --N N                     number of episodes to use to train
  --env_name ENV_NAME       name of environment
  --time_steps TIME_STEPS   time steps in each rollout
  --epochs EPOCHS           epochs to train for
```

#### Testing the model
Once training has been completed, the agent can be tested against the environment it was trained on. This can be done by executing the `python test_agent.py` script as follows.

```
usage: test_agent.py [-h] [--env_name ENV_NAME] [--runs RUNS]
                     [--resolution RESOLUTION]

Test and plot graphs for next state agent

optional arguments:
  -h, --help                show this help message and exit
  --env_name ENV_NAME       name of environment
  --runs RUNS               number of game runs to test
  --resolution RESOLUTION   step resolution for performance graph e.g. 100 = time for 100 steps
```

The test script runs the agent against the number of runs that was set and plots the average performance across these games. It also does the same for random and DQN agents so that a comparison can be observed in the graphs. The efficiency of the agent is also timed and graphed.

## Files Overview

``` bash
    # Source folder
    .dockerignore       # ignore file for docker
    requirements.txt    # python dependencies for docker
    dockerfile          # docker execution workflow
    dockerShell.sh      # main script to open up docker container
    /agents
        /model_based
            __init__.py        # module init file
            utils.py           # contains various utility functions used by predictive model and agents        
        # Predictive model
            /predictive_model
                /data                       # folder where rollout data is stored
                /gifs                       # folder where gifs generated by test_predictive_model.py gets stored
                /images                     # folder where generated images from model predictions are stored
                /models                     # folder where model weights get stored    
                /Network_variations         # contains different configurations that were attempted 
                generate_data.py            # generates predictive model rollout data
                generate_gif.py             # script with helper function to generate gifs
                load_predictive_model.py    # module for loading the model in other scripts
                predictive_model.py         # predictive model network architecture and network functions such as predict, save, load
                test_predictive_model.py    # tests the predictive model against a rollout
                train_predictive_model.py   # trains the predictive model
        # DQN agent
            /dqn_agent
                /models                 # folder for DQN weights
                __init__.py             # module init file
                load_dqn.py             # contains functions for loading the dqn model in other scripts
                simple_dqn.py           # the DQN architecture and neural network with network functions such as choose action, save, load
                test_dqn.py             # tests the DQN agent against an environment
                train_dqn.py            # trains the DQN agent on an environment
        # Next State agent
            /next_agent
                /data                   # folder where rollout data is stored
                /graphs                 # folder for storing graphs generated from testing, contains graphs
                /models                 # folder for next agent weights
                __init__.py             # module init file
                generate_agent_data.py  # generates rollout data for Next State agent
                state_agent.py          # the Next State agent architecture with network functions such as choose action, save, load
                test_agent.py           # tests the Next State agent against an environment and plots graphs
                train_agent.py          # trains the Next State agent on an environment
``` 

