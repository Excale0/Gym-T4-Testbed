#!/bin/bash

python ./generate_agent_data.py --informed --env_name BreakoutDeterministic-v4
python ./train_agent.py --env_name BreakoutDeterministic-v4
python ./test_agent.py >> Breakout-unfinished-network.txt