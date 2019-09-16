#!/bin/bash 

ENV='BreakoutDeterministic-v4 SpaceInvadersDeterministic-v4 MsPacmanDeterministic-v4'

for i in $ENV; do
    # if [ "$i" != "PongDeterministic-v4" ]; then
    #     python generate_data.py --env_name $i --total_episodes 128 
    # fi
    # python train_cvae.py --env_name $i --N 100 --epochs 128 --new_model
    python test_cvae.py $i
done

