#!/bin/bash 

ENV='Pong-v0'

python generate_data.py --env_name $ENV && python train_cvae.py --env_name $ENV --N 128 --epochs 256 --new_model