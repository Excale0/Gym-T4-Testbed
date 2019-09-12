#!/bin/bash
echo $1
python ./train_cvae.py --env_name $1 --N 128 --epochs 256 --new_model