#!/bin/bash

python -W ignore ./RSCP/ARLCP_s_tune.py -a 0.1 -d 0.125 -e 50 -r 1.0 --n_s 32 --batch_size 4096 --dataset CIFAR10 --arc ResNet --My_model --device 0


python -W ignore ./RSCP/ARLCP_s_tune.py -a 0.1 -d 0.125 -e 50 -r 1.0 --n_s 32 --batch_size 4096 --dataset CIFAR100 --arc ResNet --My_model --device 0

