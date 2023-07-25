#!/bin/bash


python -W ignore ./RSCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 1.0 -e 5 -r 0.0 --batch_size 8192 --dataset CIFAR10 --arc ResNet --My_model --device 0 -s 0.0

python -W ignore ./RSCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 2.0 -e 5 -r 0.0 --batch_size 8192 --dataset CIFAR10 --arc ResNet --My_model --device 0 -s 0.0

python -W ignore ./RSCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 3.0 -e 5 -r 0.0 --batch_size 8192 --dataset CIFAR10 --arc ResNet --My_model --device 0 -s 0.0



python -W ignore ./RSCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 1.0 -e 5 -r 0.0 --batch_size 8192 --dataset CIFAR10 --arc VGG --My_model --device 0 -s 0.0

python -W ignore ./RSCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 2.0 -e 5 -r 0.0 --batch_size 8192 --dataset CIFAR10 --arc VGG --My_model --device 0 -s 0.0

python -W ignore ./RSCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 3.0 -e 5 -r 0.0 --batch_size 8192 --dataset CIFAR10 --arc VGG --My_model --device 0 -s 0.0



python -W ignore ./RSCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 1.0 -e 5 -r 0.0 --batch_size 1024 --dataset CIFAR10 --arc DenseNet --My_model --device 0 -s 0.0

python -W ignore ./RSCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 2.0 -e 5 -r 0.0 --batch_size 1024 --dataset CIFAR10 --arc DenseNet --My_model --device 0 -s 0.0

python -W ignore ./RSCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 3.0 -e 5 -r 0.0 --batch_size 1024 --dataset CIFAR10 --arc DenseNet --My_model --device 0 -s 0.0





python -W ignore ./RSCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 1.0 -e 5 -r 0.0 --batch_size 8192 --dataset CIFAR100 --arc ResNet --My_model --device 0 -s 0.0

python -W ignore ./RSCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 2.0 -e 5 -r 0.0 --batch_size 8192 --dataset CIFAR100 --arc ResNet --My_model --device 0 -s 0.0

python -W ignore ./RSCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 3.0 -e 5 -r 0.0 --batch_size 8192 --dataset CIFAR100 --arc ResNet --My_model --device 0 -s 0.0



python -W ignore ./RSCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 1.0 -e 5 -r 0.0 --batch_size 8192 --dataset CIFAR100 --arc VGG --My_model --device 0 -s 0.0

python -W ignore ./RSCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 2.0 -e 5 -r 0.0 --batch_size 8192 --dataset CIFAR100 --arc VGG --My_model --device 0 -s 0.0

python -W ignore ./RSCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 3.0 -e 5 -r 0.0 --batch_size 8192 --dataset CIFAR100 --arc VGG --My_model --device 0 -s 0.0



python -W ignore ./RSCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 1.0 -e 5 -r 0.0 --batch_size 1024 --dataset CIFAR100 --arc DenseNet --My_model --device 0 -s 0.0

python -W ignore ./RSCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 2.0 -e 5 -r 0.0 --batch_size 1024 --dataset CIFAR100 --arc DenseNet --My_model --device 0 -s 0.0

python -W ignore ./RSCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 3.0 -e 5 -r 0.0 --batch_size 1024 --dataset CIFAR100 --arc DenseNet --My_model --device 0 -s 0.0

