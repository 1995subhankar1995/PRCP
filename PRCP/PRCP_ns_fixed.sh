#!/bin/bash


python -W ignore ./RSCP/up1_PRLCP.py -a 0.1 -d 0.125 -e 5 -r 0.0 --batch_size 8192 --dataset CIFAR10 --arc ResNet -do 0.125 --My_model --device 0
python -W ignore ./RSCP/up1_PRLCP.py -a 0.1 -d 0.125 -e 5 -r 0.0 --batch_size 1024 --dataset CIFAR10 --arc DenseNet -do 0.125 --My_model --device 0
python -W ignore ./RSCP/up1_PRLCP.py -a 0.1 -d 0.125 -e 5 -r 0.0 --batch_size 8192 --dataset CIFAR10 --arc VGG -do 0.125 --My_model --device 0


python -W ignore ./RSCP/up1_PRLCP.py -a 0.1 -d 0.125 -e 5 -r 0.0 --batch_size 8192 --dataset CIFAR100 --arc ResNet -do 0.125 --My_model --device 0
python -W ignore ./RSCP/up1_PRLCP.py -a 0.1 -d 0.125 -e 5 -r 0.0 --batch_size 1024 --dataset CIFAR100 --arc DenseNet -do 0.125 --My_model --device 0
python -W ignore ./RSCP/up1_PRLCP.py -a 0.1 -d 0.125 -e 5 -r 0.0 --batch_size 8192 --dataset CIFAR100 --arc VGG -do 0.125 --My_model --device 0

