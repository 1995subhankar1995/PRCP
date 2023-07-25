#!/bin/bash

python -W ignore ./RSCP/NattackARLCP1.py -a 0.1 -d 0.125 -e 50 -r 0.0 --n_s 32 --batch_size 4096 --dataset CIFAR10 --arc ResNet --My_model --device 0 -do 12
python -W ignore ./RSCP/NattackARLCP1.py -a 0.1 -d 0.125 -e 50 -r 0.5 --n_s 32 --batch_size 4096 --dataset CIFAR10 --arc ResNet --My_model --device 0 -do 6
python -W ignore ./RSCP/NattackARLCP1.py -a 0.1 -d 0.125 -e 50 -r 1.0 --n_s 32 --batch_size 4096 --dataset CIFAR10 --arc ResNet --My_model --device 0 -do 6
python -W ignore ./RSCP/NattackARLCP1.py -a 0.1 -d 0.125 -e 50 -r 2.0 --n_s 32 --batch_size 4096 --dataset CIFAR10 --arc ResNet --My_model --device 0 -do 6


python -W ignore ./RSCP/NattackARLCP1.py -a 0.1 -d 0.125 -e 50 -r 0.0 --n_s 32 --batch_size 1024 --dataset CIFAR10 --arc DenseNet --My_model --device 0 -do 12
python -W ignore ./RSCP/NattackARLCP1.py -a 0.1 -d 0.125 -e 50 -r 2.0 --n_s 32 --batch_size 1024 --dataset CIFAR10 --arc DenseNet --My_model --device 0 -do 6
python -W ignore ./RSCP/NattackARLCP1.py -a 0.1 -d 0.125 -e 50 -r 0.0 --n_s 32 --batch_size 4096 --dataset CIFAR10 --arc VGG --My_model --device 0 -do 12
python -W ignore ./RSCP/NattackARLCP1.py -a 0.1 -d 0.125 -e 50 -r 2.0 --n_s 32 --batch_size 4096 --dataset CIFAR10 --arc VGG --My_model --device 0 -do 6


python -W ignore ./RSCP/NattackARLCP1.py -a 0.1 -d 0.125 -e 50 -r 3.0 --n_s 32 --batch_size 4096 --dataset CIFAR10 --arc ResNet --My_model --device 0 -do 6
python -W ignore ./RSCP/NattackARLCP1.py -a 0.1 -d 0.125 -e 50 -r 4.0 --n_s 32 --batch_size 4096 --dataset CIFAR10 --arc ResNet --My_model --device 0 -do 6
python -W ignore ./RSCP/NattackARLCP1.py -a 0.1 -d 0.125 -e 50 -r 6.0 --n_s 32 --batch_size 4096 --dataset CIFAR10 --arc ResNet --My_model --device 0 -do 6
python -W ignore ./RSCP/NattackARLCP1.py -a 0.1 -d 0.125 -e 50 -r 8.0 --n_s 32 --batch_size 4096 --dataset CIFAR10 --arc ResNet --My_model --device 0 -do 6



python -W ignore ./RSCP/NattackARLCP1.py -a 0.1 -d 0.125 -e 50 -r 0.0 --n_s 32 --batch_size 4096 --dataset CIFAR100 --arc ResNet --My_model --device 0 -do 12
python -W ignore ./RSCP/NattackARLCP1.py -a 0.1 -d 0.125 -e 50 -r 0.5 --n_s 32 --batch_size 4096 --dataset CIFAR100 --arc ResNet --My_model --device 0 -do 6
python -W ignore ./RSCP/NattackARLCP1.py -a 0.1 -d 0.125 -e 50 -r 1.0 --n_s 32 --batch_size 4096 --dataset CIFAR100 --arc ResNet --My_model --device 0 -do 6
python -W ignore ./RSCP/NattackARLCP1.py -a 0.1 -d 0.125 -e 50 -r 2.0 --n_s 32 --batch_size 4096 --dataset CIFAR100 --arc ResNet --My_model --device 0 -do 6


python -W ignore ./RSCP/NattackARLCP1.py -a 0.1 -d 0.125 -e 50 -r 0.0 --n_s 32 --batch_size 1024 --dataset CIFAR100 --arc DenseNet --My_model --device 0 -do 12
python -W ignore ./RSCP/NattackARLCP1.py -a 0.1 -d 0.125 -e 50 -r 2.0 --n_s 32 --batch_size 1024 --dataset CIFAR100 --arc DenseNet --My_model --device 0 -do 6

python -W ignore ./RSCP/NattackARLCP1.py -a 0.1 -d 0.125 -e 50 -r 0.0 --n_s 32 --batch_size 4096 --dataset CIFAR100 --arc VGG --My_model --device 0 -do 12
python -W ignore ./RSCP/NattackARLCP1.py -a 0.1 -d 0.125 -e 50 -r 2.0 --n_s 32 --batch_size 4096 --dataset CIFAR100 --arc VGG --My_model --device 0 -do 6


python -W ignore ./RSCP/NattackARLCP1.py -a 0.1 -d 0.125 -e 50 -r 3.0 --n_s 32 --batch_size 4096 --dataset CIFAR100 --arc ResNet --My_model --device 0 -do 6
python -W ignore ./RSCP/NattackARLCP1.py -a 0.1 -d 0.125 -e 50 -r 4.0 --n_s 32 --batch_size 4096 --dataset CIFAR100 --arc ResNet --My_model --device 0 -do 6
python -W ignore ./RSCP/NattackARLCP1.py -a 0.1 -d 0.125 -e 50 -r 6.0 --n_s 32 --batch_size 4096 --dataset CIFAR100 --arc ResNet --My_model --device 0 -do 6
python -W ignore ./RSCP/NattackARLCP1.py -a 0.1 -d 0.125 -e 50 -r 8.0 --n_s 32 --batch_size 4096 --dataset CIFAR100 --arc ResNet --My_model --device 0 -do 6


