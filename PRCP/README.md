# PRCP (Probabilistically Robust Conformal Prediction)
This repository contains the code and models necessary to replicate the results of our paper:

## Contents
The major content of our repo are:
 - `PRCP/` The main folder containing the python scripts for running the experiments.
 - `third_party/` Third-party python scripts imported. Specifically we make use of the SMOOTHADV attack by [Salman et al (2019)](https://github.com/Hadisalman/smoothing-adversarial)
 - `Arcitectures/` Architectures for our trained models.
 - `Pretrained models/` Cohen pretrained models. [Cohen et al (2019)](https://github.com/locuslab/smoothing)
 - `checkpoints/` Our pre trained models.
 - `datasets/` A folder that contains the datasets used in our experiments CIFAR10, CIFAR100, Imagenet.
 - `Results/` A folder that contains different csv files from different experiments, used to generate the results in the paper.

PRCP folder contains:

1. `main_ARCP_Cifar.py`: the main code for running experiments for the CIFAR data for aPRCP(worst-adv).
1. `main_ARCP_ImageNet.py`: the main code for running experiments for the ImageNet data for aPRCP(worst-adv).
1. `main_PRCP.py`: the main code for running experiments for aPRCP(\tilde \alpha).
1. `NattackARLCP.py`: the main code for running experiments for this paper 'https://proceedings.mlr.press/v97/li19g.html'.
2. `Score_Functions.py`: containing all non-conformity scores used.
3. `utills.py`: calibration and predictions functions, as well as other function used in the main code.
3. `my_utils.py`: helping codes.

## Prerequisites

Prerequisites for running our code:
 - numpy
 - scipy
 - sklearn
 - torch
 - tqdm
 - seaborn
 - torchvision
 - pandas
 - plotnine
 
## Running instructions
1.  Install dependencies:
```
conda create -n ARCP python=3.8
conda activate ARCP
conda install -c conda-forge numpy
conda install -c conda-forge scipy
conda install -c conda-forge scikit-learn
conda install -c conda-forge tqdm
conda install -c conda-forge seaborn
conda install -c conda-forge pandas
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c conda-forge plotnine
```
2. 
   1. Download trained models from [here](https://drive.google.com/file/d/1NY25J5lVGyR583J4iUFKrZP3OpfcjDmw/view?usp=sharing) and extract them to PRCP/checkpoints/.
   2. Download cohen models from [here](https://drive.google.com/file/d/1h_TpbXm5haY5f-l4--IKylmdz6tvPoR4/view) and extract them to PRCP/Pretrained_Models/. Change the name of "models" folder to "Cohen".
   3. If you want to run ImageNet experiments, obtain a copy of ImageNet ILSVRC2012 validation set and preprocess the val directory by running [this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh). Put the created folders in PRCP/datasets/imagenet/.
3. The current working directory when running the scripts should be the top folder PRCP.


To reproduce the results needed to create Figure 3 of the main paper for example run:
```
python -W ignore ./PRCP/main_ARCP_Cifar.py -a 0.1 -d 0.125 -e 50 -r 0.5 -do 0.125 --batch_size 4096 --dataset CIFAR10 --arc ResNet --My_model --ms_list [128]
python -W ignore ./PRCP/main_ARCP_Cifar.py -a 0.1 -d 0.125 -e 50 -r 0.5 -do 0.125 --batch_size 4096 --dataset CIFAR100 --arc ResNet --My_model --ms_list [128]
python -W ignore ./PRCP/main_ARCP_ImageNet.py -a 0.1 -d 0.25 -e 50 -r 1.0 -do 0.25 --batch_size 1024 --dataset ImageNet --arc ResNet --ms_list [128]
```

To reproduce the results needed to create Figures 5 and 6 respectively, please run:
```
python -W ignore ./PRCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 0.125 -e 50 -r 0.0 --batch_size 8192 --dataset CIFAR10 --arc ResNet -do 0.125 --My_model --pr_list [0.1, 0.09, 0.06, 0.03, 0.00] --CalUniform 1 --TestUniform 1 -s 0.0

python -W ignore ./PRCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 0.125 -e 50 -r 0.0 --batch_size 8192 --dataset CIFAR100 --arc ResNet -do 0.125 --My_model --pr_list [0.1, 0.09, 0.06, 0.03, 0.00] --CalUniform 1 --TestUniform 1 -s 0.0

```


To reproduce the results needed to create Figures 7 and 8 respectively, please run:
```
python -W ignore ./PRCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 0.125 -e 50 -r 0.0 --batch_size 8192 --dataset CIFAR10 --arc ResNet -do 0.125 --My_model --pr_list [0.1, 0.09, 0.06, 0.03, 0.00] --CalUniform 0 --TestUniform 0 -s 0.0

python -W ignore ./PRCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 0.125 -e 50 -r 0.0 --batch_size 8192 --dataset CIFAR100 --arc ResNet -do 0.125 --My_model --pr_list [0.1, 0.09, 0.06, 0.03, 0.00] --CalUniform 0 --TestUniform 0 -s 0.0
```


To reproduce the results needed to create Figures 9 and 10 respectively, please run:
```
chmod +x PRCP_s_changes.sh
./PRCP_s_changes.sh

```

To reproduce the results needed to create Figures 11, 12, 13, and 14 respectively, please run:
```
chmod +x PRCP_delta_changes.sh
./PRCP_delta_changes.sh

```

To reproduce the results needed to create Figures 15 and 16 respectively, please run:
```
python -W ignore ./PRCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 0.125 -e 50 -r 0.0 --batch_size 8192 --dataset CIFAR10 --arc ResNet -do 0.125 --My_model --pr_list [0.1, 0.09, 0.06, 0.03, 0.00] --CalUniform 0 --TestUniform 1 -s 0.0

python -W ignore ./PRCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 0.125 -e 50 -r 0.0 --batch_size 8192 --dataset CIFAR100 --arc ResNet -do 0.125 --My_model --pr_list [0.1, 0.09, 0.06, 0.03, 0.00] --CalUniform 0 --TestUniform 1 -s 0.0

```

To reproduce the results needed to create Figures 17 and 18 respectively, please run:
```
python -W ignore ./PRCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 0.125 -e 50 -r 0.0 --batch_size 8192 --dataset CIFAR10 --arc ResNet -do 0.125 --My_model --pr_list [0.1, 0.09, 0.06, 0.03, 0.00] --CalUniform 1 --TestUniform 0 -s 0.0

python -W ignore ./PRCP/main_PRCP.py -a 0.1 -cd 0.125 -ed 0.125 -e 50 -r 0.0 --batch_size 8192 --dataset CIFAR100 --arc ResNet -do 0.125 --My_model --pr_list [0.1, 0.09, 0.06, 0.03, 0.00] --CalUniform 1 --TestUniform 0 -s 0.0

```

To reproduce the results needed to create Figures 19, 20, 21, 22, and 23 respectively, please run:
```
python -W ignore ./PRCP/main_ARCP_Cifar.py -a 0.1 -d 0.125 -e 50 -r 1.0 -do 0.125 --batch_size 4096 --dataset CIFAR10 --arc ResNet --My_model --ms_list [32, 64, 128, 256, 512, 1024]
python -W ignore ./PRCP/main_ARCP_Cifar.py -a 0.1 -d 0.125 -e 50 -r 1.0 -do 0.125 --batch_size 4096 --dataset CIFAR100 --arc ResNet --My_model --ms_list [32, 64, 128, 256, 512, 1024]

```


To reproduce the results needed to create Figures 26 and 27 respectively, please run:
```
chmod +x Nattacks_ARCP.sh
./Nattacks_ARCP.sh
```