from ast import Break
import gc
import numpy as np
import seaborn as sns
from tqdm.auto import tqdm
import random
import torch
import torchvision
import os
import pickle
import sys
import argparse
from torchvision import transforms, datasets
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.stats.mstats import mquantiles

# My imports
sys.path.insert(0, './')
from RSCP.my_utils import ImageNet_scores_all, CIFAR_scores_all, find_threshold, find_threshold1, test_all, make_stats, \
    find_threshold_plain, test_all_ImageNet

from Third_Party.smoothing_adversarial.architectures import get_architecture
import RSCP.Score_Functions as scores
from RSCP.utils import evaluate_predictions, calculate_accuracy_smooth, \
    smooth_calibration_ImageNet, predict_sets_ImageNet, Smooth_Adv_ImageNet, get_scores, get_normalize_layer, \
    calibration, prediction, savefig_boxplot, save_plot, savefig_plot_clean, savefig_plot_clean1, savefig_plot_clean11, \
    distanced_sampling, distanced_sampling_imageNet, evaluate_predictions_pr, save_plot_one, save_plot_both
from Architectures.DenseNet import DenseNet
from Architectures.VGG import vgg19_bn, VGG
from Architectures.ResNet import ResNet

# parameters
parser = argparse.ArgumentParser(description='Experiments')
parser.add_argument('-a', '--alpha', default=0.1, type=float, help='Desired nominal marginal coverage')
parser.add_argument('-ed', '--eval_delta', default=0.25, type=float, help='L2 bound on the adversarial noise')
parser.add_argument('-cd', '--cal_delta', default=0.25, type=float, help='L2 bound on the adversarial noise')

parser.add_argument('-e', '--splits', default=1, type=int, help='Number of experiments to estimate coverage')
parser.add_argument('-r', '--ratio', default=1, type=float,
                    help='Ratio between adversarial noise bound to smoothing noise')
parser.add_argument('--n_s', default=1, type=int, help='Number of samples used for estimating smoothed score')
parser.add_argument('--dataset', default='CIFAR100', type=str, help='Dataset to be used: CIFAR100, CIFAR10, ImageNet')
parser.add_argument('--arc', default='ResNet', type=str,
                    help='Architecture of classifier : ResNet, DenseNet, VGG. Relevant only of My_model=True')
parser.add_argument('--My_model', action='store_true',
                    help='True for our trained model, False for Cohens. Relevent only for CIFAR10')
parser.add_argument('--batch_size', default=1024, type=int, help='Number of images to send to gpu at once')
parser.add_argument('--sigma_model', default=-1, type=float, help='std of Gaussian noise the model was trained with')
parser.add_argument('--Salman', action='store_true',
                    help='True for Salman adversarial model, False for Cohens. Relevent only for CIFAR10')
parser.add_argument('--coverage_on_label', action='store_true',
                    help='True for getting coverage and size for each label')
parser.add_argument("--device", type=int, default=0)
parser.add_argument('--gap', default=2, type=int, help='Need for better computation')
parser.add_argument('--pr', default=0.2, type=float, help='probability')
parser.add_argument('--ms_list', default=[128], type=list, help='Number of samples for calibration')
parser.add_argument('--pr_list', default=[0.1], type=list, help='Number of samples for calibration')

parser.add_argument('--two_steps', action='store_true', default=False)
# parser.add_argument('--s', type = float, default = 0.5)
parser.add_argument('-s', '--s_value', type=float,
                    default=0.0)
parser.add_argument('--no_search', action='store_true', default=False)

args = parser.parse_args()
print(f"args = {args}")
# parameters
alpha = args.alpha  # desired nominal marginal coverage
epsilon = args.cal_delta  # L2 bound on the adversarial noise
n_experiments = args.splits  # number of experiments to estimate coverage
ratio = args.ratio  # ratio between adversarial noise bound to smoothed noise
sigma_smooth = ratio * epsilon  # sigma used fro smoothing
# sigma used for training the model
if args.sigma_model != -1:
    sigma_model = args.sigma_model
else:
    sigma_model = sigma_smooth
n_smooth = args.n_s  # number of samples used for smoothing
My_model = args.My_model
N_steps = 20  # number of gradiant steps for PGD attack
dataset = args.dataset  # dataset to be used  CIFAR100', 'CIFAR10', 'ImageNet'
calibration_scores = ['HCC', 'SC']  # score function to check 'HCC', 'SC', 'SC_Reg'
model_type = args.arc  # Architecture of the model
Salman = args.Salman  # Whether to use Salman adversarial model or not
coverage_on_label = args.coverage_on_label  # Whether to calculate coverage and size per class
# number of test points (if larger then available it takes the entire set)
if dataset == 'ImageNet':
    n_test = 50000
else:
    n_test = 10000

# Validate parameters
assert dataset == 'CIFAR10' or dataset == 'CIFAR100' or dataset == 'ImageNet', 'Dataset can only be CIFAR10 or CIFAR100 or ImageNet.'
assert 0 <= alpha <= 1, 'Nominal level must be between 0 to 1'
assert not (n_smooth & (n_smooth - 1)), 'n_s must be a power of 2.'
assert not (args.batch_size & (args.batch_size - 1)), 'batch size must be a power of 2.'
assert args.batch_size >= n_smooth, 'batch size must be larger than n_s'
assert model_type == 'ResNet' or model_type == 'DenseNet' or model_type == 'VGG', 'Architecture can only be Resnet, ' \
                                                                                  'VGG or DenseNet '
assert sigma_model >= 0, 'std for training the model must be a non negative number.'
assert epsilon >= 0, 'L2 bound of noise must be non negative.'
assert isinstance(n_experiments, int) and n_experiments >= 1, 'number of splits must be a positive integer.'
assert ratio >= 0, 'ratio between sigma and delta must be non negative.'

# CIFAR100 has only my models
if dataset == "CIFAR100":
    My_model = True

# All our models are needs to be added a normalization layer
if My_model:
    normalized = True

# Cohen models already have this layer, Plus only ResNet is available for them.
else:
    normalized = False
    model_type = 'ResNet'

# The GPU used for oue experiments can only handle the following quantities of images per batch
GPU_CAPACITY = args.batch_size

# Save results to final results directories only if full data is taken. Otherwise.
if ((dataset == 'ImageNet') and (n_experiments == 50) and (n_test == 50000)) \
        or ((dataset != 'ImageNet') and (n_experiments == 50) and (n_test == 10000)):
    save_results = True
else:
    save_results = False

# calculate correction based on the Lipschitz constant
if sigma_smooth == 0:
    correction = 10000
else:
    correction = float(epsilon) / float(sigma_smooth)

# set random seed
seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# load datasets
if dataset == "CIFAR10":
    # Load train set
    train_dataset = torchvision.datasets.CIFAR10(root='./datasets/',
                                                 train=True,
                                                 transform=torchvision.transforms.ToTensor(),
                                                 download=True)
    # load test set
    test_dataset = torchvision.datasets.CIFAR10(root='./datasets',
                                                train=False,
                                                transform=torchvision.transforms.ToTensor())

elif dataset == "CIFAR100":
    # Load train set
    train_dataset = torchvision.datasets.CIFAR100(root='./datasets/',
                                                  train=True,
                                                  transform=torchvision.transforms.ToTensor(),
                                                  download=True)
    # load test set
    test_dataset = torchvision.datasets.CIFAR100(root='./datasets',
                                                 train=False,
                                                 transform=torchvision.transforms.ToTensor())
elif dataset == "ImageNet":
    # get dir of imagenet validation set
    imagenet_dir = "../Data/imagenet_val"

    # ImageNet images pre-processing
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

    # load dataset
    test_dataset = datasets.ImageFolder(imagenet_dir, transform)

else:
    print("No such dataset")
    exit(1)

# cut the size of the test set if necessary
if n_test < len(test_dataset):
    torch.manual_seed(0)
    test_dataset = torch.utils.data.random_split(test_dataset, [n_test, len(test_dataset) - n_test])[0]


n_test = len(test_dataset)

# Create Data loader for test set
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=n_test,
                                          shuffle=False)

# convert test set into tensor
examples = enumerate(test_loader)
batch_idx, (x_test, y_test) = next(examples)

# get dimension of data
rows = x_test.size()[2]
cols = x_test.size()[3]
channels = x_test.size()[1]
if dataset == 'ImageNet':
    num_of_classes = 1000
elif dataset == 'CIFAR100':
    num_of_classes = 100
else:
    num_of_classes = 10
min_pixel_value = 0.0
max_pixel_value = 1.0

# automatically choose device use gpu 0 if it is available o.w. use the cpu
args.device = torch.device("cuda:" + f"{args.device}")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = args.device
# print the chosen device robust
print("device: ", device)

# load my models
if My_model:
    if dataset == "CIFAR10":
        if model_type == 'ResNet':
            model = ResNet(depth=110, num_classes=10)
            state = torch.load('../Adversarial/checkpoints/CIFAR10_ResNet110_Robust_sigma_' + str(
                sigma_model) + '.pth.tar',
                               map_location=device)
        elif model_type == 'DenseNet':
            model = DenseNet(depth=100, num_classes=10, growthRate=12)
            state = torch.load(
                '../Adversarial/checkpoints/CIFAR10_DenseNet_sigma_' + str(sigma_model) + '.pth.tar',
                map_location=device)
        elif model_type == 'VGG':
            model = vgg19_bn(num_classes=10)
            state = torch.load(
                '../Adversarial/checkpoints/CIFAR10_VGG_sigma_' + str(sigma_model) + '.pth.tar',
                map_location=device)
        else:
            print("No such architecture")
            exit(1)
        normalize_layer = get_normalize_layer(args, "cifar10")
        model = torch.nn.Sequential(normalize_layer, model)
        model.load_state_dict(state['state_dict'])
    elif dataset == "CIFAR100":
        if model_type == 'ResNet':
            model = ResNet(depth=110, num_classes=100)
            state = torch.load(
                '../Adversarial/checkpoints/ResNet110_Robust_sigma_' + str(sigma_model) + '.pth.tar',
                map_location=device)
        elif model_type == 'DenseNet':
            model = DenseNet(depth=100, num_classes=100, growthRate=12)
            state = torch.load(
                '../Adversarial/checkpoints/DenseNet_sigma_' + str(sigma_model) + '.pth.tar',
                map_location=device)
        elif model_type == 'VGG':
            model = vgg19_bn(num_classes=100)
            state = torch.load(
                '../Adversarial/checkpoints/VGG_sigma_' + str(sigma_model) + '.pth.tar',
                map_location=device)
        else:
            print("No such architecture")
            exit(1)
        normalize_layer = get_normalize_layer(args, "cifar10")
        model = torch.nn.Sequential(normalize_layer, model)
        model.load_state_dict(state['state_dict'])
    else:
        print("No My model exist for ImageNet dataset")
        exit(1)

# load cohen and salman models
else:
    if dataset == "CIFAR10":
        if Salman:
            # checkpoint = torch.load(
            #    './Pretrained_Models/Salman/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs_multinoise/2-multitrain/eps_64/cifar10/resnet110/noise_'+str(sigma_model)+'/checkpoint.pth.tar', map_location=device)
            checkpoint = torch.load(
                './Pretrained_Models/Salman/cifar10/PGD_10steps_multiNoiseSamples/2-multitrain/eps_32/cifar10/resnet110/noise_' + str(
                    sigma_model) + '/checkpoint.pth.tar', map_location=device)
        else:
            checkpoint = torch.load(
                './Pretrained_Models/Cohen/models/cifar10/resnet110/noise_' + str(sigma_model) + '/checkpoint.pth.tar',
                map_location=device)
        model = get_architecture(checkpoint["arch"], "cifar10")
    elif dataset == "ImageNet":
        checkpoint = torch.load(
            './Pretrained_Models/Cohen/models/imagenet/resnet50/noise_' + str(sigma_model) + '/checkpoint.pth.tar',
            map_location=device)
        model = get_architecture(checkpoint["arch"], "imagenet")
    else:
        print("No Cohens model for CIFAR100")
        exit(1)
    model.load_state_dict(checkpoint['state_dict'])

# send model to device
model.to(device)

# put model in evaluation mode
model.eval()

# create indices for the test points
indices = torch.arange(n_test)

# directory to store adversarial examples and noises
directory = "./Adversarial_Examples/" + str(dataset) + "/" + str(dataset) + "/epsilon_" + str(
    epsilon) + "/sigma_model_" + str(
    sigma_model) + "/sigma_smooth_" + str(sigma_smooth) + "/n_smooth_" + str(n_smooth)

# normalization layer to my model
if normalized:
    directory = directory + "/Robust"

# different attacks for different architectures
if model_type != 'ResNet':
    directory = directory + "/" + str(model_type)

# different attacks for my or cohens and salman model
if dataset == "CIFAR10" and model_type == 'ResNet':
    if My_model:
        directory = directory + "/My_Model"
    else:
        directory = directory + "/Their_Model"
        if Salman:
            directory = directory + "/Salman"

# create the noises for the base classifiers only to check its accuracy
noises_base = torch.empty_like(x_test)
for k in range(n_test):
    torch.manual_seed(k)
    noises_base[k:(k + 1)] = torch.randn(
        (1, channels, rows, cols)) * sigma_model

# Calculate accuracy of classifier on clean test points
#acc, _, _ = calculate_accuracy_smooth(model, x_test, y_test, noises_base, num_of_classes, k=1, device=device,
#                                      GPU_CAPACITY=GPU_CAPACITY)
#print("True Model accuracy :" + str(acc * 100) + "%")

# print(s)
print("Save results on directories: " + str(save_results))
print("Searching for adversarial examples in: " + str(directory))

# Calculate accuracy of classifier on adversarial test points
# adv_acc, _, _ = calculate_accuracy_smooth(model, x_test_adv_base, y_test, noises_base, num_of_classes, k=1,
#                                          device=device, GPU_CAPACITY=GPU_CAPACITY)
# print("True Model accuracy on adversarial examples :" + str(adv_acc * 100) + "%")

del noises_base
# translate desired scores to their functions and put in a list
scores_list = []
for score in calibration_scores:
    if score == 'HCC':
        scores_list.append(scores.class_probability_score)
    elif score == 'SC':
        scores_list.append(scores.generalized_inverse_quantile_score)
    elif score == 'SC_Reg':
        scores_list.append(scores.rank_regularized_score)
    else:
        print("Undefined score function")
        exit(1)

indices = torch.arange(n_test)

# scores_simple_ours_cal = torch.zeros((len(scores_list), x_test.shape[0], args.ns, num_of_classes))

if args.dataset == 'ImageNet':
    gap = 2
    act_ns_list = [128]
    ns_x_test_list = [128]


else:
    act_ns_list = args.ms_list
    ns_x_test_list = [128]

act_ns = 0

rho_list = args.pr_list

for sg in range(len(act_ns_list)):
    n_smooth = act_ns_list[sg]
    if args.dataset == 'ImageNet':
        scores_simple_cal_ball, y_test_expand = ImageNet_scores_all(args, model, n_smooth, sigma_model, num_of_classes, x_test,
                                                        y_test,
                                                        act_ns_list[sg], scores_list,
                                                        gap1=2, channels=3, rows=224, cols=224, device=device,
                                                        GPU_CAPACITY=args.batch_size,
                                                        n_test=n_test)


    else:
        scores_simple_cal_ball, y_test_expand = CIFAR_scores_all(args, model, n_smooth, sigma_model, num_of_classes,
                                                                 x_test, y_test,
                                                                 act_ns_list[sg], scores_list, gap1=act_ns_list[sg],
                                                                 channels=3, rows=32, cols=32, device=device,
                                                                 GPU_CAPACITY=args.batch_size,
                                                                 n_test=n_test)

    # print(f"s1 = {scores_simple_cal.shape}")
    # exit(1)
    gap1 = act_ns_list[sg]

    # get base scores of whole clean test set
    print("Calculating base scores on the clean test points:\n")
    scores_simple_clean_test = get_scores(model, x_test, indices, n_smooth, sigma_model, num_of_classes, scores_list,
                                          base=True, device=device, GPU_CAPACITY=GPU_CAPACITY)

    # get smooth scores of whole clean test set
    print("Calculating smoothed scores on the clean test points:\n")
    smoothed_scores_clean_test, scores_smoothed_clean_test = get_scores(model, x_test, indices, n_smooth, sigma_smooth,
                                                                        num_of_classes, scores_list, base=False,
                                                                        device=device, GPU_CAPACITY=GPU_CAPACITY)

    idx2_list, idx11_list, idx12_list, thresholds_list = [], [], [], []
    for experiment in tqdm(range(n_experiments)):
        # Split test data into calibration and test
        idx1, idx2 = train_test_split(indices, test_size=0.5)
        idx11, idx12 = train_test_split(idx1, test_size=0.5)
        idx2_list.append(idx2)
        idx12_list.append(idx12)
        idx11_list.append(idx11)
        # calibrate base model with the desired scores and get the thresholds
        thresholds_base, _ = calibration(scores_simple=scores_simple_clean_test[:, idx11, y_test[idx11]], alpha=alpha,
                                         num_of_scores=len(scores_list), correction=correction, base=True)

        # calibrate the model with the desired scores and get the thresholds
        thresholds, bounds = calibration(scores_smoothed=scores_smoothed_clean_test[:, idx11, y_test[idx11]],
                                         smoothed_scores=smoothed_scores_clean_test[:, idx11, y_test[idx11]],
                                         alpha=alpha, num_of_scores=len(scores_list), correction=correction, base=False)

        thresholds = thresholds + thresholds_base
        print(f"thresholds_theirs1 = {thresholds}")

        thresholds_list.append(thresholds)

    n_test_expand = n_test * act_ns
    indices_expand = torch.arange(n_test_expand)
    # create dataframe for storing results

    rho_values = np.zeros((len(rho_list), len(ns_x_test_list), len(scores_list), 2))

    for sg1 in range(len(rho_list)):
        args.pr = rho_list[sg1]
        print(f"pr = {args.pr}")

        print("\nRunning experiments for " + str(n_experiments) + " random splits:\n")

        args.cvg_list_ours, args.cvg_list_ro, args.si_list_ours, args.si_list_ro = [[] for i in
                                                                                    range(len(scores_list))], [[]
                                                                                                               for i
                                                                                                               in
                                                                                                               range(
                                                                                                                   len(scores_list))], [
                                                                                       [] for i in
                                                                                       range(len(scores_list))], [[]
                                                                                                                  for
                                                                                                                  i
                                                                                                                  in
                                                                                                                  range(
                                                                                                                      len(scores_list))]

        args.cvg_list_base0, args.si_list_base0 = [[] for i in range(len(scores_list))], [[] for i in
                                                                                          range(len(scores_list))]
        args.clean_test_cvr, args.clean_test_size = [[] for i in range(len(scores_list))], [[] for i in
                                                                                            range(len(scores_list))]
        alphas = [[] for i in range(len(scores_list))]
        args.cvg_list_clean, args.si_list_clean = [[] for i in range(len(scores_list))], [[] for i in
                                                                                          range(len(scores_list))]
        vanila_cp_cvr_hps, vanila_cp_size_hps, RSCP_cvr_hps, RSCP_size_hps, ours_cvr_hps, ours_size_hps = [], [], [], [], [], []
        vanila_cp_cvr_aps, vanila_cp_size_aps, RSCP_cvr_aps, RSCP_size_aps, ours_cvr_aps, ours_size_aps = [], [], [], [], [], []

        for mm in range(len(ns_x_test_list)):
            patha = '../results/' + '/alpha_' + str(args.alpha) + '/dataset_' + str(
                args.dataset) + '/network_' + str(args.arc) + '/cal_delta' + str(
                args.cal_delta) + '/num_exp_' + str(args.splits) + '/ratio_' + str(args.ratio) + '/eval_delta_' + str(
                args.eval_delta)
            patha += '/sigma_smooth_' + str(sigma_smooth) + '/sigma_model_' + str(sigma_model) + '/num_uni_noise_' + str(
                act_ns_list[sg]) + '/Probability_' + str(1 - args.pr) + '/s_value_' + str(args.s_value)
            if My_model:
                patha += "/My_model"
            if normalized:
                patha += "/Robust"

            path = patha + '/test_ns_' + str(ns_x_test_list[mm])
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save({
                'scores_simple_cal_ball': scores_simple_cal_ball,
                'idx11_list': idx11_list,
                'idx12_list': idx12_list,
                'idx2_list': idx2_list,
                'scores_simple_clean_test': scores_simple_clean_test,
                'thresholds_list': thresholds_list,
                'y_test_expand': y_test_expand
            }, path + '/before_test.pt')

            for experiment in range(n_experiments):
                if args.no_search:
                    thresholds_ours, alpha_tilde_star = find_threshold_plain(args, y_test,
                                                                             indices,
                                                                             scores_simple_cal_ball,
                                                                             scores_list,
                                                                             correction,
                                                                             ns_x_test_list[mm],
                                                                             num_of_classes,
                                                                             idx11=idx11_list[experiment],
                                                                             idx12=idx12_list[experiment],
                                                                             channels=x_test.shape[1],
                                                                             rows=x_test.shape[2],
                                                                             cols=x_test.shape[3], device=device,
                                                                             GPU_CAPACITY=args.batch_size)
                else:
                    thresholds_ours, alpha_tilde_star = find_threshold1(args, y_test,
                                                                       indices,
                                                                       scores_simple_cal_ball,
                                                                       scores_list,
                                                                       correction,
                                                                       act_ns_list[sg],
                                                                       num_of_classes,
                                                                       idx11=idx11_list[experiment],
                                                                       idx12=idx12_list[experiment],
                                                                       channels=x_test.shape[1],
                                                                       rows=x_test.shape[2],
                                                                       cols=x_test.shape[3], device=device,
                                                                       GPU_CAPACITY=args.batch_size)

                alphas[0].append(alpha_tilde_star[0])
                alphas[1].append(alpha_tilde_star[1])
                if args.dataset == 'ImageNet':
                    test_all_ImageNet(args, y_test_expand, scores_simple_cal_ball, scores_simple_clean_test, model, scores_list, x_test, y_test, idx2_list[experiment],
                                      ns_x_test_list[mm],
                                      sigma_model,
                                      thresholds_list[experiment], correction, thresholds_ours,
                                      num_of_classes=num_of_classes,
                                      channels=3, rows=x_test.shape[2], cols=x_test.shape[3], device=device,
                                      GPU_CAPACITY=args.batch_size)
                else:
                    test_all(args, scores_simple_clean_test, model, scores_list, x_test, y_test, idx2_list[experiment],
                             ns_x_test_list[mm],
                             sigma_model,
                             thresholds_list[experiment], correction, thresholds_ours, num_of_classes=num_of_classes,
                             channels=3, rows=x_test.shape[2], cols=x_test.shape[3], device=device,
                             GPU_CAPACITY=args.batch_size)

            make_stats(args, path, alphas)

