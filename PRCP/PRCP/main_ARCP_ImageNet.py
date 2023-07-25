# general imports
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
from Third_Party.smoothing_adversarial.architectures import get_architecture
import RSCP.Score_Functions as scores
from RSCP.utils import evaluate_predictions, calculate_accuracy_smooth, \
    smooth_calibration_ImageNet, predict_sets_ImageNet, Smooth_Adv_ImageNet, get_scores, get_normalize_layer, \
    calibration, prediction, savefig_boxplot, save_plot, savefig_plot_clean, savefig_plot_clean1, savefig_plot_clean11, \
    distanced_sampling, distanced_sampling_imageNet
from Architectures.DenseNet import DenseNet
from Architectures.VGG import vgg19_bn, VGG
from Architectures.ResNet import ResNet

# parameters
parser = argparse.ArgumentParser(description='Experiments')
parser.add_argument('-a', '--alpha', default=0.1, type=float, help='Desired nominal marginal coverage')
parser.add_argument('-d', '--delta', default=0.25, type=float, help='L2 bound on the adversarial noise')
parser.add_argument('-do', '--delta_ours', default=8, type=float, help='L2 bound on the adversarial noise')

parser.add_argument('-e', '--splits', default=50, type=int, help='Number of experiments to estimate coverage')
parser.add_argument('-r', '--ratio', default=0.0, type=float,
                    help='Ratio between adversarial noise bound to smoothing noise')
parser.add_argument('--n_s', default=32, type=int, help='Number of samples used for estimating smoothed score')
parser.add_argument('--dataset', default='ImageNet', type=str, help='Dataset to be used: CIFAR100, CIFAR10, ImageNet')
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
parser.add_argument('--ns', default=128, type=int, help='Number of samples used for estimating smoothed score')
parser.add_argument('--gap', default=2, type=int, help='Number score')
parser.add_argument('--ms_list', type=list,
                    default=[128])
parser.add_argument('--two_steps', action='store_true', default=False)
# parser.add_argument('--s', type = float, default = 0.5)
parser.add_argument('--s_list', type=list,
                    default=[0.001, 0.005, 0.008, 0.01, 0.05, 0.1, 0.2, 0.3])

args = parser.parse_args()
print(f"args = {args}")
# parameters
args.device = torch.device("cuda:" + f"{args.device}")

alpha = args.alpha  # desired nominal marginal coverage
epsilon = args.delta  # L2 bound on the adversarial noise
n_experiments = args.splits  # number of experiments to estimate coverage
ratio = args.ratio  # ratio between adversarial noise bound to smoothed noise
sigma_smooth = ratio * epsilon  # sigma used fro smoothing
# sigma used for training the model
if args.sigma_model != -1:
    sigma_model = args.sigma_model
else:
    sigma_model = sigma_smooth

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

if args.dataset == 'ImageNet':
    gap = 2
    act_ns_list = args.ms_list

else:
    act_ns_list = args.ms_list  # , 256, 512, 1024]

act_ns = 0
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
    imagenet_dir = "../imagenet_val"

    # ImageNet images pre-processing
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

    # load dataset
    test_dataset = datasets.ImageFolder(imagenet_dir, transform)

else:
    print("No such dataset")
    exit(1)

for sg in range(len(act_ns_list)):

    n_smooth = act_ns_list[sg]  # number of samples used for smoothing

    # Validate parameters
    assert dataset == 'CIFAR10' or dataset == 'CIFAR100' or dataset == 'ImageNet', 'Dataset can only be CIFAR10 or CIFAR100 or ImageNet.'
    assert 0 <= alpha <= 1, 'Nominal level must be between 0 to 1'
    assert not (n_smooth & (n_smooth - 1)), 'n_s must be a power of 2.'
    assert not (args.batch_size & (args.batch_size - 1)), 'batch size must be a power of 2.'
    assert args.batch_size >= n_smooth, 'batch size must be larger than n_s'
    #assert model_type == 'ResNet' or model_type == 'DenseNet' or model_type == 'VGG' or model_type == 'resnet152', 'Architecture can only be ResNet152, ' 'VGG or DenseNet '
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
        correction = 5
    else:
        correction = float(epsilon) / float(sigma_smooth)

    # set random seed

    # cut the size of the test set if necessary
    if n_test < len(test_dataset):
        torch.manual_seed(0)
        test_dataset = torch.utils.data.random_split(test_dataset, [n_test, len(test_dataset) - n_test])[0]

    # save the sizes of each one of the sets
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
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = args.device
    # print the chosen device robust
    print("device: ", device)


    if args.arc == 'ResNet':
        model = torchvision.models.resnet50(pretrained=True, progress=True).to(device)
    elif args.arc == 'DenseNet':
        model = torchvision.models.densenet161(pretrained=True, progress=True).to(device)
    elif args.arc == 'VGG':
        model = torchvision.models.vgg16(pretrained=True, progress=True).to(device)          

    #model = torch.nn.DataParallel(model) 
    model.eval()


    # create indices for the test points
    indices = torch.arange(n_test)

    # directory to store adversarial examples and noises
    directory = "./Adversarial_Examples/" + str(dataset) + "/" + str(dataset) + "/epsilon_" + str(
        epsilon) + "/sigma_model_" + str(
        sigma_model) + "/sigma_smooth_" + str(sigma_smooth) + "/n_smooth_" + str(n_smooth)


    # different attacks for different architectures
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
    if os.path.exists(directory):
        print("Are there saved adversarial examples: Yes")
    else:
        print("Are there saved adversarial examples: No")

    # If there are no pre created adversarial examples, create new ones
    if ((dataset != 'ImageNet') and (n_test != 10000)) or (
            (dataset == 'ImageNet') and (n_test != 50000)) or not os.path.exists(directory):
        # Generate adversarial test examples
        print("Generate adversarial test examples for the smoothed model:\n")
        x_test_adv = Smooth_Adv_ImageNet(model, x_test, y_test, indices, n_smooth, sigma_smooth, N_steps, epsilon,
                                         device,
                                         GPU_CAPACITY=GPU_CAPACITY)

        # Generate adversarial test examples for the base classifier
        print("Generate adversarial test examples for the base model:\n")
        x_test_adv_base = Smooth_Adv_ImageNet(model, x_test, y_test, indices, 1, sigma_model, N_steps, epsilon, device,
                                              GPU_CAPACITY=GPU_CAPACITY)

        # Only store examples for full dataset
        if ((dataset == 'ImageNet') and (n_test == 50000)) \
                or ((dataset != 'ImageNet') and (n_test == 10000)):
            os.makedirs(directory)
            # with open(directory + "/data.pickle", 'wb') as f:
            pickle.dump([x_test_adv, x_test_adv_base], open(directory + "/data.pickle", 'wb'), protocol=4)

            #    pickle.dump([x_test_adv, x_test_adv_base], f)

    # If there are pre created adversarial examples, load them
    else:
        with open(directory + "/data.pickle", 'rb') as f:
            x_test_adv, x_test_adv_base = pickle.load(f)
    print(f"directory = {directory}")
    # Calculate accuracy of classifier on adversarial test points
    adv_acc, _, _ = calculate_accuracy_smooth(model, x_test_adv_base, y_test, noises_base, num_of_classes, k=1,
                                              device=device, GPU_CAPACITY=GPU_CAPACITY)
    print("True Model accuracy on adversarial examples :" + str(adv_acc * 100) + "%")

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

    print(str(act_ns_list[sg]), 'ss', act_ns_list[sg])
    # exit(1)
    if args.dataset == 'ImageNet':
        scores_simple_cal = []

        gap1 = 2
        deltas = np.linspace(1, args.delta_ours, int(act_ns_list[sg] / gap1))
        for ik in range(len(deltas)):
            x_test_adv_base_cal = x_test.unsqueeze(dim=1).repeat(1, gap1, 1, 1, 1).reshape(-1, channels, rows, cols)
            # epsilon1 = distanced_sampling(x_test.shape, args.delta_ours, act_ns_list[sg], gap = 2)
            x_test_adv_base_cal = x_test_adv_base_cal + distanced_sampling_imageNet(x_test.shape, deltas[ik])
            n_test_expand = n_test * gap1
            indices_expand = torch.arange(n_test_expand)
            s_temp = get_scores(model, x_test_adv_base_cal, indices_expand, n_smooth, sigma_model, num_of_classes,
                                scores_list, base=True, device=device, GPU_CAPACITY=GPU_CAPACITY)
            s_temp = torch.reshape(torch.from_numpy(s_temp),
                                   (len(calibration_scores), x_test.shape[0], gap1, num_of_classes))
            scores_simple_cal.append(s_temp)
            del s_temp, x_test_adv_base_cal
        scores_simple_cal = torch.cat(scores_simple_cal, dim=2)

    else:
        gap1 = act_ns_list[sg]

        x_test_adv_base_cal = x_test.unsqueeze(dim=1).repeat(1, gap1, 1, 1, 1).reshape(-1, channels, rows, cols)
        # y_test_expand = y_test.unsqueeze(dim = 1).repeat(1, gap).reshape(-1)

        # epsilon1 = torch.FloatTensor(x_test_adv_base_cal.shape).uniform_(0, 1)
        # epsilon1 = torch.renorm(epsilon1, p =2, dim = 0, maxnorm = args.delta_ours)
        print(f"do = {args.delta_ours}")
        epsilon1 = distanced_sampling(x_test.shape, args.delta_ours, act_ns_list[sg], gap=2)
        # print(epsilon1.shape, 'shape')

        # print(torch.norm(epsilon1, p = 2, dim = (1,2,3)), 'sas')
        # exit(1)
        n_test_expand = n_test * gap1
        indices_expand = torch.arange(n_test_expand)
        x_test_adv_base_cal = x_test_adv_base_cal + epsilon1
        s_temp = get_scores(model, x_test_adv_base_cal, indices_expand, n_smooth, sigma_model, num_of_classes,
                            scores_list, base=True, device=device, GPU_CAPACITY=GPU_CAPACITY)
        scores_simple_cal = torch.reshape(torch.from_numpy(s_temp),
                                          (len(calibration_scores), x_test.shape[0], gap1, num_of_classes))

        del epsilon1
        gc.collect()

        # s_temp = get_scores(model, x_test_adv_base_cal, indices_expand, n_smooth, sigma_model, num_of_classes,
        #                    scores_list,
        #                    base=True, device=device, GPU_CAPACITY=GPU_CAPACITY)
        # scores_simple_cal = torch.reshape(torch.from_numpy(s_temp),
        #                                  (len(calibration_scores), x_test.shape[0], gap1, num_of_classes))
        # scores_simple_cal_ours.append(s_temp)
        # print(s_temp.shape, 's13')
        # scores_simple_ours_cal[:, :, sg*gap : (sg+1)*gap, :] = s_temp

        del x_test_adv_base_cal, s_temp

    gap1 = act_ns_list[sg]
    # act_ns += gap

    # print(f"actual ns = {act_ns}")
    # scores_simple_cal = torch.cat(scores_simple_cal_ours, dim=2)

    # scores_simple_ours_cal = torch.from_numpy(scores_simple_ours_cal)
    print("Calculating scores for entire dataset:\n")

    # get base scores of whole clean test set
    print("Calculating base scores on the clean test points:\n")
    scores_simple_clean_test = get_scores(model, x_test, indices, n_smooth, sigma_model, num_of_classes, scores_list,
                                          base=True, device=device, GPU_CAPACITY=GPU_CAPACITY)
    print(scores_simple_clean_test.shape, 'score shape')
    # print(sg)

    # get smooth scores of whole clean test set
    print("Calculating smoothed scores on the clean test points:\n")
    smoothed_scores_clean_test, scores_smoothed_clean_test = get_scores(model, x_test, indices, n_smooth, sigma_smooth,
                                                                        num_of_classes, scores_list, base=False,
                                                                        device=device, GPU_CAPACITY=GPU_CAPACITY)

    # get base scores of whole clean adversarial set
    print("Calculating base scores on the adversarial test points:\n")
    scores_simple_adv_test = get_scores(model, x_test_adv_base, indices, n_smooth, sigma_model, num_of_classes,
                                        scores_list, base=True, device=device, GPU_CAPACITY=GPU_CAPACITY)

    # get smooth scores of whole adversarial test set
    print("Calculating smoothed scores on the adversarial test points:\n")
    smoothed_scores_adv_test, scores_smoothed_adv_test = get_scores(model, x_test_adv, indices, n_smooth, sigma_smooth,
                                                                    num_of_classes, scores_list, base=False,
                                                                    device=device, GPU_CAPACITY=GPU_CAPACITY)

    # print(f"score cal shape = {scores_simple_ours_cal.shape}")

    # cal_scores_ours = scores_simple_ours_cal[:, :, y_test_expand[indices_expand]]
    # print(f"ours cal shape = {scores_simple_ours_cal.shape} {len(calibration_scores)}, {x_test.shape[0]}, {act_ns}, {num_of_classes}")

    # scores_simple_cal = torch.reshape(scores_simple_ours_cal, (len(calibration_scores), x_test.shape[0], args.ns, num_of_classes))[:, :, :act_ns, :]
    # scores_simple_cal = scores_simple_ours_cal[:, :, :act_ns, :]
    # print(f"cal shape1 = {scores_simple_cal.shape}")
    # del x_test, x_test_adv, x_test_adv_base

    print(f"sigma model = {sigma_model} and sigma_smooth = {sigma_smooth}")

    # clean unnecessary data
    gc.collect()

    n_test_expand = n_test * act_ns
    indices_expand = torch.arange(n_test_expand)
    # create dataframe for storing results
    results = pd.DataFrame()

    # container for storing bounds on "CP+SS"
    quantiles = np.zeros((len(scores_list), 4, n_experiments))

    quantiles_ours = np.zeros((len(scores_list), 2, n_experiments))

    # run for n_experiments data splittings
    print("\nRunning experiments for " + str(n_experiments) + " random splits:\n")

    s_list = args.s_list
    vanila_cp_cvr_hps, vanila_cp_size_hps, RSCP_cvr_hps, RSCP_size_hps, ours_cvr_hps, ours_size_hps = [], [], [], [], [], []
    vanila_cp_cvr_aps, vanila_cp_size_aps, RSCP_cvr_aps, RSCP_size_aps, ours_cvr_aps, ours_size_aps = [], [], [], [], [], []


    cvg_list_ours, cvg_list_ro, si_list_ours, si_list_ro = [[] for i in range(len(scores_list))], [[] for i in
                                                                                                   range(
                                                                                                       len(scores_list))], [
                                                               [] for i in range(len(scores_list))], [[] for i in
                                                                                                      range(
                                                                                                          len(scores_list))]
    cvg_list_base0, cvg_list_base1, si_list_base0, si_list_base1 = [[] for i in range(len(scores_list))], [[] for i
                                                                                                           in range(
            len(scores_list))], [[] for i in range(len(scores_list))], [[] for i in range(len(scores_list))]
    # ss = [[] for i in range(len(scores_list))]
    clean_test_cvr, clean_test_size = [[] for i in range(len(scores_list))], [[] for i in range(len(scores_list))]
    alphas = [[] for i in range(len(scores_list))]

    for experiment in range(n_experiments):

        # Split test data into calibration and test
        idx1, idx2 = train_test_split(indices, test_size=0.5)
        idx11, idx12 = train_test_split(idx1, test_size=0.5)

        # idx21, idx2 = train_test_split(idx2, test_size=0.5)

        # print(f"cal shape11 = {scores_simple_cal.shape}")

        cal_scores = scores_simple_cal[:, idx11, :, y_test[idx11]].transpose(0, 1)

        # print(f"cal_scores = {cal_scores.shape}")

        alpha_tilde_star = np.zeros(len(scores_list))
        for k in range(len(alpha_tilde_star)):
            lb, ub = 0.0, 1.0
            alpha_tilde = 0.5

            while ub - lb > 0.01:
                # print(f"alpha_tilde123 = {alpha_tilde}, {k}")
                quantiles = torch.quantile(cal_scores[k, :, :], q=1 - alpha_tilde, dim=1)
                quantiles_expand = quantiles.unsqueeze(dim=1).repeat(1, gap1)
                # print(cal_scores.shape, 's', quantiles_expand.shape, 's2', quantiles.shape)
                sg1 = (1 / (gap1 * len(idx11))) * torch.sum(cal_scores[k, :, :] <= quantiles_expand)
                # print(f"sg = {sg}, {k}")
                if sg1 >= 1 - args.alpha:
                    lb = alpha_tilde
                else:
                    ub = alpha_tilde
                if ub - lb <= 0.01:
                    break
                alpha_tilde = (lb + ub) / 2
                # print(f"tilde = {alpha_tilde}, {lb}, {ub}")
            alpha_tilde_star[k] = alpha_tilde
        alphas[0].append(alpha_tilde_star[0])
        alphas[1].append(alpha_tilde_star[1])

        # print(f"tilde star = {alpha_tilde_star}")
        # exit(1)

        # print(scores_simple_clean_test[:, idx1, y_test[idx1]].shape, 's1', scores_simple_clean_test.shape)
        # calibrate base model with the desired scores and get the thresholds
        thresholds_base, _ = calibration(scores_simple=scores_simple_clean_test[:, idx11, y_test[idx11]], alpha=alpha,
                                         num_of_scores=len(scores_list), correction=correction, base=True)

        # calibrate the model with the desired scores and get the thresholds
        thresholds, bounds = calibration(scores_smoothed=scores_smoothed_clean_test[:, idx11, y_test[idx11]],
                                         smoothed_scores=smoothed_scores_clean_test[:, idx11, y_test[idx11]],
                                         alpha=alpha, num_of_scores=len(scores_list), correction=correction,
                                         base=False)

        thresholds = thresholds + thresholds_base

        thresholds_ours = np.zeros((len(scores_list), 3))

        ############### HPS ############

        lb_s, ub_s = 0.0, 1.0
        thresholds_ours_tune = np.zeros((len(scores_list), 3))
        s_value = 0.5
        while ub_s - lb_s >= 0.02:
            for p in range(1):
                thresholds_ours_tune[p, 0] = torch.quantile(
                    torch.quantile(cal_scores[p, :, :], q=1 - alpha_tilde_star[p], dim=1),
                    q=1 - s_value * args.alpha)

            predicted_adv_sets_base_ours_tune = prediction(scores_simple=scores_simple_adv_test[:, idx12, :],
                                                      num_of_scores=len(scores_list), thresholds=thresholds_ours_tune,
                                                      base=True)
            predicted_adv_sets_ours_tune = prediction(scores_simple=scores_simple_adv_test[:, idx12, :],
                                                 num_of_scores=len(scores_list), thresholds=thresholds_ours_tune, ours=True)
            for p in range(1):
                predicted_adv_sets_ours_tune[p].insert(0, predicted_adv_sets_base_ours_tune[p])
                _, marg_coverage, size = evaluate_predictions(predicted_adv_sets_ours_tune[p][0], None, y_test[idx12].numpy(),
                                                              conditional=False, coverage_on_label=coverage_on_label,
                                                              num_of_classes=num_of_classes)

            if marg_coverage - (1 - args.alpha) < 0.015 and marg_coverage >= args.alpha:
                break
            if marg_coverage >= 1 - args.alpha:
                lb_s = s_value
            else:
                ub_s = s_value
            s_value = (lb_s+ub_s)/2

        s_HPS = s_value
        for p in range(1):
            thresholds_ours[p, 0] = torch.quantile(
                torch.quantile(cal_scores[p, :, :], q=1 - alpha_tilde_star[p], dim=1),
                q=1 - s_HPS * args.alpha)
        ############### APS ############
        lb_s, ub_s = 0.0, 1.0
        thresholds_ours_tune = np.zeros((len(scores_list), 3))
        s_value = 0.5
        while ub_s - lb_s >= 0.02:
            for p in range(1, 2):
                thresholds_ours_tune[p, 0] = torch.quantile(
                    torch.quantile(cal_scores[p, :, :], q=1 - alpha_tilde_star[p], dim=1),
                    q=1 - s_value * args.alpha)

            predicted_adv_sets_base_ours_tune = prediction(scores_simple=scores_simple_adv_test[:, idx12, :],
                                                           num_of_scores=len(scores_list),
                                                           thresholds=thresholds_ours_tune,
                                                           base=True)
            predicted_adv_sets_ours_tune = prediction(scores_simple=scores_simple_adv_test[:, idx12, :],
                                                      num_of_scores=len(scores_list), thresholds=thresholds_ours_tune,
                                                      ours=True)
            for p in range(1, 2):
                predicted_adv_sets_ours_tune[p].insert(0, predicted_adv_sets_base_ours_tune[p])
                _, marg_coverage, size = evaluate_predictions(predicted_adv_sets_ours_tune[p][0], None,
                                                              y_test[idx12].numpy(),
                                                              conditional=False,
                                                              coverage_on_label=coverage_on_label,
                                                              num_of_classes=num_of_classes)

            if marg_coverage - (1 - args.alpha) < 0.015 and marg_coverage >= 1 - args.alpha:
                break
            if marg_coverage >= 1 - args.alpha:
                lb_s = s_value
            else:
                ub_s = s_value
            s_value = (lb_s + ub_s) / 2

        s_APS = s_value
        for p in range(1, 2):
            #print(f"p APS= {p}")
            thresholds_ours[p, 0] = torch.quantile(
                    torch.quantile(cal_scores[p, :, :], q=1 - alpha_tilde_star[p], dim=1),
                    q=1 - s_APS * args.alpha)
            ############### APS ############

        # print(thresholds_ours, 's1')
        ################# ours ################

        # generate prediction sets on the clean test set for base model
        predicted_clean_sets_base = prediction(scores_simple=scores_simple_clean_test[:, idx2, :],
                                               num_of_scores=len(scores_list), thresholds=thresholds, base=True)

        # generate robust prediction sets on the clean test set
        predicted_clean_sets = prediction(scores_smoothed=scores_smoothed_clean_test[:, idx2, :],
                                          smoothed_scores=smoothed_scores_clean_test[:, idx2, :],
                                          num_of_scores=len(scores_list), thresholds=thresholds,
                                          correction=correction, base=False)

        # generate prediction sets on the adversarial test set for base model
        predicted_adv_sets_base = prediction(scores_simple=scores_simple_adv_test[:, idx2, :],
                                             num_of_scores=len(scores_list), thresholds=thresholds, base=True)

        # generate robust prediction sets on the adversarial test set
        predicted_adv_sets = prediction(scores_smoothed=scores_smoothed_adv_test[:, idx2, :],
                                        smoothed_scores=smoothed_scores_adv_test[:, idx2, :],
                                        num_of_scores=len(scores_list), thresholds=thresholds,
                                        correction=correction, base=False)

        ################### ours #############
        predicted_adv_sets_base_ours = prediction(scores_simple=scores_simple_adv_test[:, idx2, :],
                                                  num_of_scores=len(scores_list), thresholds=thresholds_ours,
                                                  base=True)
        predicted_adv_sets_ours = prediction(scores_simple=scores_simple_adv_test[:, idx2, :],
                                             num_of_scores=len(scores_list), thresholds=thresholds_ours, ours=True)

        ################ ours ##############
        # print(predicted_adv_sets_base_ours.shape, 's', predicted_adv_sets_base.shape)
        # arrange results on clean test set in dataframe
        for p in range(len(scores_list)):
            predicted_clean_sets[p].insert(0, predicted_clean_sets_base[p])
            predicted_adv_sets[p].insert(0, predicted_adv_sets_base[p])
            predicted_adv_sets_ours[p].insert(0, predicted_adv_sets_base_ours[p])

            _, marg_coverage, size = evaluate_predictions(predicted_adv_sets[p][3], None, y_test[idx2].numpy(),
                                                          conditional=False, coverage_on_label=coverage_on_label,
                                                          num_of_classes=num_of_classes)

            cvg_list_ro[p].append(marg_coverage)
            si_list_ro[p].append(size)
            del marg_coverage, size

            _, marg_coverage, size = evaluate_predictions(predicted_adv_sets[p][0], None, y_test[idx2].numpy(),
                                                          conditional=False, coverage_on_label=coverage_on_label,
                                                          num_of_classes=num_of_classes)

            cvg_list_base0[p].append(marg_coverage)
            si_list_base0[p].append(size)
            del marg_coverage, size

            _, marg_coverage, size = evaluate_predictions(predicted_adv_sets[p][2], None, y_test[idx2].numpy(),
                                                          conditional=False, coverage_on_label=coverage_on_label,
                                                          num_of_classes=num_of_classes)

            cvg_list_base1[p].append(marg_coverage)
            si_list_base1[p].append(size)
            del marg_coverage, size

            # cvg_list_base0, cvg_list_base1, si_list_base0, si_list_base1

            _, marg_coverage, size = evaluate_predictions(predicted_adv_sets_ours[p][0], None, y_test[idx2].numpy(),
                                                          conditional=False, coverage_on_label=coverage_on_label,
                                                          num_of_classes=num_of_classes)
            # print(f"marg_coverage = {marg_coverage}, size = {size}")
            cvg_list_ours[p].append(marg_coverage)
            si_list_ours[p].append(size)
            del marg_coverage, size

            score_name = calibration_scores[p]
            methods_list = [score_name + '_simple', score_name + '_smoothed_classifier',
                            score_name + '_smoothed_score',
                            score_name + '_smoothed_score_correction']
            for r, method in enumerate(methods_list):
                res, marg_coverage, size = evaluate_predictions(predicted_clean_sets[p][r], None,
                                                                y_test[idx2].numpy(),
                                                                conditional=False,
                                                                coverage_on_label=coverage_on_label,
                                                                num_of_classes=num_of_classes)

                if r == 0:
                    clean_test_cvr[p].append(marg_coverage), clean_test_size[p].append(size)
                res['Method'] = methods_list[r]
                res['noise_L2_norm'] = 0
                res['Black box'] = 'CNN sigma = ' + str(sigma_model)
                # Add results to the list
                results = results.append(res)

        # arrange results on adversarial test set in dataframe
        for p in range(len(scores_list)):
            score_name = calibration_scores[p]
            methods_list = [score_name + '_simple', score_name + '_smoothed_classifier',
                            score_name + '_smoothed_score',
                            score_name + '_smoothed_score_correction']
            for r, method in enumerate(methods_list):
                res, marg_coverage, size = evaluate_predictions(predicted_adv_sets[p][r], None,
                                                                y_test[idx2].numpy(),
                                                                conditional=False,
                                                                coverage_on_label=coverage_on_label,
                                                                num_of_classes=num_of_classes)
                res['Method'] = methods_list[r]
                res['noise_L2_norm'] = epsilon
                res['Black box'] = 'CNN sigma = ' + str(sigma_model)
                # Add results to the list
                results = results.append(res)

        # ours
        # print("ours")
        for p in range(len(scores_list)):
            score_name = calibration_scores[p]
            methods_list = [score_name + '_ours_base', score_name + '_ours_smoothed_classifier',
                            score_name + '_ours_smoothed_score']
            for r, method in enumerate(methods_list):
                res, marg_coverage, size = evaluate_predictions(predicted_adv_sets_ours[p][r], None,
                                                                y_test[idx2].numpy(),
                                                                conditional=False,
                                                                coverage_on_label=coverage_on_label,
                                                                num_of_classes=num_of_classes)
                res['Method'] = methods_list[r]
                res['noise_L2_norm'] = epsilon
                res['Black box'] = 'CNN sigma = ' + str(sigma_model)
                # Add results to the list
                results = results.append(res)

        # clean memory
    del idx1, idx2, predicted_clean_sets, predicted_clean_sets_base, predicted_adv_sets, predicted_adv_sets_base, bounds, thresholds, thresholds_base, thresholds_ours
    gc.collect()

        # print(si_list_ours[1], cvg_list_ours[1])


    patha = '../results/' + '/alpha_' + str(args.alpha) + '/delta_' + str(
        args.delta) + '/num_exp_' + str(args.splits) + '/ratio_' + str(args.ratio) + '/ours_delta_' + str(
        args.delta_ours)
    patha += '/sigma_smooth_' + str(sigma_smooth) + '/sigma_model_' + str(sigma_model) + '/dataset_' + str(
        args.dataset) + '/network_' + str(args.arc) + '/num_uni_noise_' + str(act_ns_list[sg])


    #path = patha + '/s_' + str(s_list[kkk])
    if not os.path.exists(patha):
        os.makedirs(patha)

    stat_list = pd.DataFrame({
        'ours_APS_size': si_list_ours[1],
        'ours_APS_cvg': cvg_list_ours[1],
        'RSCP_APS_size': si_list_ro[1],
        'RSCP_APS_cvg': cvg_list_ro[1],

        'ours_HPS_size': si_list_ours[0],
        'ours_HPS_cvg': cvg_list_ours[0],
        'RSCP_HPS_size': si_list_ro[0],
        'RSCP_HPS_cvg': cvg_list_ro[0],

        'vanilla_APS_size': si_list_base0[1],
        'vanilla_APS_cvg': cvg_list_base0[1],
        'cp_ss_APS_size': si_list_base1[1],
        'cp_ss_APS_cvg': cvg_list_base1[1],

        'vanilla_HPS_size': si_list_base0[0],
        'vanilla_HPS_cvg': cvg_list_base0[0],
        'cp_ss_HPS_size': si_list_base1[0],
        'cp_ss_HPS_cvg': cvg_list_base1[0],

        'alphas_APS': alphas[0],
        'alphas_HPS': alphas[1],

        'clean_test_HPS_cvr': clean_test_cvr[0],
        'clean_test_HPS_size': clean_test_size[0],
        'clean_test_APS_cvr': clean_test_cvr[1],
        'clean_test_APS_size': clean_test_size[1],

    })


    torch.save(stat_list, patha + '/stats.pkl')


"""    
    std_names = {f"{name}_std": round(np.std(elem), 3) for name, elem in stat_list}
    mean_names = {f"{name}_mean": round(np.mean(elem), 3) for name, elem in stat_list}
    # print(f"mean_names = {mean_names}")
    series = pd.Series(data={**std_names, **mean_names})
    # print(f"series = {series}")
    vanila_cp_cvr_hps.append(mean_names['vanilla_hcc_cvg_mean']), vanila_cp_size_hps.append(
        mean_names['vanilla_hcc_size_mean'])
    RSCP_cvr_hps.append(mean_names['RSCP_hcc_cvg_mean']), RSCP_size_hps.append(mean_names['RSCP_hcc_size_mean'])
    ours_cvr_hps.append(mean_names['ours_hcc_cvg_mean']), ours_size_hps.append(mean_names['ours_hcc_size_mean'])
    vanila_cp_cvr_aps.append(mean_names['vanilla_sc_cvg_mean']), vanila_cp_size_aps.append(
        mean_names['vanilla_sc_size_mean'])
    RSCP_cvr_aps.append(mean_names['RSCP_sc_cvg_mean']), RSCP_size_aps.append(mean_names['RSCP_sc_size_mean'])
    ours_cvr_aps.append(mean_names['ours_sc_cvg_mean']), ours_size_aps.append(mean_names['ours_sc_size_mean'])

    all1 = {'Vanilla CP(APS)': cvg_list_base0[1], 'Vanilla CP(HPS)': cvg_list_base0[0], 'RSCP(APS)': cvg_list_ro[1],
            'RSCP(HPS)': cvg_list_ro[0], 'ARCP(APS)': cvg_list_ours[1], 'ARCP(HPS)': cvg_list_ours[0]}

    all2 = {'Vanilla CP(APS)': si_list_base0[1], 'Vanilla CP(HPS)': si_list_base0[0], 'RSCP(APS)': si_list_ro[1],
            'RSCP(HPS)': si_list_ro[0], 'ARCP(APS)': si_list_ours[1], 'ARCP(HPS)': si_list_ours[0]}

    first_fig_hcc_cvg = {'Vanilla CP1': clean_test_cvr[0], 'Vanilla CP2': cvg_list_base0[0], 'RSCP': cvg_list_ro[0],
                            'Our Method': cvg_list_ours[0]}
    first_fig_hcc_size = {'Vanilla CP1': clean_test_size[0], 'Vanilla CP2': si_list_base0[0], 'RSCP': si_list_ro[0],
                            'Our Method': si_list_ours[0]}

    first_fig_sc_cvg = {'Vanilla CP1': clean_test_cvr[1], 'Vanilla CP2': cvg_list_base0[1], 'RSCP': cvg_list_ro[1],
                        'Our Method': cvg_list_ours[1]}
    first_fig_sc_size = {'Vanilla CP1': clean_test_size[1], 'Vanilla CP2': si_list_base0[1], 'RSCP': si_list_ro[1],
                            'Our Method': si_list_ours[1]}

    first_fig_hcc_cvg11 = {'Vanilla CP1': clean_test_cvr[0], 'Vanilla CP2': cvg_list_base0[0],
                            'Our Method': cvg_list_ours[0]}
    first_fig_hcc_size11 = {'Vanilla CP1': clean_test_size[0], 'Vanilla CP2': si_list_base0[0],
                            'Our Method': si_list_ours[0]}

    first_fig_sc_cvg11 = {'Vanilla CP1': clean_test_cvr[1], 'Vanilla CP2': cvg_list_base0[1],
                            'Our Method': cvg_list_ours[1]}
    first_fig_sc_size11 = {'Vanilla CP1': clean_test_size[1], 'Vanilla CP2': si_list_base0[1],
                            'Our Method': si_list_ours[1]}

    first_fig_hcc_cvg1 = {'Vanilla CP': cvg_list_base0[0], 'RSCP': cvg_list_ro[0], 'ARLCP': cvg_list_ours[0]}
    first_fig_hcc_size1 = {'Vanilla CP': si_list_base0[0], 'RSCP': si_list_ro[0], 'ARLCP': si_list_ours[0]}

    first_fig_sc_cvg1 = {'Vanilla CP': cvg_list_base0[1], 'RSCP': cvg_list_ro[1], 'ARLCP': cvg_list_ours[1]}
    first_fig_sc_size1 = {'Vanilla CP': si_list_base0[1], 'RSCP': si_list_ro[1], 'ARLCP': si_list_ours[1]}

    del cvg_list_ours, cvg_list_ro, si_list_ours, si_list_ro, cvg_list_base0, cvg_list_base1, si_list_base0, si_list_base1, alphas, clean_test_cvr, clean_test_size





    
    # print(f"Saving results in {path}")

    ax.savefig(patha + "/Marginal" + add_string + ".pdf")

    if save_results:
        ax.savefig(directory + "/Marginal" + add_string + ".pdf")
    else:
        ax.savefig("./Results/Marginal" + add_string + ".pdf")

    # plot set sizes results
    # print(results)
    ax = sns.catplot(x="Black box", y="Size",
                        hue="Method", col="noise_L2_norm",
                        data=results, kind="box",
                        height=4, aspect=.7)
    for i, graph in enumerate(ax.axes[0]):
        graph.set(xlabel='Classifier', ylabel='Set Size')

    if save_results:
        ax.savefig(directory + "/Size" + add_string + ".pdf")
    else:
        ax.savefig("./Results/Size" + add_string + ".pdf")

    ax.savefig(patha + "/Size" + add_string + ".pdf")
    results.to_csv(patha + "/results" + add_string + ".csv")
    
    
    series.to_csv(patha + "/results_ours.csv")
    savefig_boxplot(args, path1=patha + "/coverage_ours.png", all=all1, label=None, ylabel='Marginal coverage',
                    xlabel='Methods')
    savefig_boxplot(args, path1=patha + "/size_ours.png", all=all2, label=None, ylabel='Average set size',
                    xlabel='Methods')

    savefig_plot_clean(args, path1=patha + '/hcc_clean_cvg.png', all=first_fig_hcc_cvg, ylabel='Marginal coverage',
                        xlabel='Methods')
    savefig_plot_clean(args, path1=patha + '/hcc_clean_size.png', all=first_fig_hcc_size, ylabel='Average set size',
                        xlabel='Methods')
    savefig_plot_clean(args, path1=patha + '/sc_clean_cvg.png', all=first_fig_sc_cvg, ylabel='Marginal coverage',
                        xlabel='Methods')
    savefig_plot_clean(args, path1=patha + '/sc_clean_size.png', all=first_fig_sc_size, ylabel='Average set size',
                        xlabel='Methods')

    savefig_plot_clean1(args, path1=patha + '/hcc_clean_cvg1.png', all=first_fig_hcc_cvg1,
                        ylabel='Marginal coverage', xlabel='Methods')
    savefig_plot_clean1(args, path1=patha + '/hcc_clean_size1.png', all=first_fig_hcc_size1,
                        ylabel='Average set size', xlabel='Methods')
    savefig_plot_clean1(args, path1=patha + '/sc_clean_cvg1.png', all=first_fig_sc_cvg1, ylabel='Marginal coverage',
                        xlabel='Methods')
    savefig_plot_clean1(args, path1=patha + '/sc_clean_size1.png', all=first_fig_sc_size1, ylabel='Average set size',
                        xlabel='Methods')

    savefig_plot_clean11(args, path1=patha + '/hcc_clean_cvg11.png', all=first_fig_hcc_cvg11,
                            ylabel='Marginal coverage', xlabel='Methods')
    savefig_plot_clean11(args, path1=patha + '/hcc_clean_size11.png', all=first_fig_hcc_size11,
                            ylabel='Average set size', xlabel='Methods')
    savefig_plot_clean11(args, path1=patha + '/sc_clean_cvg11.png', all=first_fig_sc_cvg11,
                            ylabel='Marginal coverage', xlabel='Methods')
    savefig_plot_clean11(args, path1=patha + '/sc_clean_size11.png', all=first_fig_sc_size11,
                            ylabel='Average set size', xlabel='Methods')


"""