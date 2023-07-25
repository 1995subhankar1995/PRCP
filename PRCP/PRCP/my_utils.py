from RSCP.utils import distanced_sampling_imageNet, get_scores, distanced_sampling, calibration, prediction, \
    evaluate_predictions_pr, evaluate_predictions_no_pr, evaluate_predictions
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch
from Third_Party.smoothing_adversarial.attacks import PGD_L2, DDN
import numpy as np
import gc
import pandas as pd
from torch.nn.functional import softmax
from scipy.stats import rankdata
from numpy.random import default_rng
from scipy.stats.mstats import mquantiles
from scipy.stats import norm
from tqdm import tqdm
from typing import List
from numpy.random import default_rng
import os
import random
import pickle
import seaborn as sns

def Nattack(args, mu_0, std, dataloader, model, sigma_smooth):
  for t in range(args.T):
    for data, labels in dataloader:
      data = data.to(args.device)
      labels = labels.to(args.device)
      epsilon = torch.randn(data.shape)
      mu_exp = torch.unsqueeze(mu_0, dim = 0).repeat(data.shape[0], 1, 1, 1)  
      std_exp = torch.unsqueeze(std, dim = 0).repeat(data.shape[0], 1, 1, 1)
      epsilon_new = mu_exp + epsilon * std_exp
      g = 0.5 * (torch.tanh(epsilon_new) + 1)
      g = g.to(args.device)
      delta = g - data
      mask = torch.norm(delta, p = 2, dim = (1,2,3)) >= tau_lim
      delta_norm = torch.norm(delta, p = 2, dim = (1,2,3))
      mask_delta = mask * delta_norm
      mask_delta = mask_delta/tau_lim
      mask_delta[torch.where(mask_delta == 0.00)] = 1.0
      #print(mask_delta)
      mask_delta = mask_delta.unsqueeze(dim = 1).unsqueeze(dim = 2).unsqueeze(dim = 3)
      #print(mask_delta)
      delta = delta/mask_delta
      data = data + delta

      outs = model(data)
      outs = m(outs) #softmax outputs
      #print(torch.sum(outs, dim = 1))
      #print(s1)
      fx = -outs[torch.arange(len(outs)), labels]
      #print(fx.shape, 's0')
      fx_std, fx_mean = torch.std_mean(fx)
      fx_norm = (fx - fx_mean)/(fx_std + 1e-6)
      #print(fx_norm.shape, 's1')
      fx_norm = fx_norm.unsqueeze(dim = 1).unsqueeze(dim = 2).unsqueeze(dim = 3)
      part2 = torch.sum(fx_norm*epsilon.to(args.device), dim = 0)
      part2 = part2/std.to(args.device)
      mu_t = mu_0 - args.lr/data.shape[0] * part2.to('cpu')
      mu_0 = mu_t.to('cpu')

      #print(data.shape, labels.shape)

      del mu_t, data, labels, epsilon, std_exp, epsilon_new, g, delta, mask, delta_norm, mask_delta, outs, fx, fx_std, fx_mean, fx_norm, part2

  print(len(dataloader), '00')
  adv_data = torch.zeros((10000, 3, 32, 32))
  for i, (data, labels) in enumerate(dataloader):
    data = data.to(args.device)
    labels = labels.to(args.device)
    epsilon = torch.randn(data.shape)
    mu_exp = torch.unsqueeze(mu_0, dim = 0).repeat(data.shape[0], 1, 1, 1)  
    std_exp = torch.unsqueeze(std, dim = 0).repeat(data.shape[0], 1, 1, 1)
    epsilon_new = mu_exp + epsilon * std_exp
    g = 0.5 * (torch.tanh(epsilon_new) + 1)
    g = g.to(args.device)
    delta = g - data
    mask = torch.norm(delta, p = 2, dim = (1,2,3)) >= tau_lim
    delta_norm = torch.norm(delta, p = 2, dim = (1,2,3))
    mask_delta = mask * delta_norm
    mask_delta = mask_delta/tau_lim
    mask_delta[torch.where(mask_delta == 0.00)] = 1.0
    #print(mask_delta)
    mask_delta = mask_delta.unsqueeze(dim = 1).unsqueeze(dim = 2).unsqueeze(dim = 3)
    #print(mask_delta)
    delta = delta/mask_delta
    #print(delta.shape, '1')
    delta = torch.renorm(delta, p = 2, dim = 0, maxnorm = sigma_smooth)
    #print(delta.shape, '2')

    data = data + delta
    adv_data[i*data.shape[0]:(i+1)*data.shape[0]] = data
  return adv_data


def SavePlot(args, path, x = None, y = None, column = None, data = None, kind = None, legend = False, rotation = None):
    plt.style.use('seaborn')

    font = 17
    #colors_list = sns.color_palette("husl", len(scores_list) * 4)

    sns.set(rc={"figure.figsize":(2.5, 4)})

    s1 = sns.catplot(x = x, y = y, column = column,
                 data=data, kind=kind, legend = legend)
    
    if y == 'Marginal Coverage':
        plt.axhline(1 - args.alpha, ls='--', color="red")
    #black_line = mlines.Line2D([], [], color='red', linestyle='--',
    #                      markersize=font, label='Nominal Coverage')
    #plt.legend(handles=[black_line], loc='best', frameon=True, fontsize=12, fancybox=True, framealpha=0.9)
    if y == 'Prediction Size':
        s1.set(ylim=(-0.1, 10.1))
    else:
        s1.set(ylim=(-0.01, 1.01))


    plt.yticks(fontsize=font)
    plt.xticks(fontsize=font, rotation=45)
    s1.savefig(path, dpi = 100, bbox_inches='tight', pad_inches=.1)
    plt.close('all')


def ImageNet_scores_all(args, model, n_smooth, sigma_model, num_of_classes, x_test, y_test, n_s, scores_list, gap1=2,
                        channels=3, rows=224, cols=224, device='cpu', GPU_CAPACITY=1024, n_test=50000):
    scores_simple_cal = []
    y_expand = []
    deltas = np.linspace(1, args.cal_delta, int(n_s / gap1))
    for i in range(len(deltas)):
        x_test_adv_base_cal = x_test.unsqueeze(dim=1).repeat(1, gap1, 1, 1, 1).reshape(-1, channels, rows, cols) + distanced_sampling_imageNet(x_test.shape, deltas[i])
        y_test_expand = y_test.unsqueeze(dim = 1).repeat(1, gap1)
        y_expand.append(y_test_expand)
        #x_test_adv_base_cal = x_test_adv_base_cal + distanced_sampling_imageNet(x_test.shape, deltas[i])
        n_test_expand = n_test * gap1
        indices_expand = torch.arange(n_test_expand)
        s_temp = get_scores(model, x_test_adv_base_cal, indices_expand, n_smooth, sigma_model, num_of_classes,
                            scores_list, base=True, device=device, GPU_CAPACITY=GPU_CAPACITY)
        s_temp = torch.reshape(torch.from_numpy(s_temp),
                               (len(scores_list), x_test.shape[0], gap1, num_of_classes))
        scores_simple_cal.append(s_temp)
        del s_temp, x_test_adv_base_cal
    scores_simple_cal = torch.cat(scores_simple_cal, dim=2)
    #y_expand = torch.cat(y_expand, dim=1).reshape(-1)
    return scores_simple_cal, y_expand


def CIFAR_scores_all(args, model, n_smooth, sigma_model, num_of_classes, x_test, y_test, n_s, scores_list, gap1=2,
                     channels=3,
                     rows=32, cols=32, device='cpu', GPU_CAPACITY=1024, n_test=10000):
    x_test_adv_base_cal = x_test.unsqueeze(dim=1).repeat(1, gap1, 1, 1, 1).reshape(-1, channels, rows, cols)
    y_test_expand = y_test.unsqueeze(dim=1).repeat(1, n_s).reshape(-1)

    # epsilon1 = torch.FloatTensor(x_test_adv_base_cal.shape).uniform_(0, 1)
    # epsilon1 = torch.renorm(epsilon1, p =2, dim = 0, maxnorm = args.delta_ours)
    # print(f"do = {args.delta_ours}")
    sh1, sh2, sh3, sh4 = x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3]
    sh = [sh1, sh2, sh3, sh4]
    epsilon1 = distanced_sampling(sh, args.cal_delta, n_s, gap=2, uniform = True)
    # print(epsilon1.shape, 'shape')

    # print(torch.norm(epsilon1, p = 2, dim = (1,2,3)), 'sas')
    # exit(1)
    n_test_expand = n_test * gap1
    indices_expand = torch.arange(n_test_expand)
    x_test_adv_base_cal = x_test_adv_base_cal + epsilon1
    s_temp = get_scores(model, x_test_adv_base_cal, indices_expand, n_smooth, sigma_model, num_of_classes,
                        scores_list, base=True, device=device, GPU_CAPACITY=GPU_CAPACITY)
    scores_simple_cal = torch.reshape(torch.from_numpy(s_temp),
                                      (len(scores_list), x_test.shape[0], gap1, num_of_classes))

    del epsilon1, x_test_adv_base_cal, s_temp
    return scores_simple_cal, y_test_expand



def CIFAR_scores_all1(args, model, n_smooth, sigma_model, num_of_classes, x_test, y_test, n_s, scores_list, gap1=2,
                     channels=3,
                     rows=32, cols=32, device='cpu', GPU_CAPACITY=1024, n_test=10000):
    x_test_adv_base_cal = x_test.unsqueeze(dim=1).repeat(1, gap1, 1, 1, 1).reshape(-1, channels, rows, cols)
    y_test_expand = y_test.unsqueeze(dim=1).repeat(1, n_s).reshape(-1)

    # epsilon1 = torch.FloatTensor(x_test_adv_base_cal.shape).uniform_(0, 1)
    # epsilon1 = torch.renorm(epsilon1, p =2, dim = 0, maxnorm = args.delta_ours)
    # print(f"do = {args.delta_ours}")
    sh1, sh2, sh3, sh4 = x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3]
    sh = [sh1, sh2, sh3, sh4]
    epsilon1 = distanced_sampling(sh, args.delta_ours, n_s, gap=2, uniform = True)
    # print(epsilon1.shape, 'shape')

    # print(torch.norm(epsilon1, p = 2, dim = (1,2,3)), 'sas')
    # exit(1)
    n_test_expand = n_test * gap1
    indices_expand = torch.arange(n_test_expand)
    x_test_adv_base_cal = x_test_adv_base_cal + epsilon1
    #print(x_test_adv_base_cal.shape, 'p1')
    x_test_adv_base_cal = torch.reshape(x_test_adv_base_cal, (x_test.shape[0], gap1, x_test.shape[1], x_test.shape[2], x_test.shape[3]))
    #print(x_test_adv_base_cal.shape, 'p2')

    print("Calculate Conformal Scores:")
    all_scores_list = []
    for i in tqdm(range(gap1)):
        #print(f"shape = {x_test_adv_base_cal[:, i, :, :, :].shape}, {i}")

        s_temp = get_scores(model, x_test_adv_base_cal[:, i, :, :, :], indices_expand, n_smooth, sigma_model, num_of_classes,
                        scores_list, base=True, device=device, GPU_CAPACITY=GPU_CAPACITY)
        #print(f"s_temp = {s_temp.shape}")
        all_scores_list.append(torch.from_numpy(s_temp))

    scores_simple_cal = torch.cat(all_scores_list, dim = 1).reshape(len(scores_list), x_test.shape[0], gap1, num_of_classes)

    del epsilon1, x_test_adv_base_cal, s_temp
    return scores_simple_cal, y_test_expand


def find_threshold(args, y_test, indices, scores_simple_cal_ball, scores_list, correction, n_s_test,
                   num_of_classes, idx11, idx12, channels=3, rows=32, cols=32, device='cpu', GPU_CAPACITY=1024):
    thresholds_ours = np.zeros((len(scores_list), 3))
    cal_scores1 = scores_simple_cal_ball[:, idx11, :, y_test[idx11]].transpose(0, 1)
    samples = torch.randint(0, cal_scores1.shape[2], (cal_scores1.shape[2],))
    cal_scores = cal_scores1[:, :, samples][:, :, :int((1 - args.pr) * cal_scores1.shape[2])]

    alpha_tilde_star = np.zeros(len(scores_list))
    for k in range(len(scores_list)):
        lb, ub = 0.0, 1.0
        alpha_tilde = 0.5

        while ub - lb > 0.01:
            quantiles = torch.quantile(cal_scores[k, :, :], q=1 - alpha_tilde, dim=1)
            quantiles_expand = quantiles.unsqueeze(dim=1).repeat(1, cal_scores.shape[2])
            frac = (1 / (cal_scores.shape[2] * len(idx11))) * torch.sum(cal_scores[k, :, :] <= quantiles_expand)

            if frac >= 1 - args.alpha:
                lb = alpha_tilde
            else:
                ub = alpha_tilde
            if ub - lb <= 0.01:
                break
            alpha_tilde = (lb + ub) / 2
        alpha_tilde_star[k] = alpha_tilde

    lb_s, ub_s = 0.0, 1.0
    thresholds_ours_tune = np.zeros((len(scores_list), 3))
    s_value = 0.5
    while ub_s - lb_s >= 0.02:
        for p in range(1):
            thresholds_ours_tune[p, 0] = torch.quantile(
                torch.quantile(cal_scores[p, :, :], q=1 - alpha_tilde_star[p], dim=1),
                q=1 - s_value * args.alpha)

        scores_simple_cal_new = scores_simple_cal_ball[:, idx12, :, :]
        y_test_expand = y_test[idx12].unsqueeze(dim=1).repeat(1, n_s_test).reshape(-1, )
        y_test_expand = y_test_expand.numpy()

        scores_simple_cal_new = scores_simple_cal_new.reshape(len(scores_list),
                                                              len(idx12) * n_s_test,
                                                              num_of_classes).numpy()

        ################### ours #############
        predicted_adv_sets_base_ours = prediction(scores_simple=scores_simple_cal_new,
                                                  num_of_scores=len(scores_list), thresholds=thresholds_ours_tune,
                                                  base=True)
        predicted_adv_sets_ours = prediction(scores_simple=scores_simple_cal_new,
                                             num_of_scores=len(scores_list), thresholds=thresholds_ours_tune,
                                             ours=True)
        for p in range(1):
            predicted_adv_sets_ours[p].insert(0, predicted_adv_sets_base_ours[p])

            marg_coverage, size = evaluate_predictions_no_pr(predicted_adv_sets_ours[p][0], None, y_test_expand,
                                                          n=len(idx12), m=n_s_test,
                                                          pr=args.pr)

        ###############
        if marg_coverage - (1 - args.alpha) < 0.015 and marg_coverage >= args.alpha:
            break
        if marg_coverage >= 1 - args.alpha:
            lb_s = s_value
        else:
            ub_s = s_value
        s_value = (lb_s + ub_s) / 2

    s_HPS = s_value
    s_HPS = 0.5 # Remove this line if we want to binary search over s value
    d_HPS = 0.03
    for p in range(1):
        thresholds_ours[p, 0] = torch.quantile(
            torch.quantile(cal_scores[p, :, :], q=1 - alpha_tilde_star[p], dim=1),
            q=1 - s_HPS * args.alpha + d_HPS)
    ############### APS ############
    lb_s, ub_s = 0.0, 1.0
    thresholds_ours_tune = np.zeros((len(scores_list), 3))
    s_value = 0.5
    while ub_s - lb_s >= 0.02:
        # print(f"s value1 = {s_value}")
        for p in range(1, 2):
            thresholds_ours_tune[p, 0] = torch.quantile(
                torch.quantile(cal_scores[p, :, :], q=1 - alpha_tilde_star[p], dim=1),
                q=1 - s_value * args.alpha)

        scores_simple_cal_new = scores_simple_cal_ball[:, idx12, :, :]

        scores_simple_cal_new = scores_simple_cal_new.reshape(len(scores_list),
                                                              len(idx12) * n_s_test,
                                                              num_of_classes).numpy()

        ################### ours #############
        predicted_adv_sets_base_ours = prediction(scores_simple=scores_simple_cal_new,
                                                  num_of_scores=len(scores_list), thresholds=thresholds_ours_tune,
                                                  base=True)
        predicted_adv_sets_ours = prediction(scores_simple=scores_simple_cal_new,
                                             num_of_scores=len(scores_list), thresholds=thresholds_ours_tune,
                                             ours=True)

        for p in range(1, 2):
            predicted_adv_sets_ours[p].insert(0, predicted_adv_sets_base_ours[p])

            marg_coverage, size = evaluate_predictions_no_pr(predicted_adv_sets_ours[p][0], None, y_test_expand,
                                                          n=len(idx12), m=n_s_test,
                                                          pr=args.pr)
        #print(f"marg_coverage = {marg_coverage}, {size}, {s_value}")
        ###############
        if marg_coverage - (1 - args.alpha) < 0.015 and marg_coverage >= 1 - args.alpha:
            break
        if marg_coverage >= 1 - args.alpha:
            lb_s = s_value
        else:
            ub_s = s_value
        s_value = (lb_s + ub_s) / 2
        # print(f"all2 = {size} {marg_coverage}, {size}, {lb_s}, {ub_s}")
    # exit(1)
    s_APS = s_value
    s_APS = 0.5  # Remove this line if we want to binary search over s value
    d_APS = 0.03
    for p in range(1, 2):
        # print(f"p APS= {p}")
        thresholds_ours[p, 0] = torch.quantile(
            torch.quantile(cal_scores[p, :, :], q=1 - alpha_tilde_star[p], dim=1),
            q=1 - s_APS * args.alpha + d_APS)

    return thresholds_ours, alpha_tilde_star


def find_threshold1(args, y_test, indices, scores_simple_cal_ball, scores_list, correction, n_s_test,
                   num_of_classes, idx11, idx12, channels=3, rows=32, cols=32, device='cpu', GPU_CAPACITY=1024):

    #print(f"scores_simple_cal_ball = {scores_simple_cal_ball.shape}")
    thresholds_ours = np.zeros((len(scores_list), 3))
    cal_scores = scores_simple_cal_ball[:, idx11, :, y_test[idx11]].transpose(0, 1)
    #print(f"cal_scores = {cal_scores.shape}")
    alpha_tilde_star = np.zeros(len(scores_list))
    #cal_scores_pd_APS = pd.DataFrame({'APS': cal_scores[1], 'Labels': y_test[idx11]})
    #cal_scores_pd_HPS = pd.DataFrame({'HPS': cal_scores[0], 'Labels': y_test[idx11]})
    #plot_path = ''

    #exit(1)

    s_APS = args.s_value
    d_APS = 0.1


    for p in range(2):
        thresholds_ours[p, 0] = torch.quantile(
            torch.quantile(cal_scores[p, :, :], q=1 - args.pr + d_APS, dim=1),
            q=1 - args.alpha + s_APS + 0.02)  
        alpha_tilde_star[p] = 1 - args.pr  

    return thresholds_ours, alpha_tilde_star


def find_threshold_plain(args, y_test, indices, scores_simple_cal_ball, scores_list, correction):
    s_value = 1

    thresholds_ours = np.zeros((len(scores_list), 3))

    idx1, idx2 = train_test_split(indices, test_size=0.5)
    idx11, idx12 = train_test_split(idx1, test_size=0.5)
    cal_scores1 = scores_simple_cal_ball[:, idx11, :, y_test[idx11]].transpose(0, 1)
    samples = torch.randint(0, cal_scores1.shape[2], (cal_scores1.shape[2],))
    # print(f"cal_scores shape = {cal_scores1.shape}, {samples.shape}")
    cal_scores = cal_scores1[:, :, samples][:, :, :int((1 - args.pr) * cal_scores1.shape[2])]
    # cal_scores, _ = torch.sort(cal_scores1, dim = 2)
    # cal_scores = cal_scores[:, :, int(args.pr*cal_scores1.shape[2]):]
    # print(f"cal_scores = {cal_scores.shape}")
    alpha_tilde_star = np.zeros(len(scores_list))
    for k in range(len(scores_list)):
        lb, ub = 0.0, 1.0
        alpha_tilde = 0.5

        while ub - lb > 0.01:
            quantiles = torch.quantile(cal_scores[k, :, :], q=1 - alpha_tilde, dim=1)
            quantiles_expand = quantiles.unsqueeze(dim=1).repeat(1, cal_scores.shape[2])
            # print(cal_scores.shape, 's', quantiles_expand.shape, 's2', quantiles.shape)
            num_satisfied = cal_scores[k, :, :] <= quantiles_expand
            frac = (1 / (cal_scores.shape[2] * len(idx11))) * torch.sum(cal_scores[k, :, :] <= quantiles_expand)
            # print(f"frac = {frac}, ub = {ub} lb = {lb}")
            # print(f"sg = {sg}, {k}")
            if frac >= 1 - args.alpha:
                lb = alpha_tilde
            else:
                ub = alpha_tilde
            if ub - lb <= 0.01:
                break
            alpha_tilde = (lb + ub) / 2
            # print(f"tilde = {alpha_tilde}, {lb}, {ub}")
        alpha_tilde_star[k] = alpha_tilde

    for p in range(len(scores_list)):
        thresholds_ours[p, 0] = torch.quantile(
            torch.quantile(cal_scores[p, :, :], q=1 - alpha_tilde_star[p], dim=1),
            q=1 - s_value * args.alpha)

    return thresholds_ours, alpha_tilde_star


def test_all_ImageNet(args, y_test_expand, scores_simple_cal, scores_simple_clean_test, model, scores_list, x_test, y_test, idx2, n_s_test, sigma_model,
                      thresholds_theirs, correction,
                      thresholds_ours, num_of_classes=1000, channels=3, rows=224, cols=224, device='cpu',
                      GPU_CAPACITY=1024):
    
    scores_simple_cal = scores_simple_cal.reshape(len(scores_list), len(y_test), n_s_test, num_of_classes)[:, idx2, :, :]
    scores_simple_cal = scores_simple_cal.reshape(len(scores_list), len(idx2)*n_s_test, num_of_classes)
    y_test_expand = y_test[idx2].unsqueeze(dim=1).repeat(1, n_s_test).reshape(-1, )
    y_test_expand = y_test_expand.numpy()
    predicted_adv_sets_base = prediction(scores_simple=scores_simple_cal,
                                         num_of_scores=len(scores_list), thresholds=thresholds_theirs,
                                         base=True)

    # generate robust prediction sets on the adversarial test set
    predicted_adv_sets = prediction(scores_smoothed=scores_simple_cal,
                                    smoothed_scores=scores_simple_cal,
                                    num_of_scores=len(scores_list), thresholds=thresholds_theirs,
                                    correction=correction, base=False)

    predicted_clean_sets_base = prediction(scores_simple=scores_simple_clean_test[:, idx2, :],
                                           num_of_scores=len(scores_list), thresholds=thresholds_ours, base=True)
    predicted_clean_sets = prediction(scores_simple=scores_simple_clean_test[:, idx2, :],
                                      num_of_scores=len(scores_list), thresholds=thresholds_ours, base=True)

    ################### ours #############
    predicted_adv_sets_base_ours = prediction(scores_simple=scores_simple_cal,
                                              num_of_scores=len(scores_list), thresholds=thresholds_ours,
                                              base=True)
    predicted_adv_sets_ours = prediction(scores_simple=scores_simple_cal,
                                         num_of_scores=len(scores_list), thresholds=thresholds_ours,
                                         ours=True)

    ################ ours ##############
    for p in range(len(scores_list)):
        predicted_adv_sets[p].insert(0, predicted_adv_sets_base[p])
        predicted_adv_sets_ours[p].insert(0, predicted_adv_sets_base_ours[p])
        predicted_clean_sets[p].insert(0, predicted_clean_sets_base[p])

        marg_coverage, size = evaluate_predictions_pr(predicted_adv_sets[p][3], None, y_test_expand,
                                                      n=len(idx2), m=n_s_test,
                                                      pr=args.pr)

        args.cvg_list_ro[p].append(marg_coverage)
        args.si_list_ro[p].append(size)
        del marg_coverage, size

        marg_coverage, size = evaluate_predictions_pr(predicted_adv_sets_ours[p][0], None, y_test_expand,
                                                      n=len(idx2), m=n_s_test,
                                                      pr=args.pr)
        # print(f"marg_coverage = {marg_coverage}, size = {size}")
        args.cvg_list_ours[p].append(marg_coverage)
        args.si_list_ours[p].append(size)
        del marg_coverage, size

        marg_coverage, size = evaluate_predictions_pr(predicted_adv_sets[p][0], None, y_test_expand,
                                                      n=len(idx2), m=n_s_test,
                                                      pr=args.pr)

        args.cvg_list_base0[p].append(marg_coverage)
        args.si_list_base0[p].append(size)

        _, marg_coverage, size = evaluate_predictions(predicted_clean_sets[p][0], None, y_test[idx2].numpy(),
                                                      conditional=False, coverage_on_label=False,
                                                      num_of_classes=num_of_classes)
        args.cvg_list_clean[p].append(marg_coverage)
        args.si_list_clean[p].append(size)
        del marg_coverage, size


def test_all(args, scores_simple_clean_test, model, scores_list, x_test, y_test, idx2, n_s_test, sigma_model,
             thresholds_theirs, correction,
             thresholds_ours, num_of_classes=10, channels=3, rows=224, cols=224, device='cpu',
             GPU_CAPACITY=1024):

    #print(f"thresholds_ours in test = {thresholds_ours} {args.pr}")
    x_test_sampled = x_test[idx2].unsqueeze(dim=1).repeat(1, n_s_test, 1, 1, 1).reshape(-1,
                                                                                        channels,
                                                                                        rows,
                                                                                        cols)
    y_test_expand = y_test[idx2].unsqueeze(dim=1).repeat(1, n_s_test).reshape(-1, )
    y_test_expand = y_test_expand.numpy()
    # epsilon11 = torch.FloatTensor(x_test_sampled.shape).uniform_(0, 1)
    # epsilon11 = torch.renorm(epsilon11, p=2, dim=0, maxnorm=args.delta_ours)
    sh1, sh2, sh3, sh4 = x_test[idx2].shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3]
    sh = [sh1, sh2, sh3, sh4]
    epsilon11 = distanced_sampling(sh, args.eval_delta, n_s_test, gap=2, uniform = True)
    n_test_expand = y_test.shape[0] * n_s_test
    indices_expand = torch.arange(n_test_expand)
    x_test_sampled = x_test_sampled + epsilon11
    del epsilon11
    s_temp = get_scores(model, x_test_sampled, indices_expand, n_s_test, sigma_model, num_of_classes,
                        scores_list, base=True, device=device, GPU_CAPACITY=GPU_CAPACITY)
    scores_simple_cal = torch.reshape(torch.from_numpy(s_temp),
                                      (len(scores_list), len(idx2), n_s_test,
                                       num_of_classes))
    scores_simple_cal = scores_simple_cal.reshape(len(scores_list),
                                                  len(idx2) * n_s_test,
                                                  num_of_classes).numpy()
    # print(f"scores_simple_cal = {scores_simple_cal.shape}")
    # generate prediction sets on the adversarial test set for base model
    # print(f"thresholds_theirs2 = {thresholds_theirs}")
    predicted_adv_sets_base = prediction(scores_simple=scores_simple_cal,
                                         num_of_scores=len(scores_list), thresholds=thresholds_theirs,
                                         base=True)

    # generate robust prediction sets on the adversarial test set
    predicted_adv_sets = prediction(scores_smoothed=scores_simple_cal,
                                    smoothed_scores=scores_simple_cal,
                                    num_of_scores=len(scores_list), thresholds=thresholds_theirs,
                                    correction=correction, base=False)

    predicted_clean_sets_base = prediction(scores_simple=scores_simple_clean_test[:, idx2, :],
                                           num_of_scores=len(scores_list), thresholds=thresholds_ours, base=True)
    predicted_clean_sets = prediction(scores_simple=scores_simple_clean_test[:, idx2, :],
                                      num_of_scores=len(scores_list), thresholds=thresholds_ours, base=True)

    ################### ours #############
    predicted_adv_sets_base_ours = prediction(scores_simple=scores_simple_cal,
                                              num_of_scores=len(scores_list), thresholds=thresholds_ours,
                                              base=True)
    predicted_adv_sets_ours = prediction(scores_simple=scores_simple_cal,
                                         num_of_scores=len(scores_list), thresholds=thresholds_ours,
                                         ours=True)

    ################ ours ##############
    for p in range(len(scores_list)):
        predicted_adv_sets[p].insert(0, predicted_adv_sets_base[p])
        predicted_adv_sets_ours[p].insert(0, predicted_adv_sets_base_ours[p])
        predicted_clean_sets[p].insert(0, predicted_clean_sets_base[p])

        marg_coverage, size = evaluate_predictions_no_pr(predicted_adv_sets[p][3], None, y_test_expand,
                                                      n=len(idx2), m=n_s_test,
                                                      pr=args.pr)

        args.cvg_list_ro[p].append(marg_coverage)
        args.si_list_ro[p].append(size)
        del marg_coverage, size

        marg_coverage, size = evaluate_predictions_no_pr(predicted_adv_sets_ours[p][0], None, y_test_expand,
                                                      n=len(idx2), m=n_s_test,
                                                      pr=args.pr)
        # print(f"marg_coverage = {marg_coverage}, size = {size}")
        args.cvg_list_ours[p].append(marg_coverage)
        args.si_list_ours[p].append(size)
        del marg_coverage, size

        marg_coverage, size = evaluate_predictions_no_pr(predicted_adv_sets[p][0], None, y_test_expand,
                                                      n=len(idx2), m=n_s_test,
                                                      pr=args.pr)

        args.cvg_list_base0[p].append(marg_coverage)
        args.si_list_base0[p].append(size)

        _, marg_coverage, size = evaluate_predictions(predicted_clean_sets[p][0], None, y_test[idx2].numpy(),
                                                      conditional=False, coverage_on_label=False,
                                                      num_of_classes=num_of_classes)
        args.cvg_list_clean[p].append(marg_coverage)
        args.si_list_clean[p].append(size)
        del marg_coverage, size


def test_all_rebuttel(args, scores_simple_clean_test, model, scores_list, x_test, y_test, idx2, n_s_test, sigma_model,
             thresholds_theirs, correction,
             thresholds_ours, num_of_classes=10, channels=3, rows=224, cols=224, device='cpu',
             GPU_CAPACITY=1024):
    x_test_sampled = x_test[idx2].unsqueeze(dim=1).repeat(1, n_s_test, 1, 1, 1).reshape(-1,
                                                                                        channels,
                                                                                        rows,
                                                                                        cols)
    y_test_expand = y_test[idx2].unsqueeze(dim=1).repeat(1, n_s_test).reshape(-1, )
    y_test_expand = y_test_expand.numpy()
    # epsilon11 = torch.FloatTensor(x_test_sampled.shape).uniform_(0, 1)
    # epsilon11 = torch.renorm(epsilon11, p=2, dim=0, maxnorm=args.delta_ours)
    sh1, sh2, sh3, sh4 = x_test[idx2].shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3]
    sh = [sh1, sh2, sh3, sh4]
    epsilon11 = distanced_sampling(sh, args.delta_ours, n_s_test, gap=2)
    n_test_expand = y_test.shape[0] * n_s_test
    indices_expand = torch.arange(n_test_expand)
    x_test_sampled = x_test_sampled + epsilon11
    del epsilon11
    s_temp = get_scores(model, x_test_sampled, indices_expand, n_s_test, sigma_model, num_of_classes,
                        scores_list, base=True, device=device, GPU_CAPACITY=GPU_CAPACITY)
    scores_simple_cal = torch.reshape(torch.from_numpy(s_temp),
                                      (len(scores_list), len(idx2), n_s_test,
                                       num_of_classes))
    scores_simple_cal = scores_simple_cal.reshape(len(scores_list),
                                                  len(idx2) * n_s_test,
                                                  num_of_classes).numpy()
    # print(f"scores_simple_cal = {scores_simple_cal.shape}")
    # generate prediction sets on the adversarial test set for base model
    # print(f"thresholds_theirs2 = {thresholds_theirs}")
    predicted_adv_sets_base = prediction(scores_simple=scores_simple_cal,
                                         num_of_scores=len(scores_list), thresholds=thresholds_theirs,
                                         base=True)

    # generate robust prediction sets on the adversarial test set
    predicted_adv_sets = prediction(scores_smoothed=scores_simple_cal,
                                    smoothed_scores=scores_simple_cal,
                                    num_of_scores=len(scores_list), thresholds=thresholds_theirs,
                                    correction=correction, base=False)

    predicted_clean_sets_base = prediction(scores_simple=scores_simple_clean_test[:, idx2, :],
                                           num_of_scores=len(scores_list), thresholds=thresholds_ours, base=True)
    predicted_clean_sets = prediction(scores_simple=scores_simple_clean_test[:, idx2, :],
                                      num_of_scores=len(scores_list), thresholds=thresholds_ours, base=True)

    ################### ours #############
    predicted_adv_sets_base_ours = prediction(scores_simple=scores_simple_cal,
                                              num_of_scores=len(scores_list), thresholds=thresholds_ours,
                                              base=True)
    predicted_adv_sets_ours = prediction(scores_simple=scores_simple_cal,
                                         num_of_scores=len(scores_list), thresholds=thresholds_ours,
                                         ours=True)

    ################ ours ##############
    for p in range(len(scores_list)):
        predicted_adv_sets[p].insert(0, predicted_adv_sets_base[p])
        predicted_adv_sets_ours[p].insert(0, predicted_adv_sets_base_ours[p])
        predicted_clean_sets[p].insert(0, predicted_clean_sets_base[p])

        marg_coverage, size = evaluate_predictions_no_pr(predicted_adv_sets[p][3], None, y_test_expand,
                                                      n=len(idx2), m=n_s_test,
                                                      pr=args.pr)

        args.cvg_list_ro[p].append(marg_coverage)
        args.si_list_ro[p].append(size)
        del marg_coverage, size

        marg_coverage, size = evaluate_predictions_no_pr(predicted_adv_sets_ours[p][0], None, y_test_expand,
                                                      n=len(idx2), m=n_s_test,
                                                      pr=args.pr)
        # print(f"marg_coverage = {marg_coverage}, size = {size}")
        args.cvg_list_ours[p].append(marg_coverage)
        args.si_list_ours[p].append(size)
        del marg_coverage, size

        marg_coverage, size = evaluate_predictions_no_pr(predicted_adv_sets[p][0], None, y_test_expand,
                                                      n=len(idx2), m=n_s_test,
                                                      pr=args.pr)

        args.cvg_list_base0[p].append(marg_coverage)
        args.si_list_base0[p].append(size)

        _, marg_coverage, size = evaluate_predictions(predicted_clean_sets[p][0], None, y_test[idx2].numpy(),
                                                      conditional=False, coverage_on_label=False,
                                                      num_of_classes=num_of_classes)
        args.cvg_list_clean[p].append(marg_coverage)
        args.si_list_clean[p].append(size)
        del marg_coverage, size


def make_stats(args, path, acc, alphas):
    stat_list = pd.DataFrame({
        'ours_APS_size': args.si_list_ours[1],
        'ours_APS_cvg': args.cvg_list_ours[1],
        'RSCP_APS_size': args.si_list_ro[1],
        'RSCP_APS_cvg': args.cvg_list_ro[1],

        'ours_HPS_size': args.si_list_ours[0],
        'ours_HPS_cvg': args.cvg_list_ours[0],
        'RSCP_HPS_size': args.si_list_ro[0],
        'RSCP_HPS_cvg': args.cvg_list_ro[0],

        'vanilla_APS_size': args.si_list_base0[1],
        'vanilla_APS_cvg': args.cvg_list_base0[1],
        'vanilla_HPS_size': args.si_list_base0[0],
        'vanilla_HPS_cvg': args.cvg_list_base0[0],

        'alphas_APS': alphas[0],
        'alphas_HPS': alphas[1],

        'clean_test_HPS_cvr': args.cvg_list_clean[0],
        'clean_test_HPS_size': args.si_list_clean[0],
        'clean_test_APS_cvr': args.cvg_list_clean[1],
        'clean_test_APS_size': args.si_list_clean[1],

    })
    torch.save(stat_list, path + '/stats.pkl')


