

import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import math
import matplotlib as mpl
import seaborn as sns
import scipy

from copy import deepcopy
from collections import defaultdict
from matplotlib.colors import Normalize, LogNorm
from pyts.metrics import dtw
# from fastdtw import fastdtw
from tqdm import tqdm
from pyts.metrics.dtw import accumulated_cost_matrix, _check_input_dtw, _compute_region, _input_to_cost, _return_path

def F_partial(L, omega):
    from scipy.stats import norm
    Q_inv = 1.959964
    sigma = omega/(2*Q_inv)
    tdomain = np.arange(L, dtype=float) * L/(L-1)
    tdomain -= tdomain[int(L/2)]
    tdomain /= sigma
    kernel = norm.pdf(tdomain)/float(norm.pdf(0.0))
    return kernel

def fastsdtw(pred, true, centers, period):
    # compute probability distribution
    from scipy.stats import norm
    L = true.shape[0]
    normal = norm.pdf(L/(period/16)*(np.arange(L)/(L-1)-0.5))
    print(L/(period/16)*(np.arange(L)/(L-1)-0.5), period/16)
    assert False
    base = np.zeros(L, dtype=float)
    base[centers] = 1.0
    mix_normal1 = np.convolve(base, normal, mode='same')
    mix_normal1 = mix_normal1/np.sum(mix_normal1)
    mix_normal2 = 1-(mix_normal1/max(mix_normal1))
    mix_normal2 = mix_normal2/np.sum(mix_normal2)
    #mix_normal2 = np.convolve(base, 1-normal, mode='same')/len(centers)
    sdtw_needle, cost_mat_needle = fastsdtw_helper(pred, true, centers, period, mix_normal1, axis=1)
    sdtw_haystack, cost_mat_haystack = fastsdtw_helper(pred, true, centers, period, mix_normal2, axis=0)
    # out = sdtw_needle/len(true)
    out = 0.5 * (sdtw_needle + sdtw_haystack)
    return out, cost_mat_needle, cost_mat_haystack

def fastsdtw_helper(pred, true, centers, period, mix_normal,
        dist='square', method='classic', options=None, axis=1
    ):
    # compute cost matrix
    precomputed_cost = None
    if options is None:
        options = dict()
    x, y, precomputed_cost, n_timestamps_1, n_timestamps_2 = \
        _check_input_dtw(pred, true, precomputed_cost, dist, method)
    region = _compute_region(n_timestamps_1, n_timestamps_2, method, dist, x=x,
                             y=y, **options)
    cost_mat = _input_to_cost(x, y, dist, precomputed_cost, region)

    # regularize cost matrix
    # print(cost_mat.shape, mix_normal.shape)'
    mix_normal = mix_normal
    if axis == 0:
        mix_normal_mat = mix_normal.reshape(-1,1)
    else:
        assert axis == 1
        mix_normal_mat = mix_normal.reshape(1,-1) #/mix_normal.reshape(-1,1)
    # plt.imshow(np.log(mix_normal_mat))#np.clip(mix_normal_mat, a_min=0.0, a_max=1.0))
    # plt.colorbar()
    # plt.savefig('mix_normal_mat.png')
    # plt.clf()

    print(mix_normal)
    cost_mat = cost_mat * mix_normal_mat #+ time_penalization
    # run DTW
    acc_cost_mat = accumulated_cost_matrix(cost_mat, region=region)
    dtw_dist = acc_cost_mat[-1, -1]
    return dtw_dist, cost_mat


def plot_SDTW():
    haystack = 0.035*np.sin(np.arange(200)/128*np.pi*3+2) #+ np.arange(200)/800
    needle = F_partial(L=132, omega=6)

    true_needle_indices = [32,132]

    base1 = np.zeros(200)
    base2 = np.zeros(200)
    base3 = np.zeros(200)
    base4 = np.zeros(200)

    base1[true_needle_indices] = 1.0
    base2[[36,125]] = 1.0
    base3[[16,100]] = 1.0
    base4[[34,74,128]] = 1.0

    needle_true = np.convolve(base1, needle, mode='same')
    needle_pred0 = np.convolve(base2, needle, mode='same')
    needle_pred1 = np.convolve(base3, needle, mode='same')
    needle_pred2 = np.convolve(base4, needle, mode='same')

    true = needle_true + haystack + 0.00*np.random.randn(*haystack.shape)
    pred_dict = {
        'pred0': needle_pred0 + haystack + 0.02*np.random.randn(*haystack.shape),
        'pred1': needle_pred1 + haystack + 0.005*np.random.randn(*haystack.shape),
        'pred2': needle_pred2 + haystack + 0.005*np.random.randn(*haystack.shape),
        'pred3': haystack + 0.005*np.random.randn(*haystack.shape),
    }
    color_dict = {
        'pred0': 'red',
        'pred1': 'violet',
        'pred2': 'green',
        'pred3': 'blue',
    }

    # compute the metrics
    metric2result = defaultdict(dict)
    metric_li = ['mae', 'mse', 'dtw', 'sdtw']
    pred_li = sorted(list(pred_dict.keys()))
    for metric in metric_li:
        for ts_key, pred in pred_dict.items():
            if metric == 'mae':
                val = np.mean(np.abs(pred-true))
            elif metric == 'mse':
                val = np.mean(np.abs(pred-true)**2)
            elif metric == 'dtw':
                val = dtw(pred, true)/len(true)
            elif metric == 'sdtw':
                val = fastsdtw(pred, true, torch.tensor(true_needle_indices), 100)[0]
            else:
                assert False
            metric2result[metric][ts_key] = val

    # compute the alignment
    for demo_key in ['pred0','pred2']:
        result, cost_mat_needle, cost_mat_haystack = fastsdtw(pred_dict[demo_key], true, torch.tensor(true_needle_indices), 100)
        a_min = 1e-12
        a_max = 1e6
        cost_mat_needle = np.clip(cost_mat_needle, a_min=a_min, a_max=a_max)
        path_needle = _return_path(cost_mat_needle)
        # path_needle[0] += 1
        path_needle = np.clip(path_needle, a_min=1, a_max=cost_mat_haystack.shape[0]-2)
        path_mat_needle = np.zeros_like(cost_mat_needle)
        path_mat_needle[path_needle[0], path_needle[1]] = 1.0
        path_mat_needle[path_needle[0], path_needle[1]+1] = 1.0
        path_mat_needle[path_needle[0], path_needle[1]-1] = 1.0
        path_mat_needle[path_needle[0]+1, path_needle[1]] = 1.0
        path_mat_needle[path_needle[0]-1, path_needle[1]] = 1.0
        path_mat_needle = np.ma.masked_where(path_mat_needle < 0.9, path_mat_needle)

        cost_mat_haystack = np.clip(cost_mat_haystack, a_min=a_min, a_max=a_max)
        path_haystack = _return_path(cost_mat_haystack)
        # path_haystack[0] += 1
        path_haystack = np.clip(path_haystack, a_min=1, a_max=cost_mat_haystack.shape[0]-2)
        path_mat_haystack = np.zeros_like(cost_mat_haystack)
        path_mat_haystack[path_haystack[0], path_haystack[1]] = 1.0
        path_mat_haystack[path_haystack[0], path_haystack[1]+1] = 1.0
        path_mat_haystack[path_haystack[0], path_haystack[1]-1] = 1.0
        path_mat_haystack[path_haystack[0]+1, path_haystack[1]] = 1.0
        path_mat_haystack[path_haystack[0]-1, path_haystack[1]] = 1.0
        path_mat_haystack = np.ma.masked_where(path_mat_haystack < 0.9, path_mat_haystack)

        # plot the imshow
        plt.subplot(2,2,2)
        # fig, ax = plt.subplots(figsize=(3,3))
        plt.imshow(cost_mat_needle, cmap ='gnuplot', norm=Normalize(), interpolation='none')
        plt.colorbar()
        plt.imshow(path_mat_needle, cmap ='binary', interpolation='none', alpha=0.8)
        # plt.savefig('asdf3.png')
        # plt.clf()

        plt.subplot(2,2,4)
        # fig, ax = plt.subplots(figsize=(3,3))
        plt.imshow(cost_mat_haystack, cmap ='gnuplot', norm=Normalize(), interpolation='none')
        plt.colorbar()
        plt.imshow(path_mat_haystack, cmap ='binary', interpolation='none', alpha=0.8)
        # plt.savefig('asdf4.png')
        # plt.clf()

        plt.subplot(2,2,1)
        true_plot = true - 1.5
        # plt.figure(figsize=(7,3))
        #plot alignment
        # norm = Normalize(vmin=0, vmax=27.607858015302238) #min(cost_mat_needle[path[0], path[1]]), vmax=max(cost_mat_needle[path[0], path[1]]))
        # print(min(cost_mat_needle[path[0], path[1]]), max(cost_mat_needle[path[0], path[1]]))
        cmap = mpl.cm.get_cmap('gnuplot')
        for i, j in zip(path_needle[0][::5], path_needle[1][::5]):
            plt.plot((i, j), (pred_dict[demo_key][i], true_plot[j]), alpha=1.0, color='gray')#norm(cost_mat_needle[i,j]), color='blue') #, color=cmap(norm(cost_mat_needle[i,j]))
        plt.plot(pred_dict[demo_key], label=demo_key, alpha=0.7, color=color_dict[demo_key])
        plt.plot(true_plot, label='true', alpha=1.0, linestyle='--', color='black')
        plt.yticks([])
        plt.legend()
        # plt.savefig('asdf5.png')
        # plt.clf()

        plt.subplot(2,2,3)
        # plt.figure(figsize=(7,3))
        #plot alignment
        # norm = Normalize(vmin=0, vmax=27.607858015302238) #min(cost_mat_needle[path[0], path[1]]), vmax=max(cost_mat_needle[path[0], path[1]]))
        # print(min(cost_mat_needle[path[0], path[1]]), max(cost_mat_needle[path[0], path[1]]))
        cmap = mpl.cm.get_cmap('gnuplot')
        for i, j in zip(path_haystack[0][::5], path_haystack[1][::5]):
            plt.plot((i, j), (pred_dict[demo_key][i], true_plot[j]), alpha=1.0, color='gray')#norm(cost_mat_needle[i,j]), color='blue') #, color=cmap(norm(cost_mat_needle[i,j]))
        plt.plot(pred_dict[demo_key], label=demo_key, alpha=0.7, color=color_dict[demo_key])
        plt.plot(true_plot, label='true', alpha=1.0, linestyle='--', color='black')
        plt.yticks([])
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'SDTW_{demo_key}.png')
        plt.clf()


    # plot the time series
    plt.figure(figsize=(5,2))
    plt.plot(true, label='true', alpha=1.0, linestyle='--', color='black')
    for ts_key, pred in pred_dict.items():
        plt.plot(pred, label=ts_key, alpha=0.9, color=color_dict[ts_key])
    # plt.colorbar()
    plt.legend()
    plt.tight_layout()
    plt.savefig('SDTW_time_series.png')
    plt.clf()

    # plot the evaluation metrics
    offset = 0.35  # the width of the bars
    width = (offset*2)/len(pred_li)

    x = np.arange(len(metric_li))-offset  # the label locations
    fig, ax = plt.subplots(figsize=(5,2))
    rects_li = []
    for i, ts_key in enumerate(pred_li):
        y_li = [metric2result[metric][ts_key] for metric in metric_li]
        rects_li.append(ax.bar(x+i*width, y_li, width, label=ts_key, color=color_dict[ts_key]))
    ax.set_yscale('log')
    ax.set_ylabel('Scores')
    ax.set_title('Evaluation Metrics')
    ax.set_xticks(x+offset)
    ax.set_xticklabels([x.upper() for x in metric_li])
    # ax.legend()
    
    fig.tight_layout()
    plt.savefig('SDTW_metrics.png')
    plt.clf()


if __name__ == '__main__':
    plot_SDTW()