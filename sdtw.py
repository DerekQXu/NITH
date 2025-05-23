import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle
import os

from collections import defaultdict
from scipy.signal import find_peaks
from sklearn.metrics import auc
from scipy.stats import norm
from pyts.metrics.dtw import dtw
from tqdm import tqdm

eps = 1e-10

def generate_needle(scale, resolution, spike_name=None, 
        needle_shape_li=('gaussian', 'exp', 'triangle'),
        needle_symmetry_li=('both', 'left', 'right')):
    if spike_name is None:
        shp, = random.sample(list(needle_shape_li), k=1) # 
        sym, = random.sample(list(needle_symmetry_li), k=1)
        spike_name = f'{shp}_{sym}'

    shp, sym = spike_name.split('_')
    center = [1.0]

    assert resolution % 2 == 0
    x = np.arange(int(resolution/2)) + 1
    if shp == 'gaussian':
        GAUSSIAN_CUTOFF = 2.14596
        tail = scipy.stats.norm.pdf(x/scale*GAUSSIAN_CUTOFF)/scipy.stats.norm.pdf(0.0)
    elif shp == 'exp':
        tail = np.exp(x*np.log(0.01)/(scale+eps))
    elif shp == 'triangle':
        tail = np.clip(-(0.9/scale+eps) * x + 1, a_min=0.0, a_max=100.0)
    else:
        assert False
    # tail = np.clip(-(1/scale+eps) * (x+1) + 1, a_min=0.0, a_max=100.0)

    tail = list(tail)

    if sym == 'both':
        signal = tail[::-1] + [1.0] + tail[:-1]
    elif sym == 'left':
        signal = tail[::-1] + [1.0] + [0]*(len(tail)-1)
    elif sym == 'right':
        signal = [0]*len(tail) + [1.0] + tail[:-1]
    else:
        assert False

    return np.array(signal)

def plot_grouped_bar_chart(data_dict, vis_dict):
    """
    Plots a grouped bar chart from a dictionary of dictionaries.

    Parameters:
        data_dict (dict): A dictionary where keys are methods, and values are dictionaries 
                          containing data labels and their respective values.

    Example input:
        {
            'method1': {'data1': 10, 'data2': 20, 'data3': 15},
            'method2': {'data1': 5, 'data2': 25, 'data3': 10},
        }
    """
    methods = list(data_dict.keys())
    data_labels = list(next(iter(data_dict.values())).keys())  # Get the data labels from the first method
    num_methods = len(methods)
    num_data_labels = len(data_labels)
    
    # Define the positions of the bars
    bar_width = 0.8 / num_methods  # Adjust bar width for grouped layout
    x_indices = np.arange(num_data_labels)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(5, 2))
    
    for i, method in enumerate(methods):
        values = [data_dict[method][label] for label in data_labels]
        bar_positions = x_indices + i * bar_width
        ax.bar(bar_positions, values, bar_width, label=method, color=vis_dict[method][0])
    
    # Set the labels, title, and legend
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Values")
    # ax.set_title("Time Series")
    ax.set_xticks(x_indices + bar_width * (num_methods - 1) / 2)
    ax.set_xticklabels(data_labels)
    ax.grid()
    # ax.legend(title="Time Series", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

# from .evaluation import get_peaks, get_anomaly_score, get_period

def get_peaks(ts, gamma=1, width_min=1, width_max=16):
    ts = np.array(ts)
    peaks, properties = find_peaks((ts-ts.mean())/(ts.std()+1e-12), prominence=2.0, width=(width_min, width_max))
    widths = list(properties['widths']) #np.array(properties['prominences']).mean()

    starts, ends = [], []
    P = set()
    for peak, width in zip(peaks, widths):
        scaled_width = (gamma*width)//2
        start = max(0, peak-scaled_width)
        end = min(peak+scaled_width+1, len(ts))
        P.update(set(np.arange(start, end)))
        starts.append(start)
        ends.append(end)
    starts, ends = np.array(starts), np.array(ends)
    return P, len(peaks), peaks, starts, ends, widths

def get_period(ts):
    t = np.arange(ts.shape[0])
    fft_result = np.fft.fft(ts)
    frequencies = np.fft.fftfreq(len(t), d=(t[1] - t[0]))
    magnitude = np.abs(fft_result)
    positive_frequencies = frequencies[frequencies > 0]
    positive_magnitude = magnitude[frequencies > 0] 
    dominant_frequency = positive_frequencies[np.argmax(positive_magnitude)]
    period = 1 / dominant_frequency
    return period

def get_anomaly_score(indices, stdev, ts_len):
    x = np.arange(ts_len)
    if len(indices) == 0:
        p = np.ones(ts_len)/2
    else:
        p = np.zeros(ts_len)
        for idx in indices:
            p += norm.pdf(x, loc=idx, scale=stdev)
        p /= (max(p) + 1e-12)
    return p

def get_label_soft(starts, ends, ts_len, l):
    label = np.zeros(ts_len)
    label_soft = np.zeros(ts_len)
    if int(l/2) == 0:
        tails = None
    else:
        tails = (1-(np.arange(int(l/2))+1)/l)**0.5
    for start, end in zip(starts, ends):
        label[start:end] = 1
        label_soft[start:end] = 1
        if tails is not None:
            tails_len = min(len(tails), start)
            if tails_len > 0: 
                label_soft[start-tails_len:start] = (tails[:tails_len])[::-1]
            tails_len = min(len(tails), len(label_soft) - end)
            if tails_len > 0: 
                label_soft[end:end+tails_len] = tails[:tails_len]
    return label, label_soft

def get_VUC(ts_pred, ts_true, needle_indices_forecast=None):
    if needle_indices_forecast is None:
        _, _, needle_indices_forecast, *_  = get_peaks(ts_true)
    period = get_period(ts_true)
    # _, _, _, starts, ends, *_ = get_peaks(ts_true)
    starts = np.clip(needle_indices_forecast-8, a_min=0, a_max=len(ts_true))
    ends = np.clip(needle_indices_forecast+8, a_min=0, a_max=len(ts_true))
    _, _, peaks, *_ = get_peaks(ts_pred)
    anomaly_score = get_anomaly_score(peaks, period//2, len(ts_pred))

    # compute VUC
    RAUC_li = []
    for l in (period*np.arange(6)/5):
        label, label_soft = get_label_soft(starts, ends, len(ts_pred), l)
        TPR_li = []
        FPR_li = []
        for t in (np.arange(6)/5):
            pred = (anomaly_score >= t).astype(np.int32)
            Pl = sum(label_soft+label)/2 
            Nl = len(label_soft) - Pl
            TPl = sum(label_soft*pred)
            FPl = sum((1-label_soft)*pred)
            TPR = np.clip(TPl/(Pl+1e-12)*sum(label*pred)/(sum(label)+1e-12), a_min=0, a_max=1)
            FPR = np.clip(FPl/(Nl+1e-12), a_min=0, a_max=1)
            TPR_li.append(TPR)
            FPR_li.append(FPR)
        TPR_li = np.array(TPR_li)
        FPR_li = np.array(FPR_li)
        RAUC_li.append(auc(FPR_li, TPR_li))
    return np.array(RAUC_li).mean()

def get_RF1(ts_pred, ts_true, beta=1.0):
    # https://www.vldb.org/pvldb/vol15/p2774-paparrizos.pdf
    P, NP, *_ = get_peaks(ts_pred)
    R, NR, *_ = get_peaks(ts_true)
    PnR = len(P.intersection(R))
    RP = PnR/(len(P)+1e-12)/(NP+1e-12)
    RR = PnR/(len(R)+1e-12)/(NR+1e-12)
    RF1 = (1+beta)**2 * RP * RR / (beta**2 * RP + RR + 1e-12)
    return RF1

def mae(x, y):
    return np.mean(np.abs(x-y))

def mse(x, y):
    return np.mean((x-y)**2)

def get_DTW(x, y, window=None):
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    out = get_DTW_partial(x, y, window=window)
    return out

def get_DTW_partial(x, y, window=None):
    assert len(y.shape) == 1
    if window is None:
        out = dtw(x,y, dist='absolute')/len(y)
    else:
        out = dtw(x, y, dist='absolute', method='sakoechiba', options={'window_size':window})/len(y)
    return out

def sdtw(x, y, eps=1e-1):
    k = np.log(eps)/len(x)
    N = len(x)
    out = 0.0
    w_sum = 0.0

    K = 20 # number of samples to take
    resolution = len(x)
    assert K < resolution-1
    values = list(range(N))[1:]
    step = (len(values) - 1) / (K - 1)
    for i in [0] + [values[round(i * step)] for i in range(K)]: # TODO
        w = eps**(i/(N-1))
        out += w * get_DTW(x, y, window=i)
        w_sum += w
    out /= w_sum
    return out

def get_metric_dict():
    metric_dict = {
        'MAE↓': mae,
        # 'MSE↓': mse, 
        'VUC↑': get_VUC,
        'DTW↓': get_DTW,
        # 'RF1↑': get_RF1,
        'SDTW↓': sdtw,
    }
    return metric_dict

def main_metric_demo():
    resolution = 512
    haystack = np.sin(np.arange(resolution)/256) * 0.1
    haystack_noisy = lambda: haystack + np.random.normal(loc=0.0, scale=0.01, size=resolution)

    needle_sharp = generate_needle(10, resolution, spike_name='exp_both')
    needle_dull = generate_needle(10, resolution, spike_name='exp_both')*2 #generate_needle(32, resolution, spike_name='gaussian_both')*0.2

    ts_locations_li = [
        np.array([62, 118, 255]),
        np.array([64, 128, 260]),
        np.array([70, 110, 250]),
        np.array([60, 258]),
        # np.array([65, 110, 252]),
        np.array([50, 90, 240]),
    ]
    ts_spike_train = []
    scaledown = 0.7
    for i, ts_locations in enumerate(ts_locations_li):
        ts = np.zeros(resolution)
        if i == 1:
            ts[ts_locations] = np.array([1.0, 1.0, 1.0]) * scaledown # 0.98, 1.52, 2.85]) * scaledown
        elif i == 3:
            ts[ts_locations] = np.array([0.5, 0.5]) * scaledown
        elif i == 4:
            # ts[ts_locations] = np.array([1.7, 1.7, 1.7]) * scaledown
            ts[ts_locations] = np.array([1.0, 1.5, 3.0]) * scaledown
        else:
            ts[ts_locations] = np.array([1.0, 1.5, 3.0]) * scaledown
        ts_spike_train.append(ts)

    ts_dict = {
        'gt': haystack_noisy() + scipy.signal.convolve(needle_sharp, ts_spike_train[0], mode='same'),
        'pred0': haystack_noisy() + scipy.signal.convolve(needle_sharp, ts_spike_train[2], mode='same'), # good
        'pred1': haystack_noisy() + scipy.signal.convolve(needle_sharp, ts_spike_train[4], mode='same'), # no amplitude
        'pred2': haystack + scipy.signal.convolve(needle_dull, ts_spike_train[1], mode='same'), # no noise
        'pred3': haystack_noisy() + scipy.signal.convolve(needle_sharp, ts_spike_train[3], mode='same'), # bad
    }
    for k in ts_dict:
        ts_dict[k] *= 8.1
    vis_dict = {
        'gt': ('black', 'dashed', 1.0),
        'pred0': ('red', 'solid', 0.75),
        'pred1': ('magenta', 'solid', 0.75),
        'pred2': ('green', 'solid', 0.75),
        'pred3': ('blue', 'solid', 0.75),
    }

    plt.figure(figsize=(5,2))
    for nm, ts in ts_dict.items():
        color, linestyle, alpha = vis_dict[nm]
        plt.plot(ts, label=nm, color=color, linestyle=linestyle, alpha=alpha)
        plt.legend(loc='right')
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.grid()
    plt.tight_layout()
    plt.savefig('sdtw_time_series.png')

    metric_dict = get_metric_dict()
    # TODO: plot results
    plot_dict = defaultdict(dict)
    for metric_nm, metric in metric_dict.items():
        ts_nm_li = []
        out_li = []
        for ts_nm, ts in ts_dict.items():
            if ts_nm != 'gt':
                ts_nm_li.append(ts_nm)
                out_li.append(metric(ts, ts_dict['gt']))
        out_li = np.array(out_li)
        for ts_nm, out in zip(ts_nm_li, out_li):
            plot_dict[ts_nm][metric_nm] = out #(out - np.mean(out_li)) / np.std(out_li)
    plot_grouped_bar_chart(plot_dict, vis_dict)
    plt.tight_layout()
    plt.savefig('sdtw_metrics.png')


if __name__ == '__main__':
    main_metric_demo()
