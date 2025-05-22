import numpy as np
import matplotlib.pyplot as plt

import time
import scipy
import random
import os
import yaml

from tqdm import tqdm
from gluonts.dataset.arrow import ArrowWriter, ArrowFile
from typing import List, Optional, Union
from pathlib import Path

PATH_TO_GAUSSIAN_PROCESSES = "/path/to/kernelsynth.arrow"
SPLIT = "pretrain"
PATH_TO_OUTPUT_BENCHMARK = ""
eps = 1e-12

class SampleDataset():
    def create_indices(self):
        print(f'creating indices for {self.id} dataset...')
        self.ts_len_li = np.array([len(ts) for ts in tqdm(self.ts_li)])
        indices = np.argsort(self.ts_len_li)
        self.ts_len_li = self.ts_len_li[indices]
        self.ts_li = [self.ts_li[i] for i in indices]
        self.max_len = max(self.ts_len_li)
        self.min_size2min_idx = {}
        print('indices created')

    def sample(self, min_size=512):
        # assert min_size <= max(self.ts_len_li)

        # caching
        if min_size in self.min_size2min_idx:
            min_idx = self.min_size2min_idx[min_size]
        else:
            min_idx = np.argmax(self.ts_len_li >= min_size)
            self.min_size2min_idx[min_size] = min_idx
        sample_idx = int((len(self.ts_len_li) - min_idx)*random.random() + min_idx)
        return self.ts_li[sample_idx]

class MergeDataset(SampleDataset):
    def __init__(self, dataset_li, id=None):
        self.ts_li = []
        for dataset in dataset_li:
            self.ts_li.extend(dataset.ts_li)
        self.id = id
    
    def load(self):
        self.create_indices()

class ArrowDataset(SampleDataset):
    def __init__(self, pathname, id=None):
        self.pathname = pathname
        self.id = id
    
    def load(self):
        print(f'loading {self.id} dataset...')
        ds = ArrowFile(self.pathname)
        ts_li = [ts['target'] for ts in tqdm(ds)]
        print('dataset loaded')
        self.ts_li = ts_li

def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    start_times: Optional[Union[List[np.datetime64], np.ndarray]] = None,
    compression: str = "lz4",
):
    if start_times is None:
        # Set an arbitrary start time
        start_times = [np.datetime64("2000-01-01 00:00", "s")] * len(time_series)

    assert len(time_series) == len(start_times)

    dataset = [
        {"start": start, "target": ts} for ts, start in zip(time_series, start_times)
    ]
    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=path,
    )

def normalize(ts):
    ts = ts - np.median(ts)
    ts = ts/max(ts.std(), 1e-5)
    return ts

def sample_wrapperv2(dataset_config, seq_len=None, min_size=None):
    if seq_len is None and min_size is not None:
        candidate_datasets = [dataset for dataset in dataset_config.values() if dataset.max_len >= min_size]
        dataset, = random.sample(candidate_datasets, k=1)
        ts_out = dataset.sample(min_size=min_size)
    elif seq_len is not None and min_size is None:
        candidate_datasets = [dataset for dataset in dataset_config.values() if dataset.max_len >= seq_len]
        dataset, = random.sample(candidate_datasets, k=1)
        ts = dataset.sample(min_size=seq_len)
        start_idx = int(random.random()*(len(ts)-seq_len))
        t0 = time.time()
        ts_out = ts[start_idx: start_idx+seq_len]
        ts_out = normalize(ts_out)
        assert len(ts_out) == seq_len
    else:
        assert False
    return ts_out

def arange_bounded(LB, UB, resolution):
    return (UB-LB)/(resolution-1) * np.arange(resolution) + LB

def rescale(LB, UB, signal):
    signal_min, signal_max = min(signal), max(signal)
    signal_0_1 = (signal - signal_min) / (signal_max - signal_min)
    return (UB - LB) * signal_0_1 + LB

def generate_resampling_functions(num_samples, resolution, cutoff, filter_bank=True):
    resolution = 1024
    x = arange_bounded(0, 1, resolution)

    ####################################################
    # Generate basis of 91 curves
    ####################################################
    print('generating basis...')
    # 30 linear curves
    basis = [x] * 30

    # 15 logarithmic curves
    if filter_bank:
        for n in arange_bounded(1.2, np.exp(1), 15):
            basis.append(rescale(0, 1, n**x))

        # 15 exponential curves
        for n in arange_bounded(1.2, np.exp(1), 15):
            basis.append(rescale(0, 1, np.log(x+1)/np.log(n)))

        # 30 polynomial curves
        for n in arange_bounded(1, 2, 15):
            basis.extend([x**n, x**(1/n)])

        # 30 staircase curves
        # https://math.stackexchange.com/questions/1671132/equation-for-a-smooth-staircase-function
        for n_steps in arange_bounded(2, 5, 5):
            x_scaled = 2 * np.pi * n_steps * x
            for bias in arange_bounded(0, 1/n_steps, 4)[:-1]:
                y = x_scaled + bias - np.sin((x_scaled+bias))
                basis.append(rescale(0, 1, y))
        for n_steps in arange_bounded(2, 5, 5):
            x_scaled = 2 * np.pi * n_steps * x
            for bias in arange_bounded(0, 1/n_steps, 4)[:-1]:
                y1 = x_scaled + bias - np.sin((x_scaled+bias))
                y2 = y1 - np.sin(y1)
                basis.append(rescale(0, 1, y2))
        print(f'basis of {len(basis)} functions generated.')

    ####################################################
    # Mixup using Combinations
    ####################################################
    print('mixing it up...')
    out = []
    for i in tqdm(range(num_samples)):
        s1, *s_other = random.sample(basis, 3)
        signal = s1
        for s in s_other:
            signal = np.interp(signal, x, s)
        out.append(signal)

    print('mixup done.')

    ####################################################
    # Find Locations
    ####################################################
    print('computing cutoffs...')
    golden_indices = []
    for i, arr in tqdm(enumerate(out)):
        golden_indices.append(np.searchsorted(arr, cutoff))
    golden_indices = np.array(golden_indices)

    return out, golden_indices

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
        tail = scipy.stats.norm.pdf(x/(scale+eps)*GAUSSIAN_CUTOFF)/scipy.stats.norm.pdf(0.0)
    elif shp == 'exp':
        tail = np.exp(x*np.log(0.01)/(scale+eps))
    elif shp == 'triangle':
        tail = np.clip(-(0.9/(scale+eps)) * x + 1, a_min=0.0, a_max=100.0)
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

def sample_inverse_gamma(mean, stdev, size=1):
    if mean <= 0 or stdev <= 0:
        raise ValueError("Mean and standard deviation must be positive.")
    
    # Calculate shape (alpha) and scale (beta)
    alpha = 2 + (mean / (stdev ** 2))
    beta = (alpha - 1) * mean
    
    # Sample from the inverse-gamma distribution
    samples = 1 / np.random.gamma(alpha, 1 / beta, size)
    
    return samples 

def sample_spike_train(pp_mean, pp_stdev, stop_sign):
    cur_accum = 0.0
    spike_loc = []
    while True:
        if pp_stdev == 0:
            v = pp_mean
        else:
            v, = sample_inverse_gamma(pp_mean, pp_stdev, 1)
        assert v > 0.0
        cur_accum += v
        if cur_accum < stop_sign:
            spike_loc.append(cur_accum)
        else:
            break
    return np.array(spike_loc)

def generate_needles_v6(
        dataset_config, num_samples=100,
        needle_shape_li = ('gaussian', 'exp', 'triangle'),
        needle_symmetry_li = ('both', 'left', 'right'),
        filter_bank=True, noise=0.2, needle_gap=None,
        context_len=512, forecast_len=128,
        period_min=4, period_max=128, exp_beta=2,
        plot_signal=False
    ):
    # sample parameters

    pp_mean_li = \
        np.random.uniform(low=period_min, high=period_max, size=num_samples)
    pp_stdev_li = \
        np.random.uniform(low=0.0, high=noise*pp_mean_li, size=num_samples)
    print('pp_mean',pp_mean_li)
    print('pp_stdev',pp_stdev_li)

    # sample functions
    resolution = context_len+forecast_len
    resampling_function_li, golden_indices = \
        generate_resampling_functions(
            num_samples, resolution, context_len/resolution, filter_bank=filter_bank)

    signal_li = []
    for i, (pp_mean, pp_stdev, rf, golden_index) in \
            tqdm(enumerate(zip(
                pp_mean_li, pp_stdev_li,
                resampling_function_li, golden_indices
            ))):
        haystack = sample_wrapperv2(dataset_config, seq_len=resolution).reshape(-1)
        needle = sample_wrapperv2(dataset_config, seq_len=resolution).reshape(-1)
        # haystack = np.sin(np.arange(resolution)/resolution*5*np.pi) #np.zeros(resolution) # TODO
        # needle = np.sin(np.arange(resolution)/resolution*5*np.pi) #np.zeros(resolution) # TODO

        # sample spike locations on [0,1] domain
        # print('aa',golden_index)
        pp_dists_context = \
            sample_spike_train(pp_mean, pp_stdev, golden_index)
        offset = 0.0 if len(pp_dists_context) == 0 else pp_dists_context[-1]
        pp_dists_forecast = \
            sample_spike_train(pp_mean, pp_stdev, resolution-offset)
        # sample_spike_train(pp_mean, 0.0, resolution-offset)

        # print('bb',pp_mean, pp_stdev, golden_index, pp_dists_forecast)
        pp_dists = np.concatenate([
            pp_dists_context, 
            offset + pp_dists_forecast])

        # discretize the spike train
        pp_discrete = np.unique((pp_dists).astype(int))
        pp_resampled = np.unique((resolution*rf[pp_discrete]).astype(int))
        if len(pp_resampled) > 0 and pp_resampled[-1] == resolution:
            pp_resampled = pp_resampled[:-1]

        spike_train = np.zeros(resolution)
        if len(pp_resampled) == 0:
            if np.std(haystack) > 0.0:
                haystack /= np.std(haystack)
            signal = haystack
        else:
            # needle_width, = \
            # 	np.random.uniform(
            # 		low=0.0, 
            # 		high=min(pp_resampled[1:] - pp_resampled[:-1]), 
            # 		size=1)
            if len(pp_resampled) == 1:
                scl = resolution
            else:
                scl = min(pp_resampled[1:] - pp_resampled[:-1])
            if plot_signal:
                needle_width = 16
            else:
                needle_width, = \
                    np.random.exponential(
                        # scale=min(pp_resampled[1:] - pp_resampled[:-1]), 
                        # scale=np.sqrt(pp_mean),
                        scale=exp_beta*np.sqrt(scl), 
                        size=1)
            needle_cfg = {
                'needle_shape_li': needle_shape_li,
                'needle_symmetry_li': needle_symmetry_li,
            }
            needle_shape = generate_needle(needle_width, resolution, **needle_cfg)

            needle_signal = needle[:len(pp_resampled)]
            if np.std(haystack) > 0.0:
                needle_signal /= (np.std(needle_signal) + eps)

            if needle_gap is None:
                (needle_std, haystack_std, _), = np.random.dirichlet((1.5,1.5,1.5),1)
                gap = 1.0
            else:
                (needle_std, haystack_std, gap), = np.random.dirichlet((1.5,1.5,needle_gap*1.5),1)
            sign = 1 if random.random() >= 0.5 else -1

            spike_train[pp_resampled] = sign * (needle_std * needle_signal + gap)
            signal = haystack * haystack_std + scipy.signal.convolve(needle_shape, spike_train, mode='same')

        signal_li.append(np.expand_dims(signal, axis=0))
    return signal_li, resampling_function_li

def plot_and_save_ts_li(nm, ts_li):
    convert_to_arrow(f'{nm}_pt.arrow', time_series=[x[0] for x in ts_li])
    ts_li = ts_li[:10]
    _, ax_li = plt.subplots(len(ts_li), 1, figsize=(10, 1.2*len(ts_li)), constrained_layout=True)
    for i, (ts, ax) in enumerate(zip(ts_li, ax_li)):
        ax.plot(ts[0], alpha=0.5, label=f'signal{i}')
    plt.title(f'{nm}')
    plt.tight_layout()
    plt.savefig(f'_{nm}.png')
    plt.clf()

def main_train():
    random.seed(123)
    np.random.seed(123)
    merged_sample = True
    dataset_config = {
        'Synthetic10K': ArrowDataset(PATH_TO_GAUSSIAN_PROCESSES, 'Synthetic'),
    }

    # get base datasets
    for nm, dataset in dataset_config.items():
        dataset.load()
    if merged_sample:
        dataset_all = MergeDataset(list(dataset_config.values()))
        dataset_all.load()
        dataset_config = {'all': dataset_all}
    ts_li, *_ = generate_needles_v6(
            dataset_config, num_samples=10_000_000, noise=0.4, #10_000_000
            needle_shape_li=('gaussian', 'exp'),
            needle_symmetry_li=('both', 'left', 'right'),
            needle_gap=20, forecast_len=512
        )
    plot_and_save_ts_li('NITH_pretrain', ts_li)

def main_inference():
    random.seed(789)
    np.random.seed(789)
    merged_sample = True
    dataset_config = {
        'Synthetic10K': ArrowDataset(PATH_TO_GAUSSIAN_PROCESSES, 'Synthetic'),
    }

    # get base datasets
    for nm, dataset in dataset_config.items():
        dataset.load()
    if merged_sample:
        dataset_all = MergeDataset(list(dataset_config.values()))
        dataset_all.load()
        dataset_config = {'all': dataset_all}

    for noise in [0, 2, 4]:
        for period in [4, 16, 64, 128]:
            for beta in [0, 1, 2, 3]:
                forecast_len = 128
                ts_li, *_ = generate_needles_v6(
                        dataset_config, num_samples=300, noise=noise/10.0, #10_000_000
                        needle_shape_li=('gaussian', 'exp'),
                        needle_symmetry_li=('both', 'left', 'right'),
                        needle_gap=20, forecast_len=forecast_len,
                        period_min=period, period_max=period, exp_beta=float(beta)
                    )
                dataset_nm = f'{noise}_{period}_{beta}'
                pn = os.path.join(dn, f"{dataset_nm}.arrow")
                convert_to_arrow(pn, time_series=[ts[0] for ts in ts_li])
                zeroshot_cfg = [{
                    'name': dataset_nm,
                    'num_rolls': 1,
                    'offset': -forecast_len,
                    'pn': pn,
                    'prediction_length': forecast_len,
                }]
                with open(f'{PATH_TO_OUTPUT_BENCHMARK}/{dataset_nm}.yaml', 'w') as fp:
                    yaml.dump(zeroshot_cfg, fp)

if __name__ == '__main__':
    if SPLIT == 'pretrain':
        main_train()
    elif SPLIT == 'inference':
        main_train()
    else:
        assert False