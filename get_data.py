from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import warnings
import time
import scipy
import random
import pickle
import math
import os
import yaml

from get_data_needlemixup import ArrowDataset, MergeDataset, normalize, plot_some
from generate_datasets import convert_to_arrow


eps = 1e-12

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
        # TODO: plot first 20 curves
        # for i in np.arange(int(len(basis)/10))*10:
        # 	plt.plot(basis[i], label=f'sample {i}', alpha=0.6)
        # plt.legend()
        # plt.savefig('part1.png')
        # plt.clf()
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

    # TODO: plot first 20 curves
    # for i in range(3):
    # 	plt.plot(out[i], label=f'sample {i}', alpha=0.6)
    # # plt.legend()
    # plt.savefig('part2.png')
    # plt.clf()
    print('mixup done.')

    ####################################################
    # Find Locations
    ####################################################
    print('computing cutoffs...')
    golden_indices = []
    for i, arr in tqdm(enumerate(out)):
        golden_indices.append(np.searchsorted(arr, cutoff))
    golden_indices = np.array(golden_indices)

    # # TODO: plot histogram
    # plt.hist(golden_indices)
    # plt.savefig('part3.png')
    # print('cutoffs computed.')
    # plt.clf()

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


# def generate_needles_v6(
# 		num_samples, min_period=4, max_period=128, noise=0.2, 
# 		context_len=512, forecast_len=512):
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

        # convolve the signal
        if plot_signal:
            plot_ts(
                signal, needle_shape, needle_signal, haystack, 
                sign, needle_std, haystack_std, 
                pp_resampled, spike_train, gap, resolution)
            assert False

        signal_li.append(np.expand_dims(signal, axis=0))
    return signal_li, resampling_function_li

def plot_ts(signal, needle_shape, needle_signal, haystack, sign, needle_std, haystack_std, pp_resampled, spike_train, gap, resolution):
    needle = (np.arange(resolution), scipy.signal.convolve(needle_shape, spike_train, mode='same'))
    haystack = (np.arange(resolution), haystack * haystack_std)
    needle_raw = (pp_resampled, sign * (needle_std * needle_signal + gap))

    plt.clf()
    plt.figure(figsize=(6,2.5))
    plt.plot(np.arange(resolution), signal, color='black', label='û(t)')
    plt.plot(*needle, color='blue', linestyle='--', label='Σi (λn ŝi(t) + ν λg) * δ(t - τi)')
    plt.plot(*haystack, color='green', linestyle=':', label='λh ĥi(t)')
    plt.scatter(*needle_raw, color='red', marker='x', label="[ĥ'(τi)]i")
    plt.legend()
    plt.title('Needle-in-a-Haystack Time Series')
    plt.xlabel('t')
    plt.tight_layout()
    plt.savefig('/mnt/fsx/_main_signal.png')

    plt.clf()
    plt.figure(figsize=(2,2.5))
    mp = int(len(needle_shape)/2)
    plt.plot(np.arange(200)-100, needle_shape[mp-100:mp+100])
    plt.title('Spike Shape')
    plt.xlabel('t')
    plt.ylabel('z(t)')
    plt.tight_layout()
    plt.savefig('/mnt/fsx/_spike_signal.png')





# def generate_needles_v5(
#         dataset_config, N=1_000_000, sequence_length=5000, forecast_length=128, 
#         fixed_period=None, fixed_omega=None, fixed_sigma_T=None, 
#         include_original_signal=False
#     ):
#     '''
#     context_length = 512
#     forecast_length = 128
#     min_needles = 8
#     -> >6.4 needles in context
#     -> >1.6 needles in forecast
#     '''
#     ts_all = []
#     for _ in tqdm(range(N)):
#         period = fixed_period if fixed_period is not None else int(U(8,64)) #int(2**U(3,6))
#         omega = fixed_omega if fixed_omega is not None else int(2**U(0,max(np.log2(period)-1, 0))) #U(1,4) #int(2**U(0,max(np.log2(period)-1, 0))) #int(2**U(0,max(np.log2(period)-2, 0)))
#         sigma_T = fixed_sigma_T if fixed_sigma_T is not None else int(2**U(-1,6-2)) #np.log2(omega)))
#         ts = generate_needles_v5_helper(
#             dataset_config, period=period, omega=omega, sigma_T=sigma_T, 
#             sequence_length=sequence_length, forecast_length=forecast_length, 
#             include_original_signal=include_original_signal)
#         ts_all.append(ts)
#     return ts_all

# def generate_needles_v5_helper(
#         dataset_config, sequence_length=5000, forecast_length=128, #512+128, forecast_length=128,
#         period=None, omega=None, sigma_T=None, include_original_signal=False
#     ):
#     '''
#     context_length = 512
#     forecast_length = 128
#     min_needles = 8
#     -> >6.4 needles in context
#     -> >1.6 needles in forecast
#     '''
#     ts_all = []
#     Q_inv = 1.959964

#     # sample periods
#     T_accum_li = None
#     needle_count = math.ceil(sequence_length/period)

#     # generate until needle in haystack
#     while T_accum_li is None or T_accum_li[-1] < sequence_length-forecast_length: # or T_accum_li[-1] >= 8192:
#         if T_accum_li is not None:
#             print(f'No needle in forecast: regerating Dirac Comb {T_accum_li[-1]} not in ({sequence_length-forecast_length}, {sequence_length}); period={period}, omega={omega}, sigma_T={sigma_T}')
#         kT = sigma_T/period
#         T_accum_li = np.cumsum(sample_T_li_good(period, needle_count, kT))
#         offset_start = int(random.random() * T_accum_li[0])
#         T_accum_li -= offset_start
#         assert all(T_accum_li >= 0)
#         T_accum_li = T_accum_li[T_accum_li < sequence_length]
#         # make sure Dirac comb includes needle in forecast and at most 256 < context < 8192
#         # T_accum_li = T_accum_li[T_accum_li >= 0]
#         # if T_accum_li[-1] >= 8192:
#         #     needle_count = max(1, needle_count-1)
#         # if T_accum_li[-1] < 256-forecast_length:
#         #     needle_count += 1
#     # assert needle_count == len(T_accum_li), f'{needle_count} {T_accum_li}'

#     # make the needle signal an outlier
#     scale = U(3,5)
#     DC = U(1,5) * scale
#     flip = +1 if random.random() < 0.8 else -1
#     ts_onlyneedle = flip*(scale * normalize(sample_wrapperv2(dataset_config, seq_len=len(T_accum_li))) + DC)

#     # construct kronecker delta comb
#     kronecker_delta_comb = np.zeros(sequence_length)
#     for n_hat_sample, T_accum in zip(ts_onlyneedle, T_accum_li):
#         if T_accum >= len(kronecker_delta_comb):
#             break
#         kronecker_delta_comb[T_accum] = n_hat_sample
#     # kronecker_delta_comb = kronecker_delta_comb[:max(256, T_accum_li[-1]+offset_end)]
#     L = len(kronecker_delta_comb) # this is just sequence_length

#     # construct kernel
#     from scipy.stats import norm
#     sigma_K = omega/(2*Q_inv)
#     tdomain = np.arange(L, dtype=float) * L/(L-1)
#     tdomain -= tdomain[int(L/2)]
#     tdomain /= sigma_K
#     kernel = 1/sigma_K * norm.pdf(tdomain) * (sigma_K/norm.pdf(0)) # renormalize
#     kernel = (kernel > 0.03).astype(float)

#     # convolution
#     ts_haystack = normalize(sample_wrapperv2(dataset_config, seq_len=L)).reshape(-1)
#     ts_needle = np.convolve(kernel, kronecker_delta_comb, mode='same').reshape(-1)

#     if include_original_signal:
#         ts_needle_in_haystack = ts_haystack + ts_needle
#     else:
#         ts_needle_in_haystack = normalize(ts_haystack + ts_needle)
    
#     # check if needle in forecast
#     needle_mask = kronecker_delta_comb.astype(bool)
#     needle_indices = np.nonzero(kronecker_delta_comb)[0].reshape(-1)
#     assert any(needle_indices >= len(ts_needle_in_haystack) - forecast_length), f'debug: {len(ts_needle_in_haystack)}, {needle_indices}, {T_accum_li}, {offset_start}, {offset_end}'

#     # construct output
#     ts_metadata = {
#             'needle_mask': needle_mask, 
#             'needle_indices': needle_indices,
#             'kernel': kernel
#         }
#     if include_original_signal:
#         new_x = np.arange(max(needle_indices)-min(needle_indices))+min(needle_indices)
#         only_needle = np.interp(new_x, needle_indices, ts_onlyneedle[:len(needle_indices)])
#         ts_metadata['ts_onlyneedle'] = (new_x, only_needle)
#         ts_metadata['ts_haystack'] = normalize(ts_haystack)
#     ts = \
#         (ts_needle_in_haystack, 
#         ts_metadata, {
#             'needle_count': needle_count,
#             'period': period,
#             'omega': f'{omega:.2f}',
#             'sigma_T': f'{sigma_T:.2f}',
#             'scale': f'{scale:.2f}',
#             'DC': f'{DC:.2f}',
#             'flip': flip,
#             # 'sigma': sigma,
#         })
#     return ts

# random.seed(789) # 123
# np.random.seed(789) # 123

def plot_and_save_ts_li(nm, ts_li):
    convert_to_arrow(f'_{nm}_pt.arrow', time_series=[x[0] for x in ts_li])
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
        'Synthetic': ArrowDataset('/mnt/fsx/chronos-forecasting/scripts/kernelsynth-data-1024.arrow', 'Synthetic'),
        'Synthetic10K': ArrowDataset('/mnt/fsx/chronos-forecasting/scripts/kernelsynth-data-eff.arrow', 'Synthetic'),
        # 'UTSD-2G': ArrowDataset('/mnt/fsx/chronos-forecasting/pretrain_datasets/UTSD-2G.arrow', 'UTSD-2G'),
    }

    # get base datasets
    for nm, dataset in dataset_config.items():
        dataset.load()
    if merged_sample:
        dataset_all = MergeDataset(list(dataset_config.values()))
        dataset_all.load()
        dataset_config = {'all': dataset_all}

    # HYPERPARAMETER TUNING DONE! LET'S SKIP THIS!
    for nm, needle_shape_li, needle_symmetry_li, ng, noise in [
        # ('nshape_g_nsymmetry_both', ('gaussian',), ('both',)),
        # ('nshape_ge_nsymmetry_both', ('gaussian', 'exp'), ('both',)),
        # ('nshape_get_nsymmetry_both', ('gaussian', 'exp', 'triangle'), ('both',)),
        # ('nshape_g_nsymmetry_all', ('gaussian',), ('both', 'left', 'right')),
        # ('nshape_ge_nsymmetry_all', ('gaussian', 'exp'), ('both', 'left', 'right')),
        # ('nshape_get_nsymmetry_all', ('gaussian', 'exp', 'triangle'), ('both', 'left', 'right'))
        # ('nshape_ge_nsymmetry_both_00_none', ('gaussian', 'exp'), ('both',), None, 0.1),
        # ('nshape_ge_nsymmetry_both_00_none', ('gaussian', 'exp'), ('both',), None, 0.2),
        # ('nshape_ge_nsymmetry_both_00_none', ('gaussian', 'exp'), ('both',), None, 0.3),
        # ('nshape_ge_nsymmetry_both_00_3', ('gaussian', 'exp'), ('both',), 3, 0.0),
        # ('nshape_ge_nsymmetry_both_00_5', ('gaussian', 'exp'), ('both',), 5, 0.0),
        # ('nshape_ge_nsymmetry_both_00_10', ('gaussian', 'exp'), ('both',), 10, 0.0),
        # ('nshape_ge_nsymmetry_both_00_20', ('gaussian', 'exp'), ('both',), 20, 0.0),
        ('nshape_ge_nsymmetry_all_00_3', ('gaussian', 'exp'), ('both', 'left', 'right'), 20, 0.0),
    ]:
        break
        ts_li, *_ = generate_needles_v6(
                dataset_config, num_samples=5_000_000, noise=noise, #5_000_000
                needle_shape_li=needle_shape_li,
                needle_symmetry_li=needle_symmetry_li,
                needle_gap=ng
            )
        plot_and_save_ts_li(nm, ts_li)
    
    # GOLDEN SETTINGS!
    # ts_li, *_ = generate_needles_v6(
    #         dataset_config, num_samples=10_000_000, noise=0.0, #5_000_000
    #         needle_shape_li=('gaussian', 'exp'),
    #         needle_symmetry_li=('both',),
    #         needle_gap=None, forecast_len=512
    #     )
    # plot_and_save_ts_li('GOLDEN_ICML_512FORECAST', ts_li)
    # ts_li, *_ = generate_needles_v6(
    #         dataset_config, num_samples=10_000_000, noise=0.0, #5_000_000
    #         needle_shape_li=('gaussian', 'exp'),
    #         needle_symmetry_li=('both',),
    #         needle_gap=None, forecast_len=128
    #     )

    # ts_li, *_ = generate_needles_v6(
    #         dataset_config, num_samples=10_000_000, noise=0.0, #10_000_000
    #         needle_shape_li=('gaussian', 'exp'),
    #         needle_symmetry_li=('both',),
    #         needle_gap=20, forecast_len=512
    #     )
    # plot_and_save_ts_li('GOLDEN_ICML_512FORECASTV1', ts_li)
    # ts_li, *_ = generate_needles_v6(
    #         dataset_config, num_samples=10_000_000, noise=0.0, #10_000_000
    #         needle_shape_li=('gaussian', 'exp'),
    #         needle_symmetry_li=('both', 'left', 'right'),
    #         needle_gap=None, forecast_len=512
    #     )
    # plot_and_save_ts_li('GOLDEN_ICML_512FORECASTV2', ts_li)
    # ts_li, *_ = generate_needles_v6(
    #         dataset_config, num_samples=10_000_000, noise=0.0, #10_000_000
    #         needle_shape_li=('gaussian', 'exp'),
    #         needle_symmetry_li=('both', 'left', 'right'),
    #         needle_gap=20, forecast_len=512
    #     )
    # plot_and_save_ts_li('GOLDEN_ICML_512FORECASTV3', ts_li)
    
    # ts_li, *_ = generate_needles_v6(
    #         dataset_config, num_samples=10_000_000, noise=0.2, #10_000_000
    #         needle_shape_li=('gaussian', 'exp'),
    #         needle_symmetry_li=('both', 'left', 'right'),
    #         needle_gap=20, forecast_len=512
    #     )
    # plot_and_save_ts_li('GOLDEN_ICML_512FORECASTV3_02', ts_li)
    ts_li, *_ = generate_needles_v6(
            dataset_config, num_samples=10_000_000, noise=0.4, #10_000_000
            needle_shape_li=('gaussian', 'exp'),
            needle_symmetry_li=('both', 'left', 'right'),
            needle_gap=20, forecast_len=512
        )
    plot_and_save_ts_li('GOLDEN_ICML_512FORECASTV3_04', ts_li)

def main_inference():
    random.seed(789) # 123
    np.random.seed(789) # 123
    merged_sample = True
    dataset_config = {
        'Synthetic': ArrowDataset('/mnt/fsx/chronos-forecasting/scripts/kernelsynth-data-1024.arrow', 'Synthetic'),
        'Synthetic10K': ArrowDataset('/mnt/fsx/chronos-forecasting/scripts/kernelsynth-data-eff.arrow', 'Synthetic'),
        # 'UTSD-2G': ArrowDataset('/mnt/fsx/chronos-forecasting/pretrain_datasets/UTSD-2G.arrow', 'UTSD-2G'),
    }

    # get base datasets
    for nm, dataset in dataset_config.items():
        dataset.load()
    if merged_sample:
        dataset_all = MergeDataset(list(dataset_config.values()))
        dataset_all.load()
        dataset_config = {'all': dataset_all}

    # HYPERPARAMETER TUNING DONE! LET'S SKIP THIS!
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
                dn = f'/mnt/fsx/chronos-forecasting/pretrain_datasets/needle_synthbench_icml'
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
                with open(f'/mnt/fsx/chronos-forecasting/pretrain_datasets/needle_synthbench_icml/{dataset_nm}.yaml', 'w') as fp:
                    yaml.dump(zeroshot_cfg, fp)



def plot_positive_negative_casestudies(dn2nm=None):
    dn = '/mnt/fsx/chronos-forecasting/scripts/evaluation/results/plot'
    dataset_li_pos = ['UCR_Lightning7', 'UCR_RefrigerationDevices', 'UCR_LargeKitchenAppliances', 'Timer_Electricity', 'NAB_realAWSCloudwatch', 'UCR_UWaveGestureLibraryX']
    dataset_li_neg = ['UCR_ShapeletSim', 'UCR_PigAirwayPressure', 'UCR_Earthquakes']
    fn_li = ['chronos_small_trained.pkl', 'chronos_small.pkl']

    for fn in fn_li:
        pn = os.path.join(dn, fn)
        with open(pn, 'rb') as fp:
            plot_dict = pickle.load(fp)
        
        cur_iter = 1
        plt.figure(figsize=(12,5))
        for dataset in dataset_li_pos:
            plt.subplot(math.ceil(len(dataset_li_pos)/3), 3, cur_iter)
            cur_iter += 1
            d = plot_dict[dataset][0]
            context = d['context'][-512:]
            true = d['true']
            forecast = d['pred']

            mean = np.mean(forecast, axis=0)
            std = np.std(forecast, axis=0)
            prediction_length = len(true)
            x = np.arange(len(context) + prediction_length)
            plt.title(dataset)
            plt.plot(
                x, np.hstack((context, true))[:len(context)+prediction_length], 
                label='true', alpha=1.0, linestyle='--', color='black')
            plt.plot(
                x[len(context):len(context)+prediction_length], 
                mean[:prediction_length], 
                label='lagllama', alpha=0.8)
            plt.fill_between(
                x[len(context):len(context)+prediction_length], 
                mean-std[:prediction_length], 
                mean+std[:prediction_length],
                alpha=0.5)
        plt.tight_layout()
        plt.savefig(pn.replace('.pkl', '_pos.png'))

        pn = os.path.join(dn, fn)
        with open(pn, 'rb') as fp:
            plot_dict = pickle.load(fp)
        
        cur_iter = 1
        plt.figure(figsize=(12,2.5))
        for dataset in dataset_li_neg:
            plt.subplot(math.ceil(len(dataset_li_neg)/3), 3, cur_iter)
            cur_iter += 1
            d = plot_dict[dataset][0]
            context = d['context'][-512:]
            true = d['true']
            forecast = d['pred']

            mean = np.mean(forecast, axis=0)
            std = np.std(forecast, axis=0)
            prediction_length = len(true)
            x = np.arange(len(context) + prediction_length)
            plt.title(dataset)
            plt.plot(
                x, np.hstack((context, true))[:len(context)+prediction_length], 
                label='true', alpha=1.0, linestyle='--', color='black')
            plt.plot(
                x[len(context):len(context)+prediction_length], 
                mean[:prediction_length], 
                label='lagllama', alpha=0.8)
            plt.fill_between(
                x[len(context):len(context)+prediction_length], 
                mean-std[:prediction_length], 
                mean+std[:prediction_length],
                alpha=0.5)
        plt.tight_layout()
        plt.savefig(pn.replace('.pkl', '_neg.png'))


def plot_all_casestudies(dn2nm=None):
    dn = '/mnt/fsx/chronos-forecasting/scripts/evaluation/results/plot'
    for fn in os.listdir(dn):
        if '.pkl' not in fn:
            continue
        print(fn)
        pn = os.path.join(dn, fn)
        with open(pn, 'rb') as fp:
            plot_dict = pickle.load(fp)

        try:
            dataset_li = sorted(list(plot_dict.keys()), key=lambda x: [float(z) for z in x.split('_')])
        except:
            dataset_li = sorted(list(plot_dict.keys()))
        num_samples = 1 #len(list(plot_dict.values())[0])
        num_datasets = len(plot_dict)

        plt.subplots(math.ceil(num_datasets/4), 4*num_samples, figsize=(20,25))#25))
        print(dataset_li)

        cur_iter = 1
        for i in range(num_datasets):
            for j in range(num_samples):
                plt.subplot(math.ceil(num_datasets/4), 4*num_samples, cur_iter)
                cur_iter += 1
                d = plot_dict[dataset_li[i]][j]
                context = d['context'][-512:]
                true = d['true']
                forecast = d['pred']

                mean = np.mean(forecast, axis=0)
                std = np.std(forecast, axis=0)
                prediction_length = len(true)
                x = np.arange(len(context) + prediction_length)
                plt.title(dataset_li[i])
                plt.plot(
                    x, np.hstack((context, true))[:len(context)+prediction_length], 
                    label='true', alpha=1.0, linestyle='--', color='black')
                plt.plot(
                    x[len(context):len(context)+prediction_length], 
                    mean[:prediction_length], 
                    label='lagllama', alpha=0.8)
                plt.fill_between(
                    x[len(context):len(context)+prediction_length], 
                    mean-std[:prediction_length], 
                    mean+std[:prediction_length],
                    alpha=0.5)
        plt.tight_layout()
        plt.savefig(pn.replace('.pkl', '.png'))


if __name__ == '__main__':
    main_train()
    # main_inference()
    # plot_all_casestudies()
    # plot_positive_negative_casestudies()

    # random.seed(678) # 123
    # np.random.seed(678) # 123
    # dataset_config = {
    #     'Synthetic': ArrowDataset('/mnt/fsx/chronos-forecasting/scripts/kernelsynth-data-1024.arrow', 'Synthetic'),
    #     'Synthetic10K': ArrowDataset('/mnt/fsx/chronos-forecasting/scripts/kernelsynth-data-eff.arrow', 'Synthetic'),
    #     # 'UTSD-2G': ArrowDataset('/mnt/fsx/chronos-forecasting/pretrain_datasets/UTSD-2G.arrow', 'UTSD-2G'),
    # }
    # # get base datasets
    # for nm, dataset in dataset_config.items():
    #     dataset.load()
    # # if merged_sample:
    # dataset_all = MergeDataset(list(dataset_config.values()))
    # dataset_all.load()
    # dataset_config = {'all': dataset_all}
    # generate_needles_v6(
    #         dataset_config, num_samples=1, noise=0.2,
    #         needle_shape_li=('exp',),
    #         needle_symmetry_li=('both',),
    #         needle_gap=20, forecast_len=128,
    #         period_min=64, period_max=64, exp_beta=3.0,
    #         plot_signal=True
    #     )
    # assert False