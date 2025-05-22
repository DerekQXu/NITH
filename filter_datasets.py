import os
import yaml
import datasets
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle

from scipy.signal import find_peaks
from scipy.stats import skew
from tqdm import tqdm
from gluonts.dataset.arrow import ArrowFile

from generate_datasets import convert_to_arrow

PATH_TO_SAVED_DATASETS = '/path/to/saved_datasets'
PATH_TO_FILTER_CONFIG = '/path/to/NITH/zero-shot.yaml'

with open(PATH_TO_FILTER_CONFIG) as stream:
    zero_shot_cfg = yaml.safe_load(stream)

dataset_nm2hf_cfg = {
    entry['name']: (entry['hf_repo'], entry['name'])
    for entry in zero_shot_cfg
}

print(dataset_nm2hf_cfg)

def ts_filterer2(ts, dist_mean, context_length, forecast_length):
    peaks, *_ = get_peaks(ts)
    is_ts_long_enough = len(ts) >= (forecast_length + forecast_length)
    is_peak_in_forecast = any(peaks >= len(ts)-forecast_length)
    is_peak_in_context = any(peaks < len(ts)-forecast_length)
    enough_peaks = len(peaks) >= 8
    inter_peak_distance = False if len(peaks) < 2 else \
        np.mean(peaks[1:]-peaks[:-1]) <= 64
    return is_ts_long_enough and is_peak_in_context and is_peak_in_forecast and enough_peaks and inter_peak_distance

def ts_filterer(ts, dist_mean, context_length, forecast_length):
    peaks, *_ = get_peaks(ts)
    is_ts_long_enough = len(ts) >= (context_length + forecast_length)
    is_peak_in_forecast = any(peaks >= len(ts)-forecast_length)
    is_peak_in_context = any(peaks < len(ts)-forecast_length)
    return is_ts_long_enough and is_peak_in_context and is_peak_in_forecast

def load_hf_dataset(hf_cfg, max_entries=None):
    ds = datasets.load_dataset(*hf_cfg)['train']

    ts_li = []
    for i, entry in enumerate(tqdm(ds)):
        series_fields = [
            col
            for col in ds.features
            if isinstance(ds.features[col], datasets.Sequence)
        ]
        series_fields.remove("timestamp")

        for nm in series_fields:
            entry_ = entry[nm]
        
        ts_li.append(entry_)
        if max_entries is not None and i >= max_entries:
            break
    return ts_li

def get_peaks(ts, width_min=1, width_max=16):
    ts = np.array(ts)
    peaks, properties = find_peaks((ts-ts.mean())/(ts.std()+1e-12), prominence=2.0, width=(width_min, width_max))
    widths = list(properties['widths']) #np.array(properties['prominences']).mean()
    if len(peaks) >= 2:
        dist = list(peaks[1:] - peaks[:-1])
    else:
        dist = []
    return peaks, widths, dist

def get_diff(ts_li):
    def get_dist_partial(ts):
        peaks, _, dist = get_peaks(ts)
        return dist

    def get_diff_partial(ts):
        peaks, _, dist = get_peaks(ts)
        # if np.isnan(prom):
        #     prom = 0.0
        return len(peaks) * min(512/len(ts), 1.0)
        # ts = (ts-ts.mean())/ts.std()
        # # return skew(np.abs(ts))
        # from scipy.fft import fft, fftfreq
        # # Assuming `signal` is your time-domain signal and `sampling_rate` is the rate at which it's sampled
        # N = len(ts)
        # T = 1.0
        # yf = fft(ts)
        # xf = fftfreq(N, T)[:N // 2]
        # # Compute the power spectrum
        # power_spectrum = np.abs(yf[:N // 2]) ** 2
        # # Compute the spectral centroid
        # spectral_centroid = np.sum(xf * power_spectrum) / np.sum(power_spectrum)
        # return spectral_centroid
    diff_li = np.array([get_diff_partial(np.array(ts)) for ts in ts_li])
    diff_mean, diff_std = diff_li.mean(), diff_li.std()
    dist_dist = []
    for ts in ts_li:
        dist_dist.extend(get_dist_partial(ts))
    return diff_mean, diff_std, dist_dist

def load_data():
    dataset2dataset_cfg = {}
    for dataset_nm in [x for x in os.listdir(PATH_TO_SAVED_DATASETS) if '.pkl' in x]:
        with open(os.path.join(PATH_TO_SAVED_DATASETS, f'{dataset_nm}'), 'rb') as fp:
            ts_li = pickle.load(fp)
        dataset_nm = dataset_nm.replace('.pkl', '')
        # ts_li = load_hf_dataset(hf_cfg, max_entries=None)

        T_li_full = []
        widths_full = []
        ts_metadata_li = []
        for ts in ts_li:
            needle_indices, widths, duration_li = get_peaks(ts)
            T_li_full.extend(duration_li)
            widths_full.extend(widths)
            ts_metadata_li.append({
                'ts': ts,
                'needle_indices': needle_indices,
                'widths': widths,
                'duration_li': duration_li
            })

        if len(T_li_full) > 1:
            T_li_full = np.array(T_li_full)
            mu_T, sigma_T = T_li_full.mean(), T_li_full.std()
            omega = np.array(widths_full).mean()
        else:
            mu_T, sigma_T = None, None
        dataset2dataset_cfg[dataset_nm] = {
            'ts_metadata_li': ts_metadata_li,
            'mu_T': mu_T,
            'sigma_T': sigma_T,
            'omega': omega
        }
    return dataset2dataset_cfg

def ts_filterer_final(ts, needle_indices, context_length, forecast_length, **kwargs):
    is_ts_long_enough = len(ts) >= (forecast_length + forecast_length)
    is_peak_in_forecast = np.sum(needle_indices >= len(ts)-forecast_length) >= 1
    is_peak_in_context = np.sum(np.logical_and(
        needle_indices < len(ts)-forecast_length, 
        needle_indices >= max(0,len(ts)-context_length-forecast_length))) >= 3
    is_enough_peaks = len(needle_indices) >= 8
    return is_ts_long_enough and is_peak_in_context and is_peak_in_forecast # and is_enough_peaks

def filter_data(dataset2dataset_cfg):
    min_diff = 8
    dataset2dataset_cfg_valid = {}
    ln_li = []
    for dataset_nm, dataset_cfg in sorted(dataset2dataset_cfg.items()):
        mu_T = dataset_cfg['mu_T']
        ts_metadata_li = dataset_cfg['ts_metadata_li']
        if mu_T is None or mu_T > 64:
            ln_li.append(f'@@ mu_T={mu_T} {dataset_nm.replace("_","-")} & 0 & {len(ts_metadata_li)} & - & - & - & - \\\\')
            continue

        forecast_length = 128 #int(2*mu_T)
        context_length = 512
        ts_metadata_li_valid = []
        T_li_full = []
        widths_full = []
        for ts_metadata in ts_metadata_li:
            if len(ts_metadata_li_valid) >= 1000:
                break
            if ts_filterer_final(
                    context_length=context_length, 
                    forecast_length=forecast_length, 
                    **ts_metadata):
                kronecker_delta_comb = np.zeros(len(ts_metadata['ts']))
                kronecker_delta_comb[ts_metadata['needle_indices']] = 1.0
                ts_metadata_li_valid.append((
                    ts_metadata['ts'],
                    {
                        'needle_mask': kronecker_delta_comb, 
                        'needle_indices': ts_metadata['needle_indices'],
                        'kernel': None
                    },
                    {
                        'needle_count': len(ts_metadata['needle_indices']),
                        'period': None,
                        'omega': None,
                        'sigma_T': None,
                        'scale': None,
                        'DC': None,
                        'flip': None,
                    },
                ))
                assert ts_metadata['duration_li'] is not None and len(ts_metadata['duration_li']) > 0
                T_li_full.extend(list(ts_metadata['duration_li']))
                widths_full.extend(list(ts_metadata['widths']))
        
        
        if len(ts_metadata_li_valid) > 0:
            assert len(T_li_full) > 0
            T_li_full = np.array(T_li_full)
            mu_T_filtered, std_T_filtered = T_li_full.mean(), T_li_full.std()
            omega = np.array(widths_full).mean()
            dataset2dataset_cfg_valid[dataset_nm] = {
                'ts_li': ts_metadata_li_valid,
                'offset': forecast_length,
                'mu_T': mu_T_filtered,
                'sigma_T': std_T_filtered,
                'T_li_full': T_li_full,
                'omega': omega
            }
            # print(f'{dataset_nm} (Inter-Peak Distance={mu_T:.3f}): {len(ts_metadata_li_valid)} valid time series (out of {len(ts_metadata_li)}), average length={np.array([len(x[0]) for x in ts_metadata_li_valid]).mean()}')
            ln_li.append(f'{dataset_nm.replace("_","-")} & {len(ts_metadata_li_valid)} & {len(ts_metadata_li)} & {np.array([len(x[0]) for x in ts_metadata_li_valid]).mean():.3f} & {mu_T_filtered:.3f} & {omega:.3f} & {std_T_filtered:.3f} \\\\')
        else:
            ln_li.append(f'@@ {dataset_nm.replace("_","-")} & 0 & {len(ts_metadata_li)} & - & - & - & - \\\\')

    for ln in sorted(ln_li):
        print(ln)
    return dataset2dataset_cfg_valid

if __name__ == '__main__':
    dataset2dataset_cfg = load_data()
    dataset2dataset_cfg_valid = filter_data(dataset2dataset_cfg)
    assert False