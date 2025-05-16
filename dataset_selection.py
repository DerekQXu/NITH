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

with open('/mnt/fsx/chronos-forecasting/scripts/evaluation/configs/zero-shot.yaml') as stream:
    zero_shot_cfg = yaml.safe_load(stream)

dn = '/mnt/fsx/chronos-forecasting/pretrain_datasets/analysis_dataset'

dataset_nm2hf_cfg = {
    entry['name']: (entry['hf_repo'], entry['name'])
    for entry in zero_shot_cfg
}

print(dataset_nm2hf_cfg)
# assert False

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

def main_dataset_saving(dataset_nm2ts_li_valid):
    zeroshot_cfg = []
    for dataset_nm, ts_li_valid in dataset_nm2ts_li_valid.items():
        ts_li = [x[0] for x in ts_li_valid['ts_li']]
        offset = ts_li_valid['offset']
        pn = f"/mnt/fsx/chronos-forecasting/pretrain_datasets/needle_zeroshot2/{dataset_nm}.arrow"
        convert_to_arrow(pn, time_series=ts_li)
        zeroshot_cfg.append({
            'name': dataset_nm,
            'num_rolls': 1,
            'offset': -offset,
            'pn': pn,
            'prediction_length': offset,
        })
        with open(f'/mnt/fsx/chronos-forecasting/pretrain_datasets/needle_zeroshot2/{dataset_nm}.pkl', 'wb') as fp:
            pickle.dump(ts_li_valid['ts_li'], fp)
    with open('/mnt/fsx/chronos-forecasting/pretrain_datasets/needle_zeroshot2/config.yaml', 'w') as fp:
        yaml.dump(zeroshot_cfg, fp)

def main_selection(dataset_nm2dataset_cfg):
    min_diff = 8
    dataset_nm2dataset_cfg_valid = {k:v for k,v in dataset_nm2dataset_cfg.items() if v['diff'] > 0}
    dataset_nm2ts_li_valid = {}
    for dataset_nm, dataset_cfg in dataset_nm2dataset_cfg_valid.items():
        ts_li = dataset_cfg['ts_li']
        dist_mean = dataset_cfg['dist_mean']
        if np.isnan(dist_mean):
            continue

        forecast_length = int(2*dist_mean) #128 #int(2*dist_mean)
        context_length = 8*forecast_length

        ts_li_valid = [ts for ts in ts_li if ts_filterer2(ts, dist_mean, context_length, forecast_length)]
        if len(ts_li_valid) > 0:
            dataset_nm2ts_li_valid[dataset_nm] = {
                'ts_li': ts_li_valid,
                'offset': forecast_length
            }
            print(f'{dataset_nm} (Inter-Peak Distance={dist_mean:.3f}): {len(ts_li_valid)} valid time series (out of {len(ts_li)})')
    

    for dataset_nm, ts_li_valid in tqdm(dataset_nm2ts_li_valid.items()):
        dataset_cfg_ = dataset_nm2dataset_cfg[dataset_nm]
        pn = os.path.join(dn, f'{dataset_cfg_["diff"]:.2f}_{dataset_nm}.png')
        plot_ts(dataset_nm, ts_li_valid['ts_li'][0], dataset_cfg_['dist_dist'], dataset_cfg_['dist_mean'], pn=pn)
    return dataset_nm2ts_li_valid

def main_in_domain():
    pn = '/mnt/fsx/chronos-forecasting/pretrain_datasets/UTSD-2G.arrow'
    ds = ArrowFile(pn)
    ts_li = [np.array(ts['target']) for ts in tqdm(ds)]
    ts_li_out = []

    dn = '/mnt/fsx/chronos-forecasting/pretrain_datasets/analysis_dataset/UTSD-2G'
    os.system(f'rm -r {dn}')
    os.system(f'mkdir {dn}')
    print('starting filtering!')
    num_needles_li, dist_li = [], []
    for ts in tqdm(ts_li):
        peaks, _, dist = get_peaks(ts)
        if not (len(peaks) > 8 and skew(dist) > 4):
            continue
        
        dist_mean = np.mean(dist)
        forecast_length = int(2*dist_mean)
        context_length = 8*forecast_length
        if not (dist_mean <= 64 and ts_filterer(ts, dist_mean, context_length, forecast_length)):
            continue
        
        plt.clf()
        plt.plot(ts)
        plt.axvline(len(ts)-forecast_length, color='red', linestyle='dashed', linewidth=1)
        plt.savefig(os.path.join(dn, f'{len(ts_li_out)}.png'))
        plt.clf()
        ts_li_out.append(ts)

        num_needles_li.append(len(peaks))
        dist_li.append(dist_mean)
    
    print(f'# In-domain samples: {len(ts_li_out)}; num_needles={np.array(num_needles_li).mean():.3f}, distance={np.array(dist_li).mean():.3f}')
    random.seed(123)
    random.shuffle(ts_li_out)
    split = int(0.8*len(ts_li_out))
    pn = f"/mnt/fsx/chronos-forecasting/pretrain_datasets/needle_indomain/UTSD-2G-train.arrow"
    convert_to_arrow(pn, time_series=ts_li_out[:split])
    pn = f"/mnt/fsx/chronos-forecasting/pretrain_datasets/needle_indomain/UTSD-2G-test.arrow"
    convert_to_arrow(pn, time_series=ts_li_out[split:])

def main():
    dn = '/mnt/fsx/chronos-forecasting/pretrain_datasets/analysis_dataset'
    os.system(f'rm -r {dn}')
    os.system(f'mkdir {dn}')

    dataset_nm2dataset_cfg = {}
    for dataset_nm, hf_cfg in tqdm(dataset_nm2hf_cfg.items()):
        ts_li = load_hf_dataset(hf_cfg, max_entries=None)#300)
        diff, diff_std, dist_dist = get_diff(ts_li)
        dataset_cfg = {
            'ts_li': ts_li,
            'ts': ts_li[0],
            'diff': diff,
            'diff_std': diff_std,
            'dist_dist': dist_dist,
            'dist_mean': np.mean(dist_dist),
        }
        dataset_nm2dataset_cfg[dataset_nm] = dataset_cfg
    
        # # for dataset_nm, dataset_cfg in tqdm(dataset_nm2dataset_cfg.items()):
        # pn = os.path.join(dn, f'{dataset_cfg["diff"]:.2f}_{dataset_nm}.png')
        # plot_ts(dataset_nm, dataset_cfg['ts'], dataset_cfg['dist_dist'], dataset_cfg['dist_mean'], pn=pn)
    
    pn = os.path.join(dn, f'selection.png')
    plot_diff(dataset_nm2dataset_cfg, pn=pn)
    return dataset_nm2dataset_cfg

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

def plot_ts(dataset_nm, ts, dist_dist, dist_mean, pn, dist_std=0.0, omega=0.0):
    plt.figure(figsize=(15,5))
    ts = np.array(ts)
    plt.clf()

    ax = plt.subplot(1,3,1)
    peaks, *_ = get_peaks(ts)
    plt.plot(ts)
    if len(peaks) > 0:
        plt.plot(peaks, ts[peaks], 'x')
    ax.set_title(f'{dataset_nm}: Full Signal')
    
    ax = plt.subplot(1,3,2)
    peaks = [x-max(0, len(ts)-512) for x in peaks if x >= len(ts)-512]
    ts = ts[-512:]
    plt.plot(ts)
    if len(peaks) > 0:
        plt.plot(peaks, ts[peaks], 'x')
    ax.set_title(f'{dataset_nm}: Last 512 Samples')

    ax = plt.subplot(1,3,3)
    bins = 200
    # start, stop = np.log10(min(dist_dist)), np.log10(max(dist_dist))
    # bins = 10 ** np.linspace(start, stop, bins)
    plt.hist(dist_dist, bins=bins, log=True)
    # plt.xscale('log')
    # sk = skew(dist_dist)
    # plt.axvline(dist_mean, color='red', linestyle='dashed', linewidth=1)
    ax.set_title(f'mu_T={dist_mean:.2f}, std_T={dist_std:.2f}, omega={omega:.2f}')

    plt.savefig(pn)
    plt.clf()

def plot_diff(dataset_nm2dataset_cfg, pn):
    plt.clf()
    x_li = []
    y_li = []
    yerr_li = []
    for dataset_nm, dataset_cfg in sorted(list(
            dataset_nm2dataset_cfg.items()), 
            key = lambda x: x[1]['diff']):
        x_li.append(dataset_nm)
        y_li.append(dataset_cfg['diff'])
        yerr_li.append(dataset_cfg['diff_std'])
    x_pos = np.arange(len(x_li))
    plt.barh(x_pos, y_li, xerr=yerr_li, align='center')
    plt.yticks(x_pos, labels=x_li)
    plt.gca().invert_yaxis()  # labels read top-to-bottom
    plt.xlabel('Metric')
    plt.title('Dataset Analysis')
    plt.savefig(pn)
    plt.tight_layout()
    plt.clf()


def load_data():
    dn = '/mnt/fsx/chronos-forecasting/pretrain_datasets/analysis_dataset'
    os.system(f'rm -r {dn}')
    os.system(f'mkdir {dn}')

    dataset2dataset_cfg = {}
    # for dataset_nm, hf_cfg in tqdm(dataset_nm2hf_cfg.items()):
    #     ts_li = load_hf_dataset(hf_cfg, max_entries=None)
    dn = '/mnt/fsx/chronos-forecasting/pretrain_datasets/realworld'
    for dataset_nm in [x for x in os.listdir(dn) if '.pkl' in x]:
        with open(os.path.join(dn, f'{dataset_nm}'), 'rb') as fp:
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
        # assert any(['Timer-Solar' in x for x in ln_li])
    for dataset_nm, dataset_cfg_valid in tqdm(dataset2dataset_cfg_valid.items()):
        pn = os.path.join(dn, f'{dataset_nm}.png')
        plot_ts(dataset_nm, dataset_cfg_valid['ts_li'][0][0], dataset_cfg_valid['T_li_full'], dataset_cfg_valid['mu_T'], dist_std=dataset_cfg_valid['sigma_T'], omega=dataset_cfg_valid['omega'], pn=pn)
    return dataset2dataset_cfg_valid

if __name__ == '__main__':
    dataset2dataset_cfg = load_data()
    dataset2dataset_cfg_valid = filter_data(dataset2dataset_cfg)
    # main_dataset_saving(dataset2dataset_cfg_valid)
    assert False

    
    dataset_nm2dataset_cfg = main()
    dataset_nm2ts_li_valid = main_selection(dataset_nm2dataset_cfg)
    main_dataset_saving(dataset_nm2ts_li_valid)
    # main_in_domain()