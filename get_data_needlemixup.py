import numpy as np
import random
import os
import matplotlib.pyplot as plt

from copy import deepcopy
from tqdm import tqdm
from gluonts.dataset.arrow import ArrowFile

from generate_datasets import convert_to_arrow

def main():
    random.seed(123)
    merged_sample = True
    dataset_config = {
        'UTSD-2G': ArrowDataset('/mnt/fsx/chronos-forecasting/pretrain_datasets/UTSD-2G.arrow', 'UTSD-2G'),
    }

    # get datasets
    for nm, dataset in dataset_config.items():
        dataset.load()
    
    if merged_sample:
        dataset_all = MergeDataset(list(dataset_config.values()))
        dataset_all.load()
        dataset_config = {'all': dataset_all}
    
    # get ts_li
    ts_li = [mixup(dataset_config) for _ in tqdm(range(100))]

    plot_some(ts_li, '/mnt/fsx/chronos-forecasting/pretrain_datasets/plots_needlemixup')
    # convert_to_arrow("output", time_series=ts_li)



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

def normalize(ts):
    ts = ts - np.median(ts)
    ts = ts/max(ts.std(), 1e-5)
    return ts

class HuggingFaceDataset(SampleDataset):
    def __init__(self, name, subname=None, split=None, id=None):
        self.name = name
        self.subname = subname
        self.split = split
        self.id = id
    
    def load(self):
        self.ts_li = []
        self.create_indices()
        pass

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

class MergeDataset(SampleDataset):
    def __init__(self, dataset_li, id=None):
        self.ts_li = []
        for dataset in dataset_li:
            self.ts_li.extend(dataset.ts_li)
        self.id = id
    
    def load(self):
        self.create_indices()

def sample_wrapper(dataset_config, seq_len):
    candidate_datasets = [dataset for dataset in dataset_config.values() if dataset.max_len >= seq_len]
    dataset, = random.sample(candidate_datasets, k=1)
    ts = dataset.sample(min_size=seq_len)
    start_idx = int(random.random()*(len(ts)-seq_len))
    ts_out = ts[start_idx: start_idx+seq_len]
    ts_out = normalize(ts_out)
    assert len(ts_out) == seq_len
    return ts_out

# mixup global variables
ADDITIVE, REPLACE, CONTINUOUS = \
    'additive', 'replace', 'continuous'
def mixup(
        dataset_config,
        haystack_size_li=[512, 1024, 2048, 4096, 8192],
        num_occurances_li=[4, 5, 6, 7, 8],
        c_li=[-1.4, -0.7, 0.0, 0.7, 1.4],
        effect_li=[ADDITIVE, REPLACE, CONTINUOUS],
        sigma_li=[-3, -1, 0.0, 1, 3],
        min_needle_size=16
    ):
    l_haystack, = random.sample(haystack_size_li, k=1)
    num_occurances, = random.sample(num_occurances_li, k=1)
    c, = random.sample(c_li, k=1)

    # kth occurance e^(c*k) period after (k-1)th occurance...
    period_increments = np.e**(c*np.arange(num_occurances+1)/num_occurances)
    period_increments[0] = 0.0
    period_accumulate = np.cumsum(period_increments)
    Z = sum(period_increments) # normalizing factor
    assert Z == period_accumulate[-1]
    min_period = int(min(period_increments[1:])/Z * l_haystack) # get min period of needle

    assert min_period//2 >= min_needle_size, f'{c} {num_occurances} {l_haystack}' # needle cannot take up >1/2 the signal!
    l_needle = int(random.random() * (min_period//2-min_needle_size) + min_needle_size)
    effect, = random.sample(effect_li, k=1)
    sigma, = random.sample(sigma_li, k=1)

    assert int(period_accumulate[num_occurances]/Z) == 1 # needle is added to the end!
    ts_needles = sample_wrapper(dataset_config, seq_len=(num_occurances+1) * l_needle)
    ts_haystack = deepcopy(sample_wrapper(dataset_config, seq_len=l_haystack + l_needle))
    ts_id = np.zeros_like(ts_haystack)
    for i in range(num_occurances+1):
        needle = sigma + ts_needles[i*l_needle: (i+1)*l_needle] # get the needle
        needle_idx = int(period_accumulate[i] / Z * l_haystack) # get where to place the needle
        
        # place the needle in the haystack
        if effect == ADDITIVE:
            ts_haystack[needle_idx: needle_idx + l_needle] += needle
        elif effect == REPLACE:
            ts_haystack[needle_idx: needle_idx + l_needle] = needle
        elif effect == CONTINUOUS:
            ts_haystack[needle_idx: needle_idx + l_needle] = \
                needle - needle[0] + ts_haystack[needle_idx]
            if i != num_occurances:
                ts_haystack[needle_idx + l_needle:] = \
                    ts_haystack[needle_idx + l_needle:] - ts_haystack[needle_idx + l_needle] + ts_haystack[needle_idx + l_needle - 1]
        else:
            assert False
        ts_id[needle_idx: needle_idx + l_needle] = 1

    cfg = {
        'context_length': l_haystack, 
        'needle_length': l_needle,
        'nonlinearity': c, 
        'number_of_needles': num_occurances, 
        'how_to_insert_needle': effect,
        'DC': sigma
    }
    return (ts_haystack, ts_id, cfg)

def plot_some(ts_li, dn):
    os.system(f'mkdir {dn}')
    for i, (ts, ts_id, cfg) in enumerate(tqdm(ts_li)):
        plt.plot(ts)
        plt.grid()
        plt.title('\n'.join([f'{k}={v}' for k,v in cfg.items()]))
        plt.tight_layout()
        plt.xlabel('Context')
        start = 0
        last_color = 'green'
        for j, value in enumerate(ts_id):
            end = j + 1
            color = 'red' if value == 0 else 'green'
            if color != last_color or j == len(ts_id)-1:
                plt.axvspan(start, end, color=last_color, alpha=0.1)
                start = end
                last_color = color
        plt.savefig(os.path.join(dn, f'{i}.png'))
        plt.clf()

if __name__ == '__main__':
    main()
