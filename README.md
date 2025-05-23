# Forecasting Needles in a Time Series Haystack

## Overview
Spikes are a common characteristic of real-world time series, often representing rare or significant events. However, time series foundation models (TSFMs) struggle to effectively forecast these spiky patterns. To address this challenge, we introduce the **Needle-in-a-Time-Series Haystack Benchmark (NITH)**, a novel benchmark designed to evaluate the performance of TSFMs on spiky time series data. Our benchmark comprises both real-world and synthetic datasets:

- **Synthetic Spiky Time Series:** Generated using stochastic point processes ("Needles") injected into regular stochastic processes ("Haystack") to simulate spiky behavior in controlled environments.
   - **Pretrain:** We provide a continuous pretraining dataset for foundation models to specialize on spiky time series.
   - **Benchmark:** We provide a benchmark datasets to test foundation model sensitivity to different tyeps of spiky time series.
- **Real-World Spiky Time Series:** A collection of real-world datasets from domains like energy, traffic, and biomedical signals, reflecting naturally occurring spikes.

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DerekQXu/NITH.git
cd NITH
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

---

## Dataset Generation

For ease of use, we preprocess all our datasets, which can be downloaded [here](https://www.kaggle.com/datasets/boranhan/syn-nith-pretraining).

### Gaussian Process Time Series

Follow **Generating Synthetic Time Series (KernelSynth)** from [Chronos](https://github.com/amazon-science/chronos-forecasting/tree/main/scripts) to generate some Gaussian Process time series. Be sure to set the output length to 10000 in `kernel-synth.py`:

```
LENGTH = 1024
```

### Pretrain Synthetic Time Series


Create base Gaussian time series, following [Gaussian Process Time Series](#gaussian-process-time-series), saving the output to:

```
/path/to/kernelsynth.arrow
```

To create the synthetic pretraining data, set the following config values in `get_data.py`:

```
PATH_TO_GAUSSIAN_PROCESSES = "/path/to/kernelsynth.arrow"
SPLIT = "pretrain"
```

Generate the **NITH-Train** benchmark by running:
```bash
python generate_dataset.py
```

### Benchmark Synthetic Time Series


Create base Gaussian time series, following [Gaussian Process Time Series](#gaussian-process-time-series), saving the output to:

```
/path/to/kernelsynth.arrow
```

Create an empty directory where the data will be saved:

```
/path/to/benchmark
```

Set the following config values in `get_data.py`:

```
PATH_TO_GAUSSIAN_PROCESSES = "/path/to/kernelsynth.arrow"
SPLIT = "inference"
PATH_TO_OUTPUT_BENCHMARK = "/path/to/benchmark"
```

Generate the **NITH-Synth** benchmark by running:
```bash
python generate_dataset.py
```

### Real-World Spiky Time Series

Due to redistribution restrictions, real-world datasets can be manually downloaded from the following sources:

- [NAB (Numenta Anomaly Benchmark)](https://github.com/numenta/NAB)
- [UCR Anomaly Archive](https://paperswithcode.com/dataset/ucr-anomaly-archive)
- [Timer Dataset](https://thuml.github.io/timer/)

Set the following config values in `filter_datasets.py`:

```
PATH_TO_SAVED_DATASETS = '/path/to/saved_datasets'
PATH_TO_FILTER_CONFIG = '/path/to/NITH/zero-shot.yaml'
```

Once downloaded, generate the **NITH-Real** benchmark by running:
```bash
python filter_datasets.py
```

## Evaluation Metric

To evaluate model performance on these challenging datasets, we propose a novel metric: **Spiky Dynamic Time Warping (SDTW)**. This metric is designed to account for temporal lags in spiky signals, overcoming the limitations of traditional regression and anomaly detection metrics. The implementation can be found in:

```bash
python3 SDTW.py
```

This reproduces the case study on different evaluation metrics in:

```
sdtw_metrics.png
sdtw_time_series.png
```

---
