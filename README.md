# Forecasting Needles in a Time Series Haystack

## Overview
Spikes are a common characteristic of real-world time series, often representing rare or significant events. However, time series foundation models (TSFMs) struggle to effectively forecast these spiky patterns. To address this challenge, we introduce the **Needle-in-a-Time-Series Haystack Benchmark (NiTS-Haystack)**, a novel benchmark designed to evaluate the performance of TSFMs on spiky time series data. Our benchmark comprises both real-world and synthetic datasets:

- **Real-World Spiky Time Series:** A collection of real-world datasets from domains like energy, traffic, and biomedical signals, reflecting naturally occurring spikes.
- **Synthetic Spiky Time Series:** Generated using stochastic point processes ("Needles") injected into regular stochastic processes ("Haystack") to simulate spiky behavior in controlled environments.

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

### Real-World Spiky Time Series
Due to redistribution restrictions, real-world datasets can be manually downloaded from the following sources:

- [NAB (Numenta Anomaly Benchmark)](https://github.com/numenta/NAB)
- [UCR Anomaly Archive](https://paperswithcode.com/dataset/ucr-anomaly-archive)
- [Timer Dataset](https://thuml.github.io/timer/)

Once downloaded, generate the **Real-NITH** benchmark by running:
```bash
python dataset_selection.py
```

### Synthetic Spiky Time Series
To simulate spiky behavior, synthetic datasets (**syn-NITH**) are generated purely through the following script:
```bash
python get_data.py
```

---

## Evaluation Metric

To evaluate model performance on these challenging datasets, we propose a novel metric: **Spiky Dynamic Time Warping (SDTW)**. This metric is designed to account for temporal lags in spiky signals, overcoming the limitations of traditional regression and anomaly detection metrics. The implementation can be found in:
```bash
SDTW.py
```




---
