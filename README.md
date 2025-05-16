# Forecasting Needles in a Time Series Haystack

## Overview
Spikes are a common characteristic of real-world time series, often representing rare or significant events. However, time series foundation models (TSFM) struggle to forecast these spiky patterns effectively. To address this challenge, we introduce the **Needle-in-a-Time-Series Haystack Benchmark (NiTS-Haystack)**, a novel benchmark designed to evaluate the performance of TSFMs on spiky time series data. Our benchmark comprises both real-world and synthetic datasets:

## Installation

1. Clone the repository:
```bash
git clone [https://github.com/your-repo/NiTS-Haystack.git](https://github.com/DerekQXu/NITH.git)
cd NITH
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```
## Dataset Generation.

1. **Real-World Spiky Time Series:** Due to the fact that some datasets have restricted redistribution rights, the real world dataset can be downloaded from their websites, run the the following to generate Real-NITH:

```bash
python dataset_selection.py
```

3. **Synthetic Spiky Time Series:** Stochastic point processes ("Needles") injected into regular stochastic processes ("Haystack") to isolate spiky behavior. The synthetic dataset (syn-NITH) can be generated purely through the python script:

```bash
python get_data.py
```

To better evaluate model performance on these challenging datasets, we propose a new metric: **Spiky Dynamic Time Warping (SDTW)**, which accounts for temporal lags in spiky signals, addressing limitations of traditional regression and anomaly detection metrics. The metric function can be found at SDTW.py. 

---
