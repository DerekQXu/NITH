from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from gluonts.dataset.arrow import ArrowWriter


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


import datasets
if __name__ == "__main__":
    ds = datasets.load_dataset('autogluon/chronos_datasets', 'solar', split="train")
    ds.set_format("numpy")
    time_series = ds['power_mw']
    convert_to_arrow("./solar.arrow", time_series=time_series)

    ds = datasets.load_dataset('autogluon/chronos_datasets', 'solar_1h', split="train")
    ds.set_format("numpy")
    time_series = ds['power_mw']
    convert_to_arrow("./solar_1h.arrow", time_series=time_series)

    ds = datasets.load_dataset('autogluon/chronos_datasets', 'weatherbench_daily', split="train")
    ds.set_format("numpy")
    time_series = ds['target']
    convert_to_arrow("./weatherbench_daily.arrow", time_series=time_series)
    
    ds = datasets.load_dataset('autogluon/chronos_datasets', 'weatherbench_weekly', split="train")
    ds.set_format("numpy")
    time_series = ds['target']
    convert_to_arrow("./weatherbench_weekly.arrow", time_series=time_series)
    
    ds = datasets.load_dataset('autogluon/chronos_datasets', 'wiki_daily_100k', split="train")
    ds.set_format("numpy")
    time_series = ds['target']
    convert_to_arrow("./wiki_daily_100k.arrow", time_series=time_series)
    
    ds = datasets.load_dataset('autogluon/chronos_datasets', 'wind_farms_hourly', split="train")
    ds.set_format("numpy")
    time_series = ds['target']
    convert_to_arrow("./wind_farms_hourly.arrow", time_series=time_series)
    
    # ds = datasets.load_dataset('autogluon/chronos_datasets', 'ushcn_daily', split="train")
    # ds.set_format("numpy")
    # time_series = ds['power_mw']
    # convert_to_arrow("./noise-data.arrow", time_series=time_series)
    
    # ds = datasets.load_dataset('autogluon/chronos_datasets_extra', 'brazilian_cities_temperature', split="train", trust_remote_code=True)
    # ds.set_format("numpy")
    # time_series = ds['power_mw']
    # convert_to_arrow("./noise-data.arrow", time_series=time_series)
    
    # ds = datasets.load_dataset('autogluon/chronos_datasets_extra', 'spanish_energy_and_weather', split="train")
    # ds.set_format("numpy")
    # time_series = ds['power_mw']
    # convert_to_arrow("./noise-data.arrow", time_series=time_series)
