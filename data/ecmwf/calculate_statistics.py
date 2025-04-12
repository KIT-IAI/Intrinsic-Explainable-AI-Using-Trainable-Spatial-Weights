import os
import json

import numpy as np
import xarray as xr

from tqdm import tqdm


def get_statistics():
    era5_path = os.path.join('era5_25', '*.nc')
    era5 = xr.open_mfdataset(era5_path)

    statistics = {}
    for key in tqdm(era5):
        mean = np.nanmean(era5[key])
        std = np.nanstd(era5[key])
        min = np.nanmin(era5[key])
        max = np.nanmax(era5[key])
        statistics[key] = {
            'mean': mean.astype(float),
            'std': std.astype(float),
            'min': min.astype(float),
            'max': max.astype(float),
        }

    data = np.sqrt(era5['u10'] ** 2 + era5['u10'] ** 2)
    mean = np.nanmean(data)
    std = np.nanstd(data)
    min = np.nanmin(data)
    max = np.nanmax(data)
    statistics['u10+v10'] = {
        'mean': mean.astype(float),
        'std': std.astype(float),
        'min': min.astype(float),
        'max': max.astype(float),
    }

    data = np.sqrt(era5['u100'] ** 2 + era5['u100'] ** 2)
    mean = np.nanmean(data)
    std = np.nanstd(data)
    min = np.nanmin(data)
    max = np.nanmax(data)
    statistics['u100+v100'] = {
        'mean': mean.astype(float),
        'std': std.astype(float),
        'min': min.astype(float),
        'max': max.astype(float),
    }

    return statistics


if __name__ == '__main__':
    statistics = get_statistics()
    with open('statistics.json', 'w') as json_file:
        json.dump(statistics, json_file)