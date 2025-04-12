from typing import List, Optional, Tuple
import json

import numpy as np
import pandas as pd
import xarray as xr

from ..base_dataset import BaseDataset


class HRESDataset(BaseDataset):
    """
    Weather dataset to load HRES forecast weather data from ECMWF.
    """

    def __init__(self,
                 time: List[pd.DatetimeIndex],
                 data_path: str,
                 upper_left: Tuple[float, float],
                 lower_right: Tuple[float, float],
                 scaling: float,
                 weather_vars: List[str],
                 weather_horizon: int,
                 normalize_method: str,
                 weather_stats_path: str,
                 store_ram: bool = False):
        """ HRES weather dataset offered by ECMWF.

        :param time: List of timesteps the dataset should cover.
        :param data_path: Path for the ECMWF datsets.
        :param upper_left: Float tuple of the upper left corner.
        :param lower_right: Float tuple of the lower right corner.
        :param scaling: Scaling of the weather dataset (needed for up or downscaling).
        :param weather_vars: List of weather variables (e.g. t2m or ssrd).
        :param weather_horizon: Future weather horizon in hours to return (e.g. 24)
        :param normalize_method: Normalization method to use (e.g. stanard).
        :param weather_stats_path: Path of the weather stats (needed for normalization).
        :param store_ram: If the dataset should be stored in ram (faster access), defaults to False.
        """
        super().__init__()

        self.time = time
        self.data_path = data_path
        self.upper_left = upper_left
        self.lower_right = lower_right
        self.scaling = scaling
        self.weather_vars = list(weather_vars)
        self.weather_horizon = weather_horizon
        self.normalize_method = normalize_method
        self.store_ram = store_ram

        with open(weather_stats_path, 'r') as stats_json:
            self.weather_stats = json.load(stats_json)

        # load weather dataset from multiple netcad files
        ds = xr.open_mfdataset(
            self.data_path,
            engine='h5netcdf',
            chunks=dict(time=1)
        )
        ds = ds.astype(np.float32)
        ds = ds.isel(step=slice(1, self.weather_horizon + 1))

        # filter by time
        ds = ds.sel(time=self.time)

        # filter location
        ds = ds.where(
            # upper left
            (ds.longitude >= self.upper_left[0]) \
            & (ds.latitude <= self.upper_left[1]) \
            # lower right
            & (ds.longitude <= self.lower_right[0]) \
            & (ds.latitude >= self.lower_right[1]),
            drop=True)

        # filter weather variables to reduce memory consumption
        vars = []
        for weather_var in self.weather_vars:
            if '+' in weather_var:
                vars.extend(weather_var.split('+'))
            else:
                vars.append(weather_var)
        ds = ds[vars]
        self.ds = ds.astype(np.float32)

        if self.store_ram:
            self.ds = self.ds.load()

    def normalize_weather(self, weather, weather_var):
        statistics = self.weather_stats[weather_var]
        if self.normalize_method == 'standard':
            mean, std = statistics['mean'], statistics['std']
            return (weather - mean) / std
        elif self.normalize_method == 'minmax':
            min, max = statistics['min'], statistics['max']
            return (weather - min) / (max - min)

    def __getitem__(self, index) -> np.ndarray:
        timestep = self.time[index]
        weather = self.ds.sel(time=timestep)

        # replace nan values:
        weather = weather.fillna(0.0)
        # weather = weather.interpolate_na(dim=['longitude', 'latitude'])

        weather_values = []
        for weather_var in self.weather_vars:
            if '+' in weather_var:
                var1, var2 = weather_var.split('+')
                weather_data = np.sqrt(weather[var1] ** 2 + weather[var2] **2)
            else:
                weather_data = weather[weather_var]
            weather_data = self.normalize_weather(weather_data, weather_var)
            weather_values.append(weather_data)

        # Create numpy array with shape [1, X, Y, WEATHER_HORIZON]
        weather_data = np.array(weather_values)
        weather_data = weather_data.swapaxes(1, -1)
        weather_data = weather_data.swapaxes(1, 2)

        return np.array(weather_values)

    def __len__(self) -> int:
        return len(self.time)
