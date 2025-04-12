import json
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from .energy_dataset import EnergyDataset


class OPSDDataset(EnergyDataset):
    """
    Energy dataset for OPSD energy data.
    """

    def __init__(self,
                 time: List[pd.DatetimeIndex],
                 data_path: str,
                 targets: List[str],
                 historical_lag: int,
                 forecast_horizon: int,
                 normalize_method: str):
        """ OPSD energy dataset without capacity normalization.

        :param time: List of timesteps for used for this dataset (needed for filtering).
        :param data_path: Location of the OPSD dataset.
        :param targets: List of energy targets (e.g. de+load).
        :param historical_lag: Historical lag in hours (e.g. 168).
        :param forecast_horizon: The forecast horizon in hours for the ground truth values.
        :param normalize_method: Normalizing method to use after
        """
        super().__init__()

        self.time = time
        self.data_path = data_path
        self.targets = targets
        self.historical_lag = historical_lag
        self.forecast_horizon = forecast_horizon
        self.normalize_method = normalize_method

        # load energy raw data
        df = pd.read_csv(self.data_path, header=[0, 1], index_col=0)
        df.index = pd.to_datetime(df.index).tz_localize(None)

        # transform raw energy data into xarray dataset
        data_vars = {}
        for target in self.targets:
            # define target keys for opsd df
            area_key, energy_key = target.split('+')
            data_vars[target] = (['time'], df.loc[:, (area_key, energy_key)])
        ds = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                time=(['time'], df.index)
            )
        )

        # calculate statistical features for normalization
        statistics = dict()
        for energy_target in ds:
            statistics[energy_target] = dict(
                mean=np.nanmean(ds[energy_target]),
                std=np.nanstd(ds[energy_target]),
                min=np.nanmin(ds[energy_target]),
                max=np.nanmax(ds[energy_target])
            )
        self.statistics = statistics

        start_time = self.time[0] - (self.historical_lag - 1) * pd.Timedelta('1h')
        end_time = self.time[-1] + self.forecast_horizon * pd.Timedelta('1h')
        time_filter = pd.date_range(start_time, end_time, freq='1h')
        ds = ds.sel(time=time_filter)

        # interpolate missing data
        print(dict([(x, ds[x].isnull().sum().values.item()) for x in ds]))
        ds = ds.interpolate_na(dim='time', method='linear', fill_value='extrapolate')

        # save dataset as float32
        self.ds = ds.astype(np.float32)

        time_deltas = np.unique(np.diff(ds.time))
        assert len(time_deltas) == 1  # probably not needed because time is set
        assert not np.any([ds[x].isnull().values.any() for x in ds])

    def normalize_energy(self, energy, energy_target):
        statistics = self.statistics[energy_target]
        if self.normalize_method == 'meanstandard':
            mean, std = statistics['mean'], statistics['std']
            return (energy - mean) / std
        elif self.normalize_method == 'standard':
            std = statistics['std']
            return energy / std
        elif self.normalize_method == 'minmax':
            min, max = statistics['min'], statistics['max']
            return (energy - min) / (max - min)

    def inverse_normalize_energy(self, energy, energy_target):
        statistics = self.statistics[energy_target]
        if self.normalize_method == 'meanstandard':
            mean, std = statistics['mean'], statistics['std']
            return (energy * std) + mean
        elif self.normalize_method == 'standard':
            std = statistics['std']
            return energy * std
        elif self.normalize_method == 'minmax':
            min, max = statistics['min'], statistics['max']
            return (energy * (max - min)) + min

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # define start and end timesteps of history and forecast
        current_timestep = self.time[index]
        historical_start = current_timestep - (self.historical_lag - 1) * pd.Timedelta('1h')
        forecast_end = current_timestep + self.forecast_horizon * pd.Timedelta('1h')

        # define historical and forecast timesteps for time series selection
        historical_timesteps = pd.date_range(historical_start, current_timestep, freq='1h')
        forecast_timesteps = pd.date_range(current_timestep + pd.Timedelta('1h'), forecast_end, freq='1h')

        # get actual history and forecast data
        history = self.ds.sel(time=historical_timesteps)
        forecast = self.ds.sel(time=forecast_timesteps)

        # normalize historical and forecast values for each target
        history_values = []
        for energy_target in history:
            energy_values = history[energy_target]
            history_values.append(self.normalize_energy(energy_values, energy_target))
        forecast_values = []
        for energy_target in forecast:
            energy_values = forecast[energy_target]
            forecast_values.append(self.normalize_energy(energy_values, energy_target))

        # convert to numpy array with shape [target, time_series_data]
        history_values = np.array(history_values)
        forecast_values = np.array(forecast_values)

        return current_timestep.isoformat(), history_values, forecast_values

    def __len__(self) -> int:
        return len(self.time)
