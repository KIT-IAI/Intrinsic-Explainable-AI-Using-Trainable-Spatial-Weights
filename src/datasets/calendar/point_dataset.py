import json

from typing import Optional, List

import numpy as np
import pandas as pd
import xarray as xr

import pytz
import pycountry
import holidays

from ..base_dataset import BaseDataset


class PointDataset(BaseDataset):
    """
    Calendar feature dataset that offer calender information
    based on the actual timestep.
    """

    def __init__(self,
                 time: List,
                 targets: List[str],
                 forecast_horizon: int,
                 eu_countries_path: str,
                 start_date: int,
                 end_date: int):
        """ Calendar features dataset based on the actual timstep.

        :param time: List of timesteps.
        :param targets: List of energy targets (e.g. de+load).
        :param forecast_horizon: The forecast horizon in hours (e.g. 24)
        :param eu_countries_path: Path to json file consisting available eu countries.
        :param start_date: Date the calendar features will start.
        :param end_date: Date the calendar features will end.
        """
        super().__init__()

        self.time = time
        self.targets = targets
        self.forecast_horizon = forecast_horizon
        self.year_start = pd.to_datetime(start_date).year
        self.year_end = pd.to_datetime(end_date).year
        self.delta = self.year_end + 1 - self.year_start
        self.hours = np.unique(self.time.hour)
        self.months = np.unique(self.time.month)

        with open(eu_countries_path, 'r') as countries_json:
            self.eu_countries = json.load(countries_json)

        holidays_countries = dict()
        for target in self.targets:
            area, energy = target.split('+')
            if energy == 'load':
                country_code = area.split('_')[0]
                if country_code == 'eu':
                    country_codes = self.eu_countries[energy]
                    for country_code in country_codes:
                        holidays_countries[country_code] = getattr(holidays, country_code.upper())()
                else:
                    holidays_countries[country_code] = getattr(holidays, country_code.upper())()
        self.holidays_countries = holidays_countries

    def __getitem__(self, index) -> np.ndarray:
        time = self.time[index]
        calendar_features = []
        for target in self.targets:
            area, energy = target.split('+')
            if energy == 'load':
                timezone = pytz.timezone('Europe/Berlin')
                local_time = timezone.localize(time)
                one_hot_encodings = [
                    *[time.hour == i for i in self.hours],
                    # *[time.month == i for i in self.months],
                    local_time.tzname().lower() == 'cet',
                    local_time.tzname().lower() == 'cest',
                    *[(time + pd.Timedelta(f'{24 * i}h')).dayofweek >= 5
                      for i in range(-3, int(self.forecast_horizon / 24) + 2)]
                ] # 2 + 7
                for holidays_country in self.holidays_countries.values():
                    one_hot_encodings.extend(
                        [(time - pd.Timedelta(f'{24 * i}h')) in holidays_country
                         for i in range(-3, int(self.forecast_horizon / 24) + 2)]
                    )
                sin_cos_encodings = [
                    # (time.hour, 24),
                    (time.dayofweek, 7),
                    # ((time.month - 1), 12),
                    ((time.dayofyear - 1), 365)
                ] # 3 * 2
            else:
                one_hot_encodings = [
                    # *[time.hour == i for i in self.hours]
                ] # 0
                sin_cos_encodings = [
                    (time.hour, 24),
                    # ((time.month - 1), 12),
                    ((time.dayofyear - 1), 365)
                ] # 2 * 2

            # trend encoding for all models first calendar feature
            features = [
                ((time.year - self.year_start) * 12 + time.month - 1) / (self.delta * 12)
            ]
            features.extend(one_hot_encodings)
            for data, tmax in sin_cos_encodings:
                features.append(np.sin(2 * np.pi * data / (tmax + 1)))
                features.append(np.cos(2 * np.pi * data / (tmax + 1)))
            features = np.array(features).T
            calendar_features.append(features)

        return np.array(calendar_features, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.time)
