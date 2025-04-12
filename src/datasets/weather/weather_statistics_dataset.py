from typing import List

import numpy as np

from ..base_dataset import BaseDataset
from .era5_dataset import ERA5Dataset
from .hres_dataset import HRESDataset


class WeatherStatisticsDataset(BaseDataset):
    """
    Weather dataset to load ERA5 reanalysis data from ECMWF.
    """

    def __init__(self,
                 weather_source: str,
                 weather_statistics: List[str],
                 **kwargs):
        """ Weather dataset transformed using statistical features.

        :param weather_source: Wanted weather source (era5 or hres).
        :param weather_statistics: List of statistical features to use.

        Additional parameters should be passed that are needed for the ERA5 or HRES dataset.
        """
        super().__init__()

        self.weather_source = weather_source
        self.weather_statistics = weather_statistics

        if weather_source == 'era5':
            self.weather_dataset = ERA5Dataset(**kwargs)
        elif weather_source == 'hres':
            self.weather_dataset = HRESDataset(**kwargs)

    def __getitem__(self, index) -> np.ndarray:
        weather_data = self.weather_dataset[index]
        weather_data = weather_data.swapaxes(1, -1)
        weather_data_flatten = weather_data.reshape(*weather_data.shape[:2], -1)
        features = []
        for statistics in self.weather_statistics:
            feature = getattr(weather_data_flatten, statistics)(axis=2)
            features.append(feature)
        weather_data = np.stack(features, axis=-1)
        return weather_data

    def __len__(self) -> int:
        return len(self.weather_dataset)
