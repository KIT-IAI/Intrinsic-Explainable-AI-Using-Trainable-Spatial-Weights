import os
import glob
import pickle
from typing import Union

import numpy as np
from sklearn.decomposition import PCA

from ..base_dataset import BaseDataset
from .era5_dataset import ERA5Dataset
from .hres_dataset import HRESDataset


class WeatherPCADataset(BaseDataset):
    """
    Weather dataset to load ERA5 reanalysis data from ECMWF.
    """

    def __init__(self,
                 weather_source: str,
                 n_components: Union[int, float],
                 save_dir: str,
                 **kwargs):
        """ Weather dataset transformed using PCA.

        :param weather_source: Wanted weather source (era5 or hres).
        :param n_components: Number of components for the PCA (int or float).
        :param save_dir: Directory to save the PCA object for other datasets.

        Additional parameters should be passed that are needed for the ERA5 or HRES dataset.
        """
        super().__init__()

        self.weather_source = weather_source
        self.n_components = n_components
        self.save_dir = save_dir

        if weather_source == 'era5':
            self.weather_dataset = ERA5Dataset(**kwargs)
        elif weather_source == 'hres':
            self.weather_dataset = HRESDataset(**kwargs)

        files = glob.glob(os.path.join(self.save_dir, 'pca_*.pkl'))
        files = np.sort(files)
        pca_transformers = []
        if len(files) > 0:
            for file in files:
                with open(file, 'rb') as pkl_file:
                    pca_transformers.append(pickle.load(pkl_file))
        else:
            n_weather_features = self.weather_dataset[0].shape[-1]
            for i in range(n_weather_features):
                x = [weather_data[..., i].flatten()
                     for weather_data in self.weather_dataset]
                x = np.stack(x, axis=0)
                pca_transformer = PCA(n_components=self.n_components)
                pca_transformer = pca_transformer.fit(x)
                pca_transformers.append(pca_transformer)
                path = os.path.join(self.save_dir, f'pca_{i:02}.pkl')
                with open(path, 'wb') as file:
                    pickle.dump(pca_transformer, file)
        self.pca_transformers = pca_transformers

    def __getitem__(self, index) -> np.ndarray:
        weather_data = self.weather_dataset[index]
        weather_features = []
        for i, pca_transformer in enumerate(self.pca_transformers):
            x = weather_data[..., i]
            x = x.reshape(x.shape[0], -1)
            weather_transformed = pca_transformer.transform(x)
            weather_features.append(weather_transformed)
        return np.array(weather_features)

    def __len__(self) -> int:
        return len(self.weather_dataset)
