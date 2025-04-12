import numpy as np

from torch.utils.data import Dataset

from src.datasets import BaseDataset
from src.datasets.energy import EnergyDataset


class ForecastingDataset(Dataset):
    """
    Dataset class to return tuple data for multiple input networks.
    """
    def __init__(self,
                 energy: EnergyDataset,
                 weather: BaseDataset,
                 calendar: BaseDataset,
                 is_test: bool=True):
        """Initialize energy, calendar, and target datasets.

        Dataset returning energy, weather, and calendar data.
        Additionally, if train dataset, also target forecast is returned.

        :param energy: Energy dataset.
        :param weather: Weather dataset.
        :param calendar: Calendar dataset.
        :param is_test: Specify if dataset is train or test, defaults to True.
        """
        assert len(energy) == len(weather) == len(calendar)
        self.energy = energy
        self.weather = weather
        self.calendar = calendar
        self.is_test = is_test

    def __getitem__(self, index):
        """Get tuple data for multiple input networks.

        :param index: Index of the data item that should be loaded.
        :return: Energy, weather, and calendar input and (optional) target forecast.
        """
        timestep, energy, target = self.energy[index]
        weather = self.weather[index]
        calendar = self.calendar[index]

        if self.is_test:
            return (timestep, energy, weather, calendar)
        else:
            return (timestep, energy, weather, calendar), target

    def __len__(self):
        """ Get the length of the dataset. """
        return len(self.energy)