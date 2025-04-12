from typing import Optional, Tuple

import numpy as np
import torch

import matplotlib.pyplot as plt


class LatentEncoder(torch.nn.Module):
    """
    Deep Neural Network for processing energy and calendar data,
    where the energy data is processed by fully-connected layers.
    """

    def __init__(self,
                 energy_architecture: Optional[str],
                 latent_energy_size: Optional[int],
                 weather_architecture: Optional[str],
                 latent_weather_size: Optional[int],
                 upper_left: Tuple[float, float],
                 lower_right: Tuple[float, float],
                 scaling: float,
                 weather_horizon: int,
                 calendar_architecture: Optional[str],
                 latent_calendar_size: int):
        """Initialize latent encoder subnetwork.

        Architecture options:
            - None (identity)
            - 'flatten'
            - 'fcn'
            - 'cnn' (energy only)
            - '2dcnn' (weather only)
            - '3dcnn' (weather only)

        :param energy_architecture: Architecture of the energy encoder network.
        :param latent_energy_size: Number of resulting latent energy encoding vector.
        :param weather_architecture: Architecture of the weather encoder network.
        :param latent_weather_size: Number of resulting latent weather encoding vector.
        :param weather_horizon: Temporal horizon of the weather input.
        :param upper_left: Upper left longitude and latitude (needed for spatial alphas).
        :param lower_right: Lower right longitude and latitude (needed for spatial alphas).
        :param scaling: Spatial scaling in longitude and latitude direction.
        :param calendar_architecture: Architecture of the calendar encoder network.
        :param latent_calendar_size: Number of resulting latent calendar encoding vector.
        """
        super().__init__()

        self.energy_architecture = energy_architecture
        self.latent_energy_size = latent_energy_size
        self.weather_architecture = weather_architecture
        self.latent_weather_size = latent_weather_size
        self.weather_horizon = weather_horizon
        self.upper_left = upper_left
        self.lower_right = lower_right
        self.scaling = scaling
        self.calendar_architecture = calendar_architecture
        self.latent_calendar_size = latent_calendar_size

        x_dim = (self.lower_right[0] - self.upper_left[0]) / self.scaling
        x_dim = int(x_dim) + 1
        y_dim = (self.upper_left[1] - self.lower_right[1]) / self.scaling
        y_dim = int(y_dim) + 1
        self.weather_shape = (y_dim, x_dim)

        self._build_energy_net()
        self._build_weather_net()
        self._build_calendar_net()

    def _build_energy_net(self):
        if self.energy_architecture is None:
            self._energy_net = torch.nn.Identity()
            return
        if self.energy_architecture == 'flatten':
            self._energy_net = torch.nn.Flatten(start_dim=1)
            return

        energy_networks = []
        if self.energy_architecture == 'fcn':
            # only use the last linear layers
            pass
        elif self.energy_architecture == 'cnn':
            block = lambda input_size, features: [
                torch.nn.Conv1d(input_size, features, 3, padding='same'),
                torch.nn.ReLU(),
                torch.nn.Conv1d(features, features, 3, padding='same'),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(2),
            ]
            energy_cnn = torch.nn.Sequential(
                *block(1, 16),  # 168 -> 84
                *block(16, 8),  # 84 -> 42
            )
            energy_networks.append(energy_cnn)

        energy_networks.append(
            torch.nn.Sequential(
                torch.nn.LazyLinear(4 * self.latent_energy_size),
                torch.nn.BatchNorm1d(4 * self.latent_energy_size),
                torch.nn.ReLU(),
                torch.nn.Linear(4 * self.latent_energy_size, 2 * self.latent_energy_size),
                torch.nn.BatchNorm1d(2 * self.latent_energy_size),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * self.latent_energy_size, self.latent_energy_size)
            )
        )
        self._energy_net = torch.nn.Sequential(*energy_networks)

    def _build_weather_net(self):
        weather_networks = []
        if self.weather_architecture is None:
            self._weather_net = torch.nn.Identity()
            return
        if self.weather_architecture == 'flatten':
            self._weather_net = torch.nn.Flatten(start_dim=1)
            return

        if self.weather_architecture == 'fcn':
            # only use the last linear layers
            pass
        elif self.weather_architecture == '2dcnn':
            block = lambda input_size, features, groups: [
                torch.nn.Conv2d(input_size, features, 3,
                                groups=groups, padding='same'),
                torch.nn.ReLU(),
                torch.nn.Conv2d(features, features, 3,
                                groups=groups, padding='same'),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2)
            ]
            groups = self.weather_horizon
            weather_cnn = torch.nn.Sequential(
                torch.nn.Upsample(size=(64, 64), mode='bilinear'),
                *block(groups, 4 * groups, groups),  # 64 -> 32
                *block(4 * groups, 4 * groups, groups),  # 32 -> 16
                *block(4 * groups, 4 * groups, groups)  # 16 -> 8
            )
            weather_networks.append(weather_cnn)
        elif self.weather_architecture == '3dcnn':
            block = lambda input_size, features: [
                torch.nn.Conv3d(input_size, features, 3, padding='same'),
                torch.nn.ReLU(),
                torch.nn.Conv3d(features, features, 3, padding='same'),
                torch.nn.ReLU(),
                torch.nn.MaxPool3d(2)
            ]
            weather_cnn = torch.nn.Sequential(
                torch.nn.Upsample(size=(64, 64, self.weather_horizon * 4), mode='trilinear'),
                *block(1, 4),  # 64 -> 32
                *block(4, 8),   # 32 -> 16
                *block(8, 16)   # 16 -> 8
            )
            weather_networks.append(weather_cnn)
        weather_networks.append(
            torch.nn.Sequential(
                torch.nn.Flatten(start_dim=1),
                torch.nn.LazyLinear(4 * self.latent_weather_size),
                torch.nn.BatchNorm1d(4 * self.latent_weather_size),
                torch.nn.ReLU(),
                torch.nn.Linear(4 * self.latent_weather_size, 2 * self.latent_weather_size),
                torch.nn.BatchNorm1d(2 * self.latent_weather_size),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * self.latent_weather_size, self.latent_weather_size),
            )
        )
        self._weather_net = torch.nn.Sequential(*weather_networks)

    def _build_calendar_net(self):
        # calendar processing part
        if self.calendar_architecture is None:
            self._calendar_net = torch.nn.Identity()
            return
        if self.calendar_architecture == 'flatten':
            self._calendar_net = torch.nn.Flatten(start_dim=1)
            return

        if self.calendar_architecture == 'fcn':
            self._calendar_net = torch.nn.Sequential(
                torch.nn.LazyLinear(4 * self.latent_calendar_size),
                torch.nn.BatchNorm1d(4 * self.latent_calendar_size),
                torch.nn.ReLU(),
                torch.nn.Linear(4 * self.latent_calendar_size, 2 * self.latent_calendar_size),
                torch.nn.BatchNorm1d(2 * self.latent_calendar_size),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * self.latent_calendar_size, self.latent_calendar_size),
            )
        else:
            raise NotImplementedError(f'Unkown calendar architecture {self.calendar_architecture} not implemented.')

    def forward(self, energy, weather, calendar, prediction=False):
        latent_energy = self._latent_energy_forward(energy)
        latent_weather = self._latent_weather_forward(weather)
        latent_calendar = self._latent_calendar_forward(calendar)
        return latent_energy, latent_weather, latent_calendar

    def _latent_energy_forward(self, energy_batch):
        if self.energy_architecture == 'cnn':
            energy_batch = energy_batch.view(energy_batch.shape[0], 1, -1)
            forward_batch = self._energy_net(energy_batch)
        else:
            # Currently the models only support one target.
            # Therefore, only one historical energy time series can be passed.
            assert energy_batch.shape[1] == 1
            forward_batch = self._energy_net(energy_batch[:, 0])[:, None]
        return forward_batch

    def _latent_weather_forward(self, weather_batch):
        if self.weather_architecture == '3dcnn':
            weather_batch = weather_batch[:, None]
        # Currently the models only support one weather variable.
        # For more weather variables more weather encoding networks are needed.
        assert weather_batch.shape[1] == 1
        return self._weather_net(weather_batch[:, 0])[:, None]

    def _latent_calendar_forward(self, calendar_batch):
        # Currently the models only support one target.
        # Therefore, only one calendar feature vector can be passed.
        assert calendar_batch.shape[1] == 1
        return self._calendar_net(calendar_batch[:, 0])[:, None]

    def debug(self, outputs=None, output_dir='.', detailed=False):
        pass
