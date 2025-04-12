import os
import glob
from typing import Optional, Tuple, List

import numpy as np
import xarray as xr
import torch
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt


class AlphaFusioner(torch.nn.Module):
    """
    Alpha fusion subnetwork to learn trainable alphas
    to use for weighted average to combine spatial forecasts.
    """

    def __init__(self,
                 fusion_method: str,
                 forecast_horizon: int,
                 upper_left: Tuple[float, float],
                 lower_right: Tuple[float, float],
                 scaling: float,
                 temporal: bool = False,
                 dropout: float = 0.0):
        """Initialize fusion subnetwork to forecast based on encoded inputs.

        'fusion_method' to choose:
            - combined: Forecast energy in a single step. Forcast energy using historical and weather forecasts as input.
            - separated: Forecast energy in two steps. Forecast energy based on historical energy
                and weather forecasts separately and fuse forecasts together afterward.

        :param fusion_method: Fusion method to choose for the alpha fusion network.
        :param forecast_horizon: Forecast horizon of the energy forecasting network.
        :param upper_left: Upper left longitude and latitude (needed for spatial alphas).
        :param lower_right: Lower right longitude and latitude (needed for spatial alphas).
        :param scaling: Spatial scaling in longitude and latitude direction.
        :param temporal: If alphas should be temporal, defaults to False.
        :param dropout: Factor of dropout on spatial alphas, defaults to 0.0.
        """
        self.fusion_method = fusion_method
        self.temporal = temporal
        self.dropout = dropout
        self.upper_left = upper_left
        self.lower_right = lower_right
        self.scaling = scaling
        self.forecast_horizon = forecast_horizon
        self.alphas_dir = os.path.join('data', 'alpha_fusion')
        super().__init__()


        x_dim = (self.lower_right[0] - self.upper_left[0]) / self.scaling
        x_dim = int(x_dim) + 1
        y_dim = (self.upper_left[1] - self.lower_right[1]) / self.scaling
        y_dim = int(y_dim) + 1
        self.weather_shape = (y_dim, x_dim)

        self.linear_capacity = torch.nn.Linear(1, 1)
        torch.nn.init.zeros_(self.linear_capacity.weight)
        torch.nn.init.zeros_(self.linear_capacity.bias)

        self.alphas_scaling = torch.nn.Parameter(torch.tensor([1.0]))
        if self.temporal:
            self.linear_alphas = torch.nn.Sequential(
                torch.nn.Linear(1, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, x_dim * y_dim)
            )
            def init_weights(m):
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.zeros_(m.weight)
                    torch.nn.init.zeros_(m.bias)
            self.linear_alphas.apply(init_weights)
        else:
            alphas = torch.zeros((y_dim * x_dim, ))
            self.alphas = torch.nn.Parameter(alphas)
            self.alphas_dropout = torch.nn.Dropout(p=self.dropout)

        if self.fusion_method == 'combined':
            self.forecast_net = torch.nn.Sequential(
                torch.nn.LazyLinear(4 * self.forecast_horizon),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(1),
                torch.nn.Linear(4 * self.forecast_horizon, 2 * self.forecast_horizon),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(1),
                torch.nn.Linear(2 * self.forecast_horizon, self.forecast_horizon),
                torch.nn.BatchNorm2d(1)
            )
        elif self.fusion_method == 'separated':
            self.energy_forecast_net = torch.nn.Sequential(
                torch.nn.LazyLinear(4 * self.forecast_horizon),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(1),
                torch.nn.Linear(4 * self.forecast_horizon, 2 * self.forecast_horizon),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(1),
                torch.nn.Linear(2 * self.forecast_horizon, self.forecast_horizon)
            )

            self.weather_forecats_net = torch.nn.Sequential(
                torch.nn.LazyLinear(4 * self.forecast_horizon),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(1),
                torch.nn.Linear(4 * self.forecast_horizon, 2 * self.forecast_horizon),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(1),
                torch.nn.Linear(2 * self.forecast_horizon, self.forecast_horizon),
                torch.nn.BatchNorm2d(1),
            )

            forecast_fusion_weights = torch.zeros((self.forecast_horizon, ))
            self.forecast_fusion_weights = torch.nn.Parameter(forecast_fusion_weights)

    def get_alphas(self, prediction=False):
        if prediction:
            alphas = self.alphas
        else:
            alphas = self.alphas_dropout(self.alphas)
        alphas = (self.alphas_scaling * alphas).softmax(dim=0)
        alphas = alphas.view(1, *self.weather_shape)
        return alphas

    def get_capacity(self, time):
        capacity = self.linear_capacity(time[:, None]).sigmoid() + 0.5
        return capacity

    def forward(self, energy, weather, calendar, input, prediction=False, **kwargs):
        assert len(energy.shape) == 3
        assert len(weather.shape) == 5
        assert len(calendar.shape) == 3
        # only one target
        assert energy.shape[1] == 1
        assert weather.shape[1] == 1
        assert calendar.shape[1] == 1

        if self.fusion_method == 'combined':
            ####
            ## Forecast on grid level using (energy, weather, calendar)
            ###
            # weather.shape = [BATCH, TARGET, WEATHER_HORIZON, LAT, LONG]
            # weather_input.shape = [BATCH, TARGET * LAT * LONG, WEATHER_HORIZON]
            weather_input = weather.swapaxes(1, 2) \
                                   .flatten(start_dim=2) \
                                   .swapaxes(1, 2)
            # history_input.shape = [BATCH, TARGET * LAT * LONG, ENERGY_FEATURES]
            energy_input = energy.repeat(1, weather_input.shape[1], 1)
            # calendar_input.shape = [BATCH, TARGET * LAT * LONG, CAL_FEATURES]
            calendar_input = calendar.repeat(1, weather_input.shape[1], 1)
            # collected_input.shape = [BATCH, TARGET * LAT * LONG, FEATURES]
            collected_input = torch.cat([energy_input, weather_input, calendar_input], axis=2)
            # weather_forecast.shape = [BATCH, TARGET, FORECAST_HORIZON, LAT, LONG]
            weather_forecast = self.forecast_net(collected_input[:, None])[:, 0] \
                                   .swapaxes(1, 2) \
                                   .view(collected_input.shape[0],
                                         weather.shape[1],
                                         self.forecast_horizon,
                                         *weather.shape[-2:])

            ####
            ## Aggregate grid level forecasts using weather_alphas
            ###
            # alphas.shape = [LAT, LONG]
            # or (temporal alphas)
            # alphas.shape = [BATCH, LAT, LONG]
            batch_size = energy.shape[0]
            alphas = self.get_alphas(prediction).repeat(batch_size, 1, 1)
            # reshape alphas from [BATCH, X, Y] to [BATCH, TARGET, FORECAST_HORIZON, X, Y]
            alphas = alphas[:, None, None].repeat(1, 1, self.forecast_horizon, 1, 1)
            # weather_forecast shape is [BATCH, TARGET, FORECAST_HORIZON, X, Y]
            forecast = weather_forecast * alphas
            forecast = forecast.flatten(start_dim=3).sum(dim=3)
            capacity = self.get_capacity(input[2][:, 0, 0])
            forecast = capacity[:, :, None].repeat(1, 1, self.forecast_horizon) * forecast

            # return.shape = [BATCH, TARGET, FORECAST, FORECAST_HORIZON]
            return forecast[:, None]
        elif self.fusion_method == 'separated':
            ####
            ## Forecast on global level using (energy, calendar)
            ###
            # energy.shape = [BATCH, TARGET, HISTORY]
            # calendar_input.shape = [BATCH, TARGET, FEATURES]
            # history_input.shape = [BATCH, TARGET, HISTORY + FEATURES]
            history_input = torch.cat([energy, calendar], axis=2)
            # history_input.shape = [BATCH, TARGET, FORECAST_HORIZON]
            history_forecast = self.energy_forecast_net(history_input)

            ####
            ## Forecast on grid level using (weather, calendar)
            ###
            # weather.shape = [BATCH, TARGET, WEATHER_HORIZON, LAT, LONG]
            # weather_input.shape = [BATCH, TARGET * LAT * LONG, WEATHER_HORIZON]
            weather_input = weather.swapaxes(1, 2) \
                                   .flatten(start_dim=2) \
                                   .swapaxes(1, 2)
            # calendar_input.shape = [BATCH, TARGET * LAT * LONG, CAL_FEATURES]
            calendar_input = calendar.repeat(1, weather_input.shape[1], 1)
            # collected_input.shape = [BATCH, TARGET * LAT * LONG, CAL_FEATURES + WEATHER_HORIZON]
            collected_input = torch.cat([weather_input, calendar_input], axis=2)
            # weather_forecast.shape = [BATCH, TARGET, FORECAST_HORIZON, LAT, LONG]
            weather_forecast = self.weather_forecats_net(collected_input[:, None])[:, 0] \
                                   .swapaxes(1, 2) \
                                   .view(collected_input.shape[0],
                                         weather.shape[1],
                                         self.forecast_horizon,
                                         *weather.shape[-2:])

            ####
            ## Aggregate grid level forecasts using weather_alphas
            ###
            # alphas.shape = [BATCH, LAT, LONG] original
            # alphas.shape = [BATCH, TARGET, FORECAST_HORIZON, LAT, LONG]
            alphas = self.get_alphas(calendar, prediction)
            capacity = self.get_capacity(calendar)
            alphas = capacity.repeat(1, alphas.shape[1], alphas.shape[2]) * alphas
            alphas = alphas[:, None, None].repeat(1, 1, self.forecast_horizon, 1, 1)
            weather_forecast = weather_forecast * alphas
            # sum_forecasts.shape = [BATCH, TARGET, FORECAST_HORIZON]
            weather_forecast = weather_forecast.flatten(start_dim=3).sum(dim=3)

            ####
            ## Get final forecast in combination with energy and calendar features
            ###
            # # energy.shape = [BATCH, TARGET, FEATURES]
            # # collected_input.shape = [BATCH, TARGET, FORECAST_HORIZON + FEATURES]
            # collected_input = torch.cat([weather_forecast, energy, calendar], axis=2)
            # # collected_input.shape = [BATCH, TARGET, FORECAST_HORIZON]

            start = 0.0
            end = self.forecast_horizon
            scaling = 1.0

            weights = (10 * self.forecast_fusion_weights).sigmoid()
            forecast = weights * history_forecast + (1 - weights) * weather_forecast

            # return.shape = [BATCH, TARGET, FORECAST, FORECAST_HORIZON]
            if prediction:
                return torch.stack([history_forecast, weather_forecast, forecast], dim=2)
            else:
                return forecast[:, None]

    def debug(self, outputs=None, output_dir='.', detailed=False):
        time = torch.arange(0, 1.01, 0.01, requires_grad=False)
        time = time.to(next(self.parameters()).device)

        capacity = self.get_capacity(time).flatten().cpu().numpy()
        plt.plot(time.cpu().numpy(), capacity)
        plt.savefig(os.path.join(output_dir, 'capacity.png'))
        plt.close()
        if detailed:
            np.save(os.path.join(output_dir, 'capacity.npy'), capacity)

        alphas = self.get_alphas(prediction=False)
        alphas = alphas.cpu().numpy()

        if self.temporal:
            steps = np.arange(0, 1.0, step=0.2)
        else:
            steps = [0]
        for step in steps:
            if step is None:
                file_name = 'alphas'
            else:
                file_name = f'alphas_{step:0.2f}'
            alphas_da = xr.DataArray(
                data = alphas[int(step * alphas.shape[0])],
                dims = ['latitude', 'longitude'],
                coords = {
                    'latitude': np.arange(self.upper_left[1],
                                        self.lower_right[1] - self.scaling,
                                        - self.scaling),
                    'longitude': np.arange(self.upper_left[0],
                                        self.lower_right[0] + self.scaling,
                                        self.scaling)
                }
            )

            if detailed:
                alphas_da.to_netcdf(os.path.join(output_dir, f'{file_name}.nc'))

            if detailed:
                fig = plt.figure(figsize=(8, 6))
            else:
                fig = plt.figure()
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE)
            # ax.add_feature(cfeature.LAND)
            # ax.add_feature(cfeature.OCEAN)
            ax.add_feature(cfeature.BORDERS)
            # ax.add_feature(cfeature.LAKES)
            # ax.add_feature(cfeature.RIVERS)

            plot = alphas_da.plot(cmap=plt.cm.plasma,
                               vmin=0, vmax=alphas.max().item(),
                               transform=ccrs.PlateCarree(),
                               cbar_kwargs={'shrink': 0.6})
            plt.title(f"alphas <scaling={self.alphas_scaling.cpu().item():2.2f}>")
            plt.tight_layout()
            if detailed:
                plt.savefig(os.path.join(output_dir, f'{file_name}.png'))
                plt.savefig(os.path.join(output_dir, f'{file_name}.pdf'))
                plt.close()
            else:
                plt.savefig(os.path.join(output_dir, f'{file_name}.png'))
                plt.close()

        if hasattr(self, 'forecast_fusion_weights'):
            weights = (10 * self.forecast_fusion_weights).sigmoid()
            weights = weights.cpu().numpy()
            plt.plot(weights)
            plt.plot(1 - weights)
            plt.savefig(os.path.join(output_dir, 'forecast_fusion_weights.png'))
            plt.close()
            if detailed:
                np.save(os.path.join(output_dir, 'forecast_fusion_weights.npy'), weights)
