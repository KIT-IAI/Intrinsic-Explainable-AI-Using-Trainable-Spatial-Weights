import os
import glob
from typing import Optional, Tuple, List

import numpy as np
import xarray as xr
import torch
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt


class SeparatedAlphaFusioner(torch.nn.Module):
    """
    Alpha fusion subnetwork to learn trainable alphas
    to use for weighted average to combine spatial forecasts.
    """

    def __init__(self,
                 forecast_horizon: int,
                 upper_left: Tuple[float, float],
                 lower_right: Tuple[float, float],
                 scaling: float,
                 dropout: float = 0.0):
        """Initialize fusion subnetwork to forecast based on encoded inputs.

        :param fusion_method: Fusion method to choose for the alpha fusion network.
        :param forecast_horizon: Forecast horizon of the energy forecasting network.
        :param upper_left: Upper left longitude and latitude (needed for spatial alphas).
        :param lower_right: Lower right longitude and latitude (needed for spatial alphas).
        :param scaling: Spatial scaling in longitude and latitude direction.
        :param dropout: Factor of dropout on spatial alphas, defaults to 0.0.
        """
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
        alphas = torch.zeros((y_dim * x_dim, ))
        self.alphas = torch.nn.Parameter(alphas)
        self.alphas_dropout = torch.nn.Dropout(p=self.dropout)

        self.energy_forecast_net = torch.nn.Sequential(
            torch.nn.LazyLinear(4 * self.forecast_horizon),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1),
            torch.nn.Linear(4 * self.forecast_horizon, 2 * self.forecast_horizon),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1),
            torch.nn.Linear(2 * self.forecast_horizon, self.forecast_horizon)
        )

        self.weather_forecast_net = torch.nn.Sequential(
            torch.nn.LazyLinear(4 * self.forecast_horizon),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(1),
            torch.nn.Linear(4 * self.forecast_horizon, 2 * self.forecast_horizon),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(1),
            torch.nn.Linear(2 * self.forecast_horizon, self.forecast_horizon),
        )

        self.forecast_fusion_weight_x0 = torch.nn.Parameter(torch.tensor([0.0]))
        self.forecast_fusion_weight_x1 = torch.nn.Parameter(torch.tensor([0.0]))
        self.forecast_fusion_scaling = torch.nn.Parameter(torch.tensor([1.0]))
        self.forecast_fusion_weights = torch.nn.Parameter(torch.zeros(self.forecast_horizon))
        self.forecast_fusion_weights_net = torch.nn.Sequential(
            torch.nn.LazyLinear(2 * self.forecast_horizon),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1),
            torch.nn.Linear(2 * self.forecast_horizon, self.forecast_horizon)
        )

    def get_alphas(self, prediction=False):
        if prediction:
            alphas = self.alphas
        else:
            alphas = self.alphas_dropout(self.alphas)
        alphas = (self.alphas_scaling * alphas).softmax(dim=0)
        alphas = alphas.view(*self.weather_shape)
        return alphas

    def get_capacity(self, time):
        capacity = self.linear_capacity(time[:, None]).sigmoid() + 0.5
        return capacity

    def get_forecast_fusion_weights(self, calendar):
        # scaling = self.forecast_fusion_scaling
        # weights = self.forecast_fusion_weights.sigmoid()
        # weights = scaling * (weights - 0.5) + 0.5
        # return weights
        weights = self.forecast_fusion_weights_net(calendar).sigmoid()
        return weights

    def forward(self, energy, weather, calendar, input, prediction=False, **kwargs):
        assert len(energy.shape) == 3
        assert len(weather.shape) == 5
        assert len(calendar.shape) == 3
        # only one target
        assert energy.shape[1] == 1
        assert weather.shape[1] == 1
        assert calendar.shape[1] == 1

        ####
        ## Forecast on global level using (energy, calendar)
        ###
        # energy.shape = [BATCH, TARGET, HISTORY]
        # calendar_input.shape = [BATCH, TARGET, FEATURES]
        # history_input.shape = [BATCH, TARGET, HISTORY + FEATURES]
        history_forecast_input = torch.cat([energy, calendar], axis=2)
        # history_input.shape = [BATCH, TARGET, FORECAST_HORIZON]
        history_forecast = self.energy_forecast_net(history_forecast_input)

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
        weather_forecast_input = torch.cat([weather_input, calendar_input], axis=2)
        # weather_forecast.shape = [BATCH, TARGET, FORECAST_HORIZON, LAT, LONG]
        weather_forecast = self.weather_forecast_net(weather_forecast_input[:, None])[:, 0] \
                                .swapaxes(1, 2) \
                                .view(weather_forecast_input.shape[0],
                                        weather.shape[1],
                                        self.forecast_horizon,
                                        *weather.shape[-2:])

        ####
        ## Aggregate grid level forecasts using weather_alphas
        ###
        # alphas.shape = [LAT, LONG]
        alphas = self.get_alphas(prediction)
        # transform alphas shape from [LAT, LONG] to [BATCH, LAT, LONG]
        alphas = alphas[None].repeat(energy.shape[0], 1, 1)
        # transform alphas shape from [BATCH, LAT, LONG] to weather_forecast shape
        alphas = alphas[:, None, None].repeat(1, 1, self.forecast_horizon, 1, 1)
        # elementwise multiply weather_forecast and alphas to get spatial forecasts
        weather_forecast = weather_forecast * alphas
        # aggregate all spatial forecasts to one forecast with shape [BATCH, 1, FORECAST_HORIZON]
        weather_forecast = weather_forecast.flatten(start_dim=3).sum(dim=3)
        # apply capacity on weather based energy forecast
        capacity = self.get_capacity(input[2][:, 0, 0])
        weather_forecast = capacity[:, :, None].repeat(1, 1, self.forecast_horizon) * weather_forecast

        ####
        ## Get final forecast in combination with energy and calendar features
        ###
        weights = self.get_forecast_fusion_weights(calendar)
        # weights = weights[:, None]
        forecast = weights * history_forecast + (1 - weights) * weather_forecast

        # return.shape = [BATCH, TARGET, FORECAST, FORECAST_HORIZON]
        if prediction:
            return torch.stack([history_forecast, weather_forecast, forecast], dim=2)
        else:
            # return forecast[:, :, None]
            return torch.stack([history_forecast, weather_forecast, forecast], dim=2)

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
        alphas_da = xr.DataArray(
            data = alphas,
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
            fig = plt.figure(figsize=(8, 6))
        else:
            fig = plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        # ax.add_feature(cfeature.LAND)
        # ax.add_feature(cfeature.OCEAN)
        # ax.add_feature(cfeature.LAKES)
        # ax.add_feature(cfeature.RIVERS)

        plot = alphas_da.plot(cmap=plt.cm.plasma,
                            vmin=0, vmax=alphas.max().item(),
                            transform=ccrs.PlateCarree(),
                            cbar_kwargs={'shrink': 0.6})
        plt.title(f"alphas <scaling={self.alphas_scaling.cpu().item():2.2f}>")
        plt.tight_layout()
        file_name = 'alphas'
        if detailed:
            plt.savefig(os.path.join(output_dir, f'{file_name}.png'))
            plt.savefig(os.path.join(output_dir, f'{file_name}.pdf'))
            plt.close()
            alphas_da.to_netcdf(os.path.join(output_dir, f'{file_name}.nc'))
        else:
            plt.savefig(os.path.join(output_dir, f'{file_name}.png'))
            plt.close()

        calendar = [output['latent_calendar'] for output in outputs]
        calendar = torch.cat(calendar, dim=0)
        calendar = calendar.to(next(self.parameters()).device)

        weights = self.get_forecast_fusion_weights(calendar)
        weights = weights.cpu().numpy()
        plt.plot(weights[:, 0].mean(axis=0), label='mean')
        plt.plot(weights[:, 0].std(axis=0), label='std')
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'forecast_fusion_weights.png'))
        plt.close()
        if detailed:
            np.save(os.path.join(output_dir, 'forecast_fusion_weights.npy'), weights)
