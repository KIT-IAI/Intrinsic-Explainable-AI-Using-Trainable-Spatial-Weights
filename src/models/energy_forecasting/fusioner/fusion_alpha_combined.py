import os
import glob
from typing import Optional, Tuple, List

import numpy as np
import xarray as xr
import torch
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt


class CombinedAlphaFusioner(torch.nn.Module):
    """
    Combined Alpha Fusion subnetwork to learn trainable alphas
    to use for weighted average to combine spatial forecasts.
    """

    def __init__(self,
                 forecast_horizon: int,
                 upper_left: Tuple[float, float],
                 lower_right: Tuple[float, float],
                 scaling: float,
                 dropout: float = 0.0):
        """Initialize fusion subnetwork to forecast based on encoded inputs.

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
        alphas = torch.zeros((np.prod(self.weather_shape), ))
        self.alphas = torch.nn.Parameter(alphas)
        self.alphas_dropout = torch.nn.Dropout(p=self.dropout)

        self.weather_encoding_net = torch.nn.Sequential(
            torch.nn.LazyLinear(4 * self.forecast_horizon),
            torch.nn.LayerNorm(4 * self.forecast_horizon),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * self.forecast_horizon, 2 * self.forecast_horizon),
            torch.nn.LayerNorm(2 * self.forecast_horizon),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * self.forecast_horizon, self.forecast_horizon),
        )

        self.forecast_net = torch.nn.Sequential(
            torch.nn.LazyLinear(4 * self.forecast_horizon),
            # torch.nn.BatchNorm1d(np.prod(self.weather_shape), affine=True),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * self.forecast_horizon, 2 * self.forecast_horizon),
            # torch.nn.BatchNorm1d(np.prod(self.weather_shape), affine=True),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * self.forecast_horizon, self.forecast_horizon),
        )

    def get_alphas(self, prediction=False):
        if prediction:
            alphas = self.alphas
        else:
            alphas = self.alphas_dropout(self.alphas)
        alphas = self.alphas_scaling * alphas
        alphas = alphas.softmax(dim=0)
        alphas = alphas.view(*self.weather_shape)
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

        num_regions = np.prod(self.weather_shape)
        batch_size = weather.shape[0]
        weather_horizon = weather.shape[4]
        weather_flattened = weather.view(batch_size, num_regions, weather_horizon)

        energy_expanded = energy[:, 0].unsqueeze(1).expand(-1, num_regions, -1)
        weather_encoded = self.weather_encoding_net(weather_flattened)
        calendar_expanded = calendar[:, 0].unsqueeze(1).expand(-1, num_regions, -1)
        region_features = torch.cat([energy_expanded, weather_encoded, calendar_expanded], dim=2)
        forecasts = self.forecast_net(region_features)
        forecast = forecasts.permute(0, 2, 1)  # Back to [batch, forecast_horizon, num_regions]


        alphas = self.get_alphas(prediction)
        forecast = forecast.view(*forecast.shape[:2], *alphas.shape)
        forecast = forecast * alphas
        forecast = forecast.flatten(start_dim=2).sum(dim=2)
        capacity = self.get_capacity(input[2][:, 0, 0])
        forecast = capacity.expand(-1, self.forecast_horizon) * forecast

        # return.shape = [BATCH, TARGET, FORECAST, FORECAST_HORIZON]
        return forecast[:, None, None]

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
        weather_forecast = self.forecast_net(collected_input) \
                               .swapaxes(1, 2) \
                               .reshape(collected_input.shape[0],
                                     weather.shape[1],
                                     self.forecast_horizon,
                                     *weather.shape[-2:])

        ####
        ## Aggregate grid level forecasts using weather_alphas
        ###
        alphas = self.get_alphas(prediction)
        # transform alphas shape from [LAT, LONG] to [BATCH, LAT, LONG]
        # batch_size = energy.shape[0]
        # alphas = alphas.repeat(batch_size, 1, 1)
        # reshape alphas from [BATCH, LAT, LONG] to [BATCH, TARGET, FORECAST_HORIZON, LAT, LONG]
        # alphas = alphas[:, None, None].repeat(1, 1, self.forecast_horizon, 1, 1)
        # weather_forecast shape is [BATCH, TARGET, FORECAST_HORIZON, LAT, LONG]
        forecast = weather_forecast * alphas
        forecast = forecast.flatten(start_dim=3).sum(dim=3)
        capacity = self.get_capacity(input[2][:, 0, 0])
        forecast = capacity[:, :, None].repeat(1, 1, self.forecast_horizon) * forecast

        # return.shape = [BATCH, TARGET, FORECAST, FORECAST_HORIZON]
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

        alphas = self.get_alphas(prediction=True)
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

        if hasattr(self, 'forecast_fusion_weights'):
            weights = (10 * self.forecast_fusion_weights).sigmoid()
            weights = weights.cpu().numpy()
            plt.plot(weights)
            plt.plot(1 - weights)
            plt.savefig(os.path.join(output_dir, 'forecast_fusion_weights.png'))
            plt.close()
            if detailed:
                np.save(os.path.join(output_dir, 'forecast_fusion_weights.npy'), weights)
