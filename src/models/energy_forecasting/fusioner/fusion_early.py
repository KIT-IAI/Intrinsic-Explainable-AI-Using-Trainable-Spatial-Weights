import numpy as np
import torch


class EarlyFusioner(torch.nn.Module):
    """
    Fusioner block to forecast energy using Early fusion approach
    on encoded energy, weather, and calendar features.
    """

    def __init__(self,
                 forecast_horizon: int,
                 fusion: str = 'concat',
                 use_history: bool = True):
        """Initialize early fusion subnetwork.

        Type of fusion techniques:
            - 'concat': Concat all latent encodings.
            - 'additive': Add all latent encodings together.
                Require all latent encodings have the same dimensions.
            - 'multiplicative: Multiply all latent encodings together.
                Require all latent encodings have the same dimensions.

        :param forecast_horizon: Energy forecast horizon.
        :param fusion: Fusion method to use, defaults to 'concat'.
        :param use_history: Enable historical energy encoding in fusion, defaults to True.
        """
        self.fusion = fusion
        self.forecast_horizon = forecast_horizon
        self.use_history = use_history
        super().__init__()

        self.forecast_shape = (1, self.forecast_horizon)

        self._fc_sequential = torch.nn.Sequential(
            torch.nn.LazyLinear(4 * self.forecast_horizon),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * self.forecast_horizon, 2 * self.forecast_horizon),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * self.forecast_horizon, np.prod(self.forecast_shape)),
        )

    def forward(self, latent_energy, latent_weather, latent_calendar, **kwargs):
        # assume shape is [BATCH, TARGET, FEATURES]
        assert len(latent_energy.shape) == 3
        assert len(latent_weather.shape) == 3
        assert len(latent_calendar.shape) == 3

        if self.use_history:
            input = torch.concat([
                latent_energy,
                latent_weather,
                latent_calendar
            ], axis=2)
        else:
            input = torch.concat([
                latent_weather,
                latent_calendar
            ], axis=2)

        if self.fusion == 'additive':
            input = input.sum(axis=2)
        elif self.fusion == 'multiplicative':
            input = input.prod(axis=2)

        output = self._fc_sequential(input)
        output = output.view(-1, *self.forecast_shape)
        return output[:, None]

    def debug(self, outputs=None, output_dir='.', detailed=False):
        pass
