from typing import Tuple

import numpy as np
import xarray as xr

from ..base_dataset import BaseDataset

class EnergyDataset(BaseDataset):
    """
    Energy dataset base class to ensure inverse_normalize_energy for y_hat.
    """

    def inverse_normalize_energy(self, ds: xr.DataArray) -> xr.DataArray:
        raise NotImplementedError("Energy datasets should implement inverse_normalize_energy.")
