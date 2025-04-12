import os
import json
from typing import Optional

import numpy as np
import pandas as pd

from omegaconf import OmegaConf, DictConfig
import hydra

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
    log_object_dict
)
from src.models.energy_forecasting import (
    ForecastingDataset,
    apply_forecasting_model,
    get_metrics
)


log = RankedLogger(__name__, rank_zero_only=True)


def set_omegaconf_operators(cfg: DictConfig):
    """Define OmegaConf config operators. 

    This includes:
        - 'add'
        - 'mul'
        - 'get_weather_targets'
        - 'get_coords'

    :param cfg: Hydra configuration dict.
    """
    OmegaConf.register_new_resolver('add', lambda *numbers: sum(numbers))
    OmegaConf.register_new_resolver('mul', lambda *numbers: np.prod(numbers))

    def get_weather_targets(targets, weather_vars):
        energy_classes = [target.split('+')[1] for target in targets]
        weather_targets = [weather_vars[energy_class] for energy_class in energy_classes]
        weather_targets = np.unique(np.array(weather_targets)).tolist()
        return weather_targets
    OmegaConf.register_new_resolver('get_weather_targets', get_weather_targets)

    def get_coords(x):
        json_path = os.path.join(cfg.paths.ecmwf_data_dir, 'areas.json')
        with open(json_path, 'r') as json_file:
            geo_json = json.load(json_file)
        longitudes = list()
        latitudes = list()
        for target in cfg.targets:
            area, _ = target.split('+')
            longitudes.extend(geo_json[area]['longitude'])
            latitudes.extend(geo_json[area]['latitude'])
        upper_left = (min(longitudes), max(latitudes))
        lower_right = (max(longitudes), min(latitudes))
        return (upper_left, lower_right)[x]
    OmegaConf.register_new_resolver('get_coords', get_coords)


def get_pandas_dataframe(energy_ds, weather_ds, calendar_ds, is_test):
    assert len(energy_ds) == len(weather_ds) == len(calendar_ds)
    for i in range(len(energy_ds)):
        if is_test:
            energy = energy_ds[i]
        else:
            energy, y = energy_ds[i]
        weather = weather_ds[i]
        calendar = calendar_ds[i]


@hydra.main(version_base='1.3', config_path='../configs', config_name='run.yaml')
def main(cfg: DictConfig):
    """Main entry point for energy forecasting pipeline using spatial weather data.

    :param cfg: Hydra configuration dict passed automatically by @hydra.main().
    """
    set_omegaconf_operators(cfg)

    # this pipeline only supports one forecasting target
    assert len(cfg.targets) == 1

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)


    log.info(f"Creating datasets.")
    # Load train datasets
    train_time = pd.date_range(cfg.train_start, cfg.train_end, freq=f'{cfg.forecast_frequency}h')
    train_energy_ds = hydra.utils.instantiate(cfg.data.energy)(time=train_time)
    train_weather_ds = hydra.utils.instantiate(cfg.data.weather)(time=train_time)
    train_calendar_ds = hydra.utils.instantiate(cfg.data.calendar)(time=train_time)
    train_dataset = ForecastingDataset(
        train_energy_ds,
        train_weather_ds,
        train_calendar_ds,
        is_test=False
    )

    # Load validation datasets
    val_time = pd.date_range(cfg.val_start, cfg.val_end, freq=f'{cfg.forecast_frequency}h')
    val_energy_ds = hydra.utils.instantiate(cfg.data.energy)(time=val_time)
    val_weather_ds = hydra.utils.instantiate(cfg.data.weather)(time=val_time)
    val_calendar_ds = hydra.utils.instantiate(cfg.data.calendar)(time=val_time)
    val_dataset = ForecastingDataset(
        val_energy_ds,
        val_weather_ds,
        val_calendar_ds,
        is_test=False
    )

    # Load test datasets
    test_time = pd.date_range(cfg.test_start, cfg.test_end, freq=f'{cfg.forecast_frequency}h')
    test_energy_ds = hydra.utils.instantiate(cfg.data.energy)(time=test_time)
    test_weather_ds = hydra.utils.instantiate(cfg.data.weather)(time=test_time)
    test_calendar_ds = hydra.utils.instantiate(cfg.data.calendar)(time=test_time)
    test_dataset = ForecastingDataset(
        test_energy_ds,
        test_weather_ds,
        test_calendar_ds,
        is_test=True
    )

    object_dict = apply_forecasting_model(cfg, train_dataset, val_dataset, test_dataset)

    # return optimized metric
    log.info("Logging metrics, models, ...")
    datasets = {
        'train': train_dataset.energy,
        'val': val_dataset.energy,
        'test': test_dataset.energy
    }
    metric_dict = get_metrics(cfg, object_dict, datasets)

    for key in [key for key in metric_dict if 'all' in key]:
        log.info(f'{key:25} {metric_dict[key]:10.3f}')

    object_dict['metrics'] = metric_dict
    if object_dict['logger']:
        log.info("Logging hyperparameters!")
        log_object_dict(object_dict)

    log.info("Done!")


if __name__ == "__main__":
    main()
