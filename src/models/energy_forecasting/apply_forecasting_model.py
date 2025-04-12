from typing import Any, Dict, List, Optional, Tuple

import xarray as xr

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.tuner import Tuner

from omegaconf import DictConfig
import hydra

from src.utils import (
    RankedLogger,
    instantiate_callbacks,
    instantiate_loggers
)
from .forecasting_dataset import ForecastingDataset


log = RankedLogger(__name__, rank_zero_only=True)


def apply_forecasting_model(cfg: DictConfig,
                            train: ForecastingDataset,
                            val: ForecastingDataset,
                            test: Optional[ForecastingDataset] = None
                            ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Applying energy forecasting models.

    :param cfg: Hydra run config.
    :param train: Forecasting train dataset.
    :param val: Forecasting validation dataset.
    :param test: Forecasting test dataset, defaults to None.
    :return: Results of the run in an object dict.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model.eval()
    with torch.no_grad():
        sample_loader = DataLoader(val, batch_size=cfg.batch_size)
        x, y = next(iter(sample_loader))
        model(x)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get('callbacks'))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get('logger'))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    if cfg.get("train"):
        log.info("Starting training!")
        train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=True)
        val_loader = DataLoader(val, batch_size=cfg.batch_size,
                                num_workers=cfg.num_workers, pin_memory=True)

        if cfg.lr_finder:
            # Run learning rate finder
            tuner = Tuner(trainer)
            lr_finder = tuner.lr_find(model,
                                      min_lr=cfg.lr_finder_min,
                                      max_lr=cfg.lr_finder_max,
                                      num_training=cfg.lr_finder_steps,
                                      train_dataloaders=train_loader)

        trainer.fit(model=model, ckpt_path=cfg.get("ckpt_path"),
                    train_dataloaders=train_loader, val_dataloaders=val_loader)

    if cfg.get("test"):
        log.info("Starting testing!")

        train_loader = DataLoader(train, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        val_loader = DataLoader(val, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        test_loader = DataLoader(test, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        else:
            log.info('Loading model from checkpoint path.')
            state_dict = torch.load(ckpt_path)['state_dict']
            model.load_state_dict(state_dict)
        log.info(f"Best ckpt path: {ckpt_path}")

        train.is_test = True
        train_results = trainer.predict(model, train_loader, ckpt_path=ckpt_path)
        train_results = [train_result['y_hat'] for train_result in train_results]
        train_results = torch.concatenate(train_results).cpu().numpy()
        train.is_test = False

        val.is_test = True
        val_results = trainer.predict(model, val_loader, ckpt_path=ckpt_path)
        val_results = [val_result['y_hat'] for val_result in val_results]
        val_results = torch.concatenate(val_results).cpu().numpy()
        val.is_test = False

        test_results = trainer.predict(model, test_loader, ckpt_path=ckpt_path)
        test_results = [test_result['y_hat'] for test_result in test_results]
        test_results = torch.concatenate(test_results).cpu().numpy()

    object_dict = {
        'cfg': cfg,
        'model': model,
        'callbacks': callbacks,
        'logger': logger,
        'trainer': trainer,
        'train': xr.Dataset(
            data_vars=dict(
                [(
                    target.replace('_target', ''),
                    (['time', 'n_forecast', 'forecast'], train_results[:, i])
                 )
                 for i, target in enumerate(cfg.targets)]
            ),
            coords= {
                'time': train.energy.time,
                'n_forecast': range(train_results.shape[2]),
                'forecast': range(1, cfg.forecast_horizon + 1)
            }
        ),
        'val': xr.Dataset(
            data_vars=dict(
                [(
                    target.replace('_target', ''),
                    (['time', 'n_forecast', 'forecast'], val_results[:, i])
                 )
                 for i, target in enumerate(cfg.targets)]
            ),
            coords= {
                'time': val.energy.time,
                'n_forecast': range(val_results.shape[2]),
                'forecast': range(1, cfg.forecast_horizon + 1)
            }
        ),
        'test': xr.Dataset(
            data_vars=dict(
                [(
                    target.replace('_target', ''),
                    (['time', 'n_forecast', 'forecast'], test_results[:, i])
                 )
                 for i, target in enumerate(cfg.targets)]
            ),
            coords= {
                'time': test.energy.time,
                'n_forecast': range(test_results.shape[2]),
                'forecast': range(1, cfg.forecast_horizon + 1)
            }
        )
    }

    return object_dict