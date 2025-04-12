import os
import json
from typing import Any, Dict

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from src.utils import pylogger


log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_object_dict(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - 'cfg': A DictConfig object containing the main config.
        - 'model': The Lightning model.
        - 'trainer': The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict['cfg'])
    model = object_dict['model']
    trainer = object_dict['trainer']

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams['model'] = cfg['model']

    # save number of model parameters
    hparams['model/params/total'] = sum(p.numel() for p in model.parameters())
    hparams['model/params/trainable'] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams['model/params/non_trainable'] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams['data'] = cfg['data']
    hparams['trainer'] = cfg['trainer']

    hparams['callbacks'] = cfg.get('callbacks')
    hparams['extras'] = cfg.get('extras')

    hparams['task_name'] = cfg.get('task_name')
    hparams['tags'] = cfg.get('tags')
    hparams['ckpt_path'] = cfg.get('ckpt_path')
    hparams['seed'] = cfg.get('seed')
    hparams['output_dir'] = object_dict['cfg'].paths['output_dir']

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)

    output_dir = object_dict['cfg'].paths.output_dir

    for key in ['train', 'val', 'test']:
        if key in object_dict:
            data = object_dict[key]
            data.to_netcdf(os.path.join(output_dir, f'{key}.nc'))

    metrics = object_dict['metrics']
    metrics = {
        key: value
        for key, value in metrics.items()
        if 'all' in key
    }

    for logger in trainer.loggers:
        logger.log_hyperparams(metrics)

    json_path = os.path.join(output_dir, 'metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
